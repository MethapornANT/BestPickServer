# ============================================================
# appai.py (DROP-IN REPLACEMENT)
# โครงเดิม แต่ย้าย logic หลักจาก AI_recomendation.py เข้ามา
# + เพิ่มชุด Evaluation/Tuning ให้จบในไฟล์เดียว (run: python appai.py --mode FAST|FULL)
# ============================================================

import os
import time
import gc
import hashlib
import warnings
import argparse
import json
from datetime import datetime
from typing import List, Tuple, Optional, Dict
from scipy.sparse import csr_matrix
import numpy as np
import pandas as pd
from sqlalchemy import create_engine

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

from surprise import SVD, Dataset, Reader
import joblib
import threading

from scipy.sparse import csr_matrix, issparse, load_npz, save_npz

warnings.filterwarnings("ignore", category=UserWarning)

# ============== Global config (เหมือนของเดิม) ==============
DB_URI = 'mysql+mysqlconnector://root:1234@localhost/bestpick'

def load_data_from_db():
    engine = create_engine(DB_URI, pool_pre_ping=True, pool_recycle=1800, pool_size=5, max_overflow=10)
    content_based_data = pd.read_sql("SELECT * FROM contentbasedview;", con=engine)
    collaborative_data = pd.read_sql("SELECT * FROM collaborativeview;", con=engine)
    return content_based_data, collaborative_data

# ====== Utils ======
def normalize_scores(series: pd.Series):
    s = pd.to_numeric(series, errors='coerce').fillna(0.0).astype(np.float32)
    mn, mx = float(s.min()), float(s.max())
    return (s - mn) / (mx - mn + 1e-12)

def normalize_engagement(data, user_column='owner_id', engagement_column='PostEngagement'):
    data = data.copy()
    data['NormalizedEngagement'] = data.groupby(user_column)[engagement_column].transform(lambda x: normalize_scores(x))
    return data

def analyze_comments(comments_series):
    out = []
    for comment_text in comments_series:
        if pd.isna(comment_text) or str(comment_text).strip() == '':
            out.append(0)
        else:
            out.append(len([c.strip() for c in str(comment_text).split(';') if c.strip()]))
    return out

# ====== Tuned params (อิง AI_recomendation.py แต่คงชื่อเดิม) ======
_TFIDF_PARAMS = dict(analyzer='char_wb', ngram_range=(2, 5), max_features=60000, min_df=2, max_df=0.95, stop_words=None)
_WEIGHTS = (0.3, 0.3, 0.4)  # (collab, item_content, user_content)
_VIEW_POS_MIN = 3

# รวม mapping จากไฟล์ใหม่ แต่คงชื่อและเติม keys เดิมให้ครบ
_ACTION_WEIGHT = {
    'view': 1.0,
    'like': 4.0,
    'comment': 4.0,
    'bookmark': 4.5,
    'share': 5.0,
    'unlike': -3.0,
    'unbookmark': -2.0
}
_RATING_MIN, _RATING_MAX = 0.5, 5.0
_POP_PRIOR_W = 0.05
_POP_ALPHA = 5.0  # Bayesian prior pseudo-counts สำหรับ PopularityPrior

# ============ Safe fallbacks (กันโปรเจ็กต์ที่ไม่ได้ประกาศของพวกนี้) ============
try:
    CACHE_EXPIRY_TIME_SECONDS
except NameError:
    CACHE_EXPIRY_TIME_SECONDS = 600

try:
    _cache_lock
except NameError:
    _cache_lock = threading.Lock()

try:
    recommendation_cache
except NameError:
    recommendation_cache = {}

try:
    impression_history_cache
except NameError:
    impression_history_cache = {}

try:
    IMPRESSION_HISTORY_MAX_ENTRIES
except NameError:
    IMPRESSION_HISTORY_MAX_ENTRIES = 500

if 'split_and_rank_recommendations' not in globals():
    # fallback ที่ไม่ทำอะไร นอกจากคืนลิสต์เดิม
    def split_and_rank_recommendations(recommendations, user_interactions, impression_history, total_posts_in_db):
        return recommendations

# ==================== Content-based MODULE ====================
class ContentBased:
    """
    - TF-IDF + KNN บน Content (char n-gram รองรับไทย/อังกฤษ)
    - item_content_scores: ค่าเฉลี่ย NormalizedEngagement ของเพื่อนบ้านใกล้เคียง
    - user text profile: mean TF-IDF ของโพสต์ที่ user เคย positive
    """
    def __init__(self, tfidf_params: dict):
        self.tfidf_params = tfidf_params
        self.tfidf = None
        self.knn = None
        self.X = None
        self.item_content_scores = None
        self.pid_to_idx = {}

    def fit(self, content_df: pd.DataFrame):
        if 'post_id' not in content_df.columns and 'id' in content_df.columns:
            content_df = content_df.rename(columns={'id':'post_id'})
        if 'post_id' not in content_df.columns:
            raise ValueError('content_df ต้องมีคอลัมน์ post_id')

        # ensure text & engagement columns
        if 'Content' not in content_df.columns:
            content_df['Content'] = ''
        if 'NormalizedEngagement' not in content_df.columns:
            # สร้าง NormalizedEngagement แบบ global ถ้าไม่มี
            content_df['NormalizedEngagement'] = normalize_scores(pd.to_numeric(content_df.get('PostEngagement', 0.0), errors='coerce').fillna(0.0))

        # TF-IDF
        self.tfidf = TfidfVectorizer(**self.tfidf_params, dtype=np.float32)
        self.X = self.tfidf.fit_transform(content_df['Content'].fillna('')).astype(np.float32)

        # KNN
        self.knn = NearestNeighbors(n_neighbors=10, metric='cosine').fit(self.X)

        # precompute item_content_scores = mean engagement ของเพื่อนบ้าน
        n = self.X.shape[0]
        if n > 0:
            n_neighbors = min(20, n)
            dists, idxs = self.knn.kneighbors(self.X, n_neighbors=n_neighbors)
            eng = pd.to_numeric(content_df['NormalizedEngagement'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
            scores = np.zeros(n, dtype=np.float32)
            for i in range(n):
                scores[i] = float(np.mean(eng[idxs[i]])) if idxs[i].size else 0.0
            self.item_content_scores = scores
        else:
            self.item_content_scores = np.zeros(0, dtype=np.float32)

        self.pid_to_idx = {int(pid): i for i, pid in enumerate(content_df['post_id'].tolist())}

    def build_user_profiles(self, labels_df: pd.DataFrame, content_df: pd.DataFrame) -> dict:
        if not {'user_id','post_id'}.issubset(labels_df.columns):
            raise ValueError('labels_df ต้องมีคอลัมน์ user_id และ post_id')

        if 'y' in labels_df.columns:
            pos_df = labels_df[labels_df['y'] == 1][['user_id','post_id']]
        else:
            pos_df = labels_df[['user_id','post_id']].copy()

        profiles = {}
        for uid, g in pos_df.groupby('user_id'):
            idxs = [self.pid_to_idx.get(int(p)) for p in g['post_id'].tolist()
                    if self.pid_to_idx.get(int(p)) is not None]
            if not idxs:
                # โปรไฟล์ว่างเป็นเวคเตอร์ศูนย์ (sparse) ป้องกันคูณแล้ว error
                profiles[int(uid)] = csr_matrix((1, self.X.shape[1]), dtype=np.float32)
                continue

            m = self.X[idxs]
            mean_vec = m.mean(axis=0)
            # แปลงเป็น dense แล้ว normalize จากนั้นห่อกลับเป็น CSR ให้คูณกับ sparse ได้
            mean_vec = mean_vec.toarray() if hasattr(mean_vec, 'toarray') else np.asarray(mean_vec)
            prof = sk_normalize(mean_vec)  # shape (1, dim)
            profiles[int(uid)] = csr_matrix(prof, dtype=np.float32)

        return profiles


    def user_content_score(self, user_id: int, profiles: dict, row_idx: int) -> float:
        prof = profiles.get(int(user_id))
        if prof is None:
            return 0.0
        v = self.X[row_idx]
        num = float(v.multiply(prof).sum()) if hasattr(v, 'multiply') else float(np.dot(v, prof.T))
        den = (np.linalg.norm(getattr(v, 'data', np.array([]))) * np.linalg.norm(getattr(prof, 'data', np.array([]))))
        return float(num/den) if den else 0.0

# ==================== Collaborative MODULE ====================
class Collaborative:
    """
    - แปลง collaborativeview (wide) → implicit rating
    - เทรน SVD (เร็วและนิ่งใน sparse)
    - ไม่มีข้อมูลพอ -> Dummy 0.5
    """
    class _Dummy:
        def predict(self, uid, iid):
            class Est: est = 0.5
            return Est()

    def __init__(self):
        self.model = None

    def fit(self, collab_df: pd.DataFrame):
        if collab_df.empty or not {'user_id','post_id'}.issubset(collab_df.columns):
            self.model = Collaborative._Dummy();  return

        # รองรับ both: wide numeric cols หรือ log แบบ melt
        num_cols = [c for c in collab_df.columns if c not in ('user_id','post_id') and pd.api.types.is_numeric_dtype(collab_df[c])]
        if num_cols:
            melted = collab_df.melt(id_vars=['user_id','post_id'], var_name='action_type', value_name='cnt')
            melted['action_type'] = melted['action_type'].astype(str).str.lower()
            t = melted.groupby(['user_id','post_id','action_type'])['cnt'].sum().reset_index()
        else:
            t = collab_df.groupby(['user_id','post_id']).size().reset_index(name='cnt')
            t['action_type'] = 'view'

        pvt = t.pivot_table(index=['user_id','post_id'], columns='action_type', values='cnt',
                            fill_value=0, aggfunc='sum').reset_index()

        rating = np.zeros(len(pvt), dtype=np.float32)
        for act, w in _ACTION_WEIGHT.items():
            if act in pvt.columns:
                rating += np.float32(w) * pvt[act].to_numpy(dtype=np.float32)
        if 'view' in pvt.columns:
            rating += np.where(pvt['view'].to_numpy(dtype=np.float32) >= _VIEW_POS_MIN, np.float32(2.0), np.float32(0.0))

        rating = np.clip(rating, _RATING_MIN, _RATING_MAX)
        data = pvt[['user_id','post_id']].copy()
        data['rating'] = rating
        data = data[data['rating'] > 0]
        if data.empty:
            self.model = Collaborative._Dummy();  return

        reader = Reader(rating_scale=(_RATING_MIN, _RATING_MAX))
        dset = Dataset.load_from_df(data[['user_id','post_id','rating']], reader)
        trainset = dset.build_full_trainset()
        self.model = SVD(n_factors=150, n_epochs=60, lr_all=0.005, reg_all=0.5)
        self.model.fit(trainset)

    def predict(self, user_id: int, post_id: int) -> float:
        if self.model is None:
            return 0.5
        try:
            return float(self.model.predict(int(user_id), int(post_id)).est)
        except Exception:
            return 0.5

# ==================== Hybrid RANKER ====================
class HybridRanker:
    """
    final_score = wc*collab + wi*item_content + wu*user_content + pop_prior
    กัน owner=self ขึ้นหัวลิสต์
    """
    def __init__(self, content_mod: ContentBased, collab_mod: Collaborative, weights=_WEIGHTS, pop_w=_POP_PRIOR_W):
        self.cb = content_mod
        self.cf = collab_mod
        self.wc, self.wi, self.wu = weights
        self.pop_w = pop_w

    def rank(self, user_id: int, content_df: pd.DataFrame, user_profiles: dict, top_k: int = 50) -> list:
        if content_df.empty:
            return []

        # popularity prior จาก Bayesian smoothing ถ้ามี ไม่งั้น fallback เป็น normalized engagement
        if 'PopularityPrior' in content_df.columns:
            pop_arr = pd.to_numeric(content_df['PopularityPrior'], errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
        else:
            pop_arr = normalize_scores(pd.to_numeric(content_df.get('PostEngagement', 0.0), errors='coerce').fillna(0.0)).to_numpy(dtype=np.float32)

        rows = []
        for i in range(len(content_df)):
            row = content_df.iloc[i]
            pid = int(row['post_id'])
            if 'owner_id' in row and row['owner_id'] == user_id:
                continue
            cscore = self.cf.predict(user_id, pid)
            ics = float(self.cb.item_content_scores[i]) if self.cb.item_content_scores is not None and i < len(self.cb.item_content_scores) else 0.0
            ucs = self.cb.user_content_score(user_id, profiles=user_profiles, row_idx=i)
            pop = float(pop_arr[i])
            final = self.wc*cscore + self.wi*ics + self.wu*ucs + self.pop_w*pop
            rows.append((pid, final))

        if not rows:
            return []
        df = pd.DataFrame(rows, columns=['post_id','score'])
        df['norm'] = normalize_scores(df['score'])
        return df.sort_values(['norm','score'], ascending=[False, False])['post_id'].head(max(top_k*3, top_k)).tolist()

# ==================== Internal state & builders ====================
_model_cache = {'content_df': None, 'collab_df': None, 'content_mod': None, 'collab_mod': None, 'hybrid': None, 'profiles': {}, 'timestamp': 0}

def _load_views():
    engine = create_engine(DB_URI, pool_pre_ping=True, pool_recycle=1800, pool_size=5, max_overflow=10)
    content_df = pd.read_sql("SELECT * FROM contentbasedview;", con=engine)
    if 'post_id' not in content_df.columns and 'id' in content_df.columns:
        content_df = content_df.rename(columns={'id':'post_id'})
    content_df['post_id'] = pd.to_numeric(content_df['post_id'], errors='coerce').dropna().astype(int)
    if 'Content' not in content_df.columns:
        content_df['Content'] = ''
    # สร้าง PopularityPrior + NormalizedEngagement แบบ Bayesian smoothing
    eng = pd.to_numeric(content_df.get('PostEngagement', 0.0), errors='coerce').fillna(0.0).astype(np.float32)
    prior = (eng + _POP_ALPHA) / (float(eng.max()) + _POP_ALPHA if float(eng.max()) > 0 else (_POP_ALPHA))
    content_df['PopularityPrior'] = normalize_scores(prior)
    content_df['NormalizedEngagement'] = normalize_scores(eng)
    if 'owner_id' not in content_df.columns: content_df['owner_id'] = -1

    collab_df = pd.read_sql("SELECT * FROM collaborativeview;", con=engine)
    return content_df, collab_df

def _labels_from_collabview(collab_df: pd.DataFrame) -> pd.DataFrame:
    # เบาสุด: ถือเป็น 'view' แล้วใช้ threshold (คงสัญญาณตามโค้ดเดิม)
    if collab_df.empty or not {'user_id','post_id'}.issubset(collab_df.columns):
        return pd.DataFrame(columns=['user_id','post_id','y'])
    t = collab_df.groupby(['user_id','post_id']).size().reset_index(name='cnt')
    t['action_type'] = 'view'
    pvt = t.pivot_table(index=['user_id','post_id'], columns='action_type', values='cnt', fill_value=0).reset_index()
    pvt.columns = [str(c).lower() for c in pvt.columns]
    pos = np.zeros(len(pvt), dtype=bool)
    if 'view' in pvt.columns:
        pos |= (pvt['view'].to_numpy(dtype=float) >= _VIEW_POS_MIN)
    pvt['y'] = pos.astype(int)
    return pvt[['user_id','post_id','y']]

def _ensure_models_fresh():
    now = time.time()
    if _model_cache['timestamp'] and now - _model_cache['timestamp'] < CACHE_EXPIRY_TIME_SECONDS:
        return
    content_df, collab_df = _load_views()

    cb = ContentBased(_TFIDF_PARAMS)
    cb.fit(content_df)

    labels = _labels_from_collabview(collab_df)
    profiles = cb.build_user_profiles(labels, content_df)

    cf = Collaborative()
    cf.fit(collab_df)

    hy = HybridRanker(cb, cf, weights=_WEIGHTS, pop_w=_POP_PRIOR_W)

    _model_cache.update({
        'content_df': content_df,
        'collab_df': collab_df,
        'content_mod': cb,
        'collab_mod': cf,
        'hybrid': hy,
        'profiles': profiles,
        'timestamp': now
    })

# ==================== PUBLIC API: ใช้ใน endpoint ====================
def get_recommendations_for_user(user_id: int, user_interactions: list, total_posts_in_db: int, top_k: int = 20) -> list:
    """
    คืน list[post_id] พร้อมใช้งาน:
      1) สร้าง/โหลดโมเดลในหน่วยความจำตาม TTL
      2) Hybrid rank
      3) ผสมกับ split_and_rank_recommendations ของระบบเดิม
      4) อัพเดต impression_history_cache
    """
    _ensure_models_fresh()
    content_df = _model_cache['content_df']
    hybrid = _model_cache['hybrid']
    profiles = _model_cache['profiles']

    now = datetime.now()
    with _cache_lock:
        cached = recommendation_cache.get(user_id)
        if cached and (now - cached['timestamp']).total_seconds() < CACHE_EXPIRY_TIME_SECONDS:
            base_list = cached['base']
        else:
            base_list = hybrid.rank(int(user_id), content_df, profiles, top_k=top_k)
            recommendation_cache[user_id] = {'base': base_list, 'timestamp': now}

        history = impression_history_cache.get(user_id, [])
        ordered = split_and_rank_recommendations(
            recommendations=base_list,
            user_interactions=user_interactions or [],
            impression_history=history,
            total_posts_in_db=total_posts_in_db
        )
        new_entries = [{'post_id': pid, 'timestamp': now} for pid in ordered[:top_k]]
        impression_history_cache[user_id] = (history + new_entries)[-IMPRESSION_HISTORY_MAX_ENTRIES:]

    return ordered[:top_k]

# ==================== Legacy-compat (คงชื่อเดิมให้เรียกได้) ====================
def create_content_based_model(data, text_column='Content', comment_column='Comments', engagement_column='PostEngagement'):
    # train TF-IDF + KNN + feature ชุดเล็ก (สำหรับโค้ดเก่า)
    data = data.copy()
    if text_column not in data.columns: data[text_column] = ''
    if engagement_column not in data.columns: data[engagement_column] = 0.0
    tfidf = TfidfVectorizer(**_TFIDF_PARAMS, dtype=np.float32)
    tfidf_matrix = tfidf.fit_transform(data[text_column].fillna('')).astype(np.float32)
    knn = NearestNeighbors(n_neighbors=10, metric='cosine').fit(tfidf_matrix)
    joblib.dump(tfidf, 'TFIDF_Model.pkl', compress=3)
    joblib.dump(knn, 'KNN_Model.pkl', compress=3)
    # เพิ่มคอลัมน์ช่วย
    data['CommentCount'] = analyze_comments(data.get(comment_column, pd.Series('', index=data.index)))
    data = normalize_engagement(data, engagement_column=engagement_column)
    data['WeightedEngagement'] = normalize_scores(data[engagement_column]) + normalize_scores(data['CommentCount'])
    return tfidf, knn, data, pd.DataFrame()  # test_data ไม่จำเป็นในโปรดักชัน

def create_collaborative_model(data, n_factors=150, n_epochs=60, lr_all=0.005, reg_all=0.5):
    # สร้าง SVD จาก implicit rating (สำหรับโค้ดเก่า)
    if data.empty or not {'user_id','post_id'}.issubset(data.columns):
        class DummySVD:
            def predict(self, uid, iid):
                class Est: est = 0.5
                return Est()
        model = DummySVD()
        joblib.dump(model, 'Collaborative_Model.pkl', compress=3)
        return model, pd.DataFrame()

    melted = data.melt(id_vars=['user_id','post_id'], var_name='action_type', value_name='cnt')
    melted['action_type'] = melted['action_type'].astype(str).lower()
    pvt = melted.pivot_table(index=['user_id','post_id'], columns='action_type', values='cnt',
                             fill_value=0, aggfunc='sum').reset_index()

    rating = np.zeros(len(pvt), dtype=np.float32)
    for act, w in _ACTION_WEIGHT.items():
        if act in pvt.columns:
            rating += np.float32(w) * pvt[act].to_numpy(dtype=np.float32)
    if 'view' in pvt.columns:
        rating += np.where(pvt['view'].to_numpy(dtype=np.float32) >= _VIEW_POS_MIN, np.float32(2.0), np.float32(0.0))
    rating = np.clip(rating, _RATING_MIN, _RATING_MAX)
    df = pvt[['user_id','post_id']].copy(); df['rating'] = rating
    df = df[df['rating'] > 0]
    if df.empty:
        class DummySVD:
            def predict(self, uid, iid):
                class Est: est = 0.5
                return Est()
        model = DummySVD()
        joblib.dump(model, 'Collaborative_Model.pkl', compress=3)
        return model, pd.DataFrame()

    reader = Reader(rating_scale=(_RATING_MIN, _RATING_MAX))
    dset = Dataset.load_from_df(df[['user_id','post_id','rating']], reader)
    trainset = dset.build_full_trainset()
    model = SVD(n_factors=n_factors, n_epochs=n_epochs, lr_all=lr_all, reg_all=reg_all)
    model.fit(trainset)
    joblib.dump(model, 'Collaborative_Model.pkl', compress=3)
    return model, pd.DataFrame()

def recommend_hybrid(user_id, all_posts_data, collaborative_model, knn, tfidf, categories=None, alpha=0.50, beta=0.20):
    # เวอร์ชันรองรับโค้ดเก่า: content score = เพื่อนบ้านเฉลี่ยของ PostEngagement
    categories = categories or []
    has_knn = hasattr(knn, "_fit_X") and getattr(knn, "_fit_X") is not None and getattr(knn, "_fit_X").shape[0] > 0
    has_tfidf = hasattr(tfidf, "vocabulary_") and tfidf.vocabulary_ is not None
    df = all_posts_data.copy()
    if 'NormalizedEngagement' not in df.columns:
        df['NormalizedEngagement'] = normalize_scores(df.get('PostEngagement', 0.0))
    # สร้าง PopularityPrior ถ้ายังไม่มี
    if 'PopularityPrior' not in df.columns:
        eng = pd.to_numeric(df.get('PostEngagement', 0.0), errors='coerce').fillna(0.0).astype(np.float32)
        prior = (eng + _POP_ALPHA) / (float(eng.max()) + _POP_ALPHA if float(eng.max()) > 0 else (_POP_ALPHA))
        df['PopularityPrior'] = normalize_scores(prior)

    recs = []
    for _, row in df.iterrows():
        pid = int(row['post_id'])
        try:
            collab_score = float(collaborative_model.predict(int(user_id), int(pid)).est)
        except Exception:
            collab_score = 0.5
        content_score = 0.0
        if has_tfidf and has_knn:
            try:
                vec = tfidf.transform([str(row.get('Content', '') or '')]).astype(np.float32)
                n_neighbors = min(20, knn._fit_X.shape[0])
                _, idxs = knn.kneighbors(vec, n_neighbors=n_neighbors)
                if len(idxs[0]) > 0:
                    content_score = float(np.mean([df.iloc[i]['NormalizedEngagement'] for i in idxs[0]]))
            except Exception:
                content_score = 0.0
        cat_score = 0.0
        if categories:
            cnt = 0
            for c in categories:
                if c in row.index and row[c] in (1, True):
                    cnt += 1
            cat_score = cnt / len(categories) if len(categories) else 0.0
        # blend base แบบเดิม แต่โยน PopularityPrior เข้าไปนิดหน่อยให้เนียน
        final = (alpha*collab_score) + ((1-alpha)*content_score) + (beta*cat_score) + _POP_PRIOR_W*float(row['PopularityPrior'])
        recs.append((int(pid), final))
    out = pd.DataFrame(recs, columns=['post_id','score'])
    out['norm'] = normalize_scores(out['score'])
    return out.sort_values(['norm','score'], ascending=[False, False])['post_id'].tolist()

# =====================================================================
# =====================  Evaluation / Tuning  ==========================
# =====================================================================

# DB objects for evaluation
CONTENT_VIEW = 'contentbasedview'
EVENT_TABLE  = 'user_interactions'

# Eval directories
OUT_DIR  = './recsys_eval_appai'
CACHE_DIR = os.path.join(OUT_DIR, 'cache')

# Eval knobs
RUN_MODE_DEFAULT = 'FAST'                    # FAST|FULL
K_LIST = [5, 10, 20]
SPLIT_RATIOS = (0.6, 0.2, 0.2)              # train/val/test ต่อ user
TUNE_K = 10
TUNE_METRIC = 'ndcg'                        # ndcg|precision|recall|hitrate

# Labeling policy
POS_ACTIONS = {'like','comment','bookmark','share'}
NEG_ACTIONS = {'unlike'}
IGNORE_ACTIONS = {'view_profile','follow','unfollow'}

# TF-IDF & weight grids (รีใช้ของจริง)
TFIDF_FAST = _TFIDF_PARAMS
TFIDF_FULL = [
    dict(analyzer='char_wb', ngram_range=(3,5), max_features=50000, min_df=2, max_df=0.9,  stop_words=None),
    dict(analyzer='char_wb', ngram_range=(2,5), max_features=60000, min_df=2, max_df=0.95, stop_words=None),
]
WEIGHT_GRID_FAST = [(0.3, 0.2, 0.5), (0.3, 0.3, 0.4)]  # (collab, item_content, user_content)
WEIGHT_GRID_FULL = [(wc, wi, 1.0-wc-wi)
    for wc in [0.2,0.3,0.4]
    for wi in [0.2,0.3]
    if 0.1 <= 1.0-wc-wi <= 0.6]

TEXT_COL = 'Content'
ENGAGE_COL = 'PostEngagement'

def _ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def _db_connect():
    return create_engine(DB_URI, pool_pre_ping=True, pool_recycle=1800)

def _md5_of_df(df: pd.DataFrame, cols: List[str]) -> str:
    snap = df[cols].copy().fillna(0)
    return hashlib.md5(pd.util.hash_pandas_object(snap, index=False).values).hexdigest()

def _precision_at_k(rel: List[int], k: int) -> float:
    k = min(k, len(rel));  return float(np.sum(rel[:k]))/k if k else 0.0

def _recall_at_k(rel: List[int], k: int, num_pos: int) -> float:
    k = min(k, len(rel));  return float(np.sum(rel[:k]))/float(max(1,num_pos)) if k else 0.0

def _dcg_at_k(rel: List[int], k: int) -> float:
    return float(sum((2**r - 1)/np.log2(i+2) for i, r in enumerate(rel[:k])))

def _ndcg_at_k(rel: List[int], k: int) -> float:
    ideal = sorted(rel, reverse=True); idcg = _dcg_at_k(ideal, k)
    return (_dcg_at_k(rel, k)/idcg) if idcg>0 else 0.0

def _hit_at_k(rel: List[int], k: int) -> float:
    return 1.0 if any(rel[:k]) else 0.0

def _guess_ts_column(eng) -> Optional[str]:
    try:
        one = pd.read_sql(f"SELECT * FROM {EVENT_TABLE} LIMIT 1", eng)
    except Exception:
        return None
    for c in ['created_at','updated_at','ts','timestamp','event_time','inserted_at']:
        if c in one.columns: return c
    return None

def _load_content_eval(eng) -> pd.DataFrame:
    df = pd.read_sql(f"SELECT * FROM {CONTENT_VIEW}", eng)
    if 'post_id' not in df.columns and 'id' in df.columns:
        df = df.rename(columns={'id':'post_id'})
    df['post_id'] = pd.to_numeric(df['post_id'], errors='coerce').dropna().astype(int)
    if TEXT_COL not in df.columns: df[TEXT_COL] = ''
    if ENGAGE_COL not in df.columns: df[ENGAGE_COL] = 0.0
    engs = pd.to_numeric(df[ENGAGE_COL], errors='coerce').fillna(0.0).astype(np.float32)
    denom = float(engs.max()) + _POP_ALPHA if float(engs.max()) > 0 else _POP_ALPHA
    prior = (engs + _POP_ALPHA) / denom
    df['PopularityPrior'] = normalize_scores(prior)
    df['NormalizedEngagement'] = normalize_scores(engs)
    if 'owner_id' not in df.columns: df['owner_id'] = -1
    return df

def _load_events_eval(eng, ts_col: Optional[str]) -> pd.DataFrame:
    base = "user_id, post_id, action_type"
    if ts_col: base += f", {ts_col} AS ts"
    ev = pd.read_sql(f"SELECT {base} FROM {EVENT_TABLE}", eng)
    ev = ev.dropna(subset=['user_id','post_id'])
    ev['user_id'] = pd.to_numeric(ev['user_id'], errors='coerce').dropna().astype(int)
    ev['post_id'] = pd.to_numeric(ev['post_id'], errors='coerce').dropna().astype(int)
    ev['action_type'] = ev['action_type'].astype(str).str.lower()
    if 'ts' in ev.columns: ev = ev.dropna(subset=['ts'])
    ev = ev[~ev['action_type'].isin(IGNORE_ACTIONS)].copy()
    return ev

def _build_true_labels(events: pd.DataFrame) -> pd.DataFrame:
    t = events.groupby(['user_id','post_id','action_type']).size().reset_index(name='cnt')
    if t.empty: return pd.DataFrame(columns=['user_id','post_id','y'])
    pvt = t.pivot_table(index=['user_id','post_id'], columns='action_type',
                        values='cnt', fill_value=0, aggfunc='sum').reset_index()
    pvt.columns = [str(c).lower() for c in pvt.columns]
    pos = np.zeros(len(pvt), dtype=bool)
    for a in POS_ACTIONS:
        if a in pvt.columns: pos |= (pvt[a].to_numpy(dtype=float) > 0)
    if 'view' in pvt.columns:
        pos |= (pvt['view'].to_numpy(dtype=float) >= _VIEW_POS_MIN)
    if NEG_ACTIONS:
        neg = np.zeros(len(pvt), dtype=bool)
        for a in NEG_ACTIONS:
            if a in pvt.columns: neg |= (pvt[a].to_numpy(dtype=float) > 0)
        pos = np.where(neg, False, pos)
    pvt['y'] = pos.astype(int)
    return pvt[['user_id','post_id','y']]

def _split_user_tvt(events: pd.DataFrame, ts_col: Optional[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a, b, c = SPLIT_RATIOS
    ev = events.copy()
    if ts_col and 'ts' in ev.columns:
        ev = ev.sort_values(['user_id','ts'])
    else:
        ev = ev.groupby('user_id', group_keys=False).apply(lambda g: g.sample(frac=1.0, random_state=42))
    tr, va, te = [], [], []
    for uid, g in ev.groupby('user_id'):
        n = len(g)
        n_tr = max(1, int(n*a)); n_va = max(1, int(n*b)); n_te = max(1, n - n_tr - n_va)
        tr.append(g.iloc[:n_tr]); va.append(g.iloc[n_tr:n_tr+n_va]); te.append(g.iloc[n_tr+n_va:])
    return pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True), pd.concat(te, ignore_index=True)

def _recommend_scores_for_user(uid: int,
                               content_df: pd.DataFrame,
                               cb: ContentBased,
                               uc_profiles: Dict[int, csr_matrix],
                               collab: Collaborative,
                               weights: Tuple[float,float,float]) -> pd.DataFrame:
    wc, wi, wu = weights
    rows = []
    pop = pd.to_numeric(content_df.get('PopularityPrior', 0.0), errors='coerce').fillna(0.0).to_numpy(dtype=np.float32)
    for i in range(len(content_df)):
        row = content_df.iloc[i]
        pid = int(row['post_id'])
        if 'owner_id' in row and row['owner_id'] == uid:
            continue
        cscore = collab.predict(int(uid), pid) if collab is not None else 0.5
        ic = float(cb.item_content_scores[i]) if cb.item_content_scores is not None else 0.0
        uc = cb.user_content_score(uid, profiles=uc_profiles, row_idx=i)
        final = wc*cscore + wi*ic + wu*uc + _POP_PRIOR_W*float(pop[i])
        rows.append((pid, cscore, ic, uc, float(pop[i]), final))
    out = pd.DataFrame(rows, columns=['post_id','collab','item_content','user_content','pop','final'])
    out['final_norm'] = normalize_scores(out['final'])
    return out.sort_values(['final_norm','final'], ascending=[False, False])

def _train_reranker(val_lab: pd.DataFrame,
                    scores_map: Dict[int, pd.DataFrame],
                    neg_per_pos: int) -> Optional[LogisticRegression]:
    X_rows, y_rows = [], []
    for uid, g in val_lab.groupby('user_id'):
        pos_items = set(g[g['y']==1]['post_id'].tolist())
        if not pos_items: continue
        sc = scores_map.get(int(uid))
        if sc is None or sc.empty:
            continue
        pos_feat = sc[sc['post_id'].isin(pos_items)][['collab','item_content','user_content','pop','final']].values
        if len(pos_feat)==0: continue
        X_rows.append(pos_feat); y_rows += [1]*len(pos_feat)
        cand_neg = sc[~sc['post_id'].isin(pos_items)]
        if len(cand_neg) == 0: continue
        sample_n = min(len(cand_neg), max(neg_per_pos*len(pos_feat), 8))
        neg_feat = cand_neg.sample(n=sample_n, random_state=42)[['collab','item_content','user_content','pop','final']].values
        X_rows.append(neg_feat); y_rows += [0]*len(neg_feat)
    if not X_rows: return None
    Xtr = np.vstack(X_rows); ytr = np.array(y_rows, dtype=int)
    lr = LogisticRegression(max_iter=500, n_jobs=1)
    lr.fit(Xtr, ytr)
    return lr

def _apply_reranker(lr: Optional[LogisticRegression], sc: pd.DataFrame) -> List[int]:
    if lr is None or sc is None or sc.empty:
        return sc['post_id'].tolist()
    feats = sc[['collab','item_content','user_content','pop','final']].values
    sc = sc.copy()
    sc['lr_score'] = lr.predict_proba(feats)[:,1]
    return sc.sort_values('lr_score', ascending=False)['post_id'].tolist()

def _evaluate_split(content_df: pd.DataFrame,
                    train_ev: pd.DataFrame, val_ev: pd.DataFrame, test_ev: pd.DataFrame,
                    tfidf_params: dict, weights: Tuple[float,float,float],
                    mode_tag: str,
                    cache_key: str):
    # cache paths
    _ensure_dirs()
    cache_base = os.path.join(CACHE_DIR, cache_key)
    tfidf_tag = hashlib.md5(json.dumps(tfidf_params, sort_keys=True).encode()).hexdigest()
    tfidf_pkl = cache_base + f'.tfidf_{tfidf_tag}.pkl'
    X_npz     = cache_base + f'.X_{tfidf_tag}.npz'
    knn_pkl   = cache_base + '.knn.pkl'
    ics_npy   = cache_base + '.item_scores.npy'
    uprof_pkl = cache_base + '.ucprof.pkl'
    collab_pkl= cache_base + '.svd.pkl'

    # Content-based model block (fit หรือโหลด)
    cb = ContentBased(tfidf_params)
    try:
        if os.path.exists(tfidf_pkl) and os.path.exists(X_npz) and os.path.exists(knn_pkl) and os.path.exists(ics_npy):
            with open(tfidf_pkl,'rb') as f: cb.tfidf = joblib.load(f) if f else joblib.load(tfidf_pkl)
            cb.X = load_npz(X_npz).astype(np.float32)
            with open(knn_pkl,'rb') as f: cb.knn = joblib.load(f) if f else joblib.load(knn_pkl)
            cb.item_content_scores = np.load(ics_npy)
            cb.pid_to_idx = {int(pid): i for i, pid in enumerate(content_df['post_id'].tolist())}
        else:
            cb.fit(content_df)
            with open(tfidf_pkl,'wb') as f: joblib.dump(cb.tfidf, f, compress=3)
            save_npz(X_npz, cb.X)
            with open(knn_pkl,'wb') as f: joblib.dump(cb.knn, f, compress=3)
            np.save(ics_npy, cb.item_content_scores)
    except Exception:
        cb.fit(content_df)

    # Labels
    train_lab = _build_true_labels(train_ev)
    val_lab   = _build_true_labels(val_ev)
    test_lab  = _build_true_labels(test_ev)

    # User text profiles (train positives)
    try:
        if os.path.exists(uprof_pkl):
            with open(uprof_pkl,'rb') as f: uc_prof = joblib.load(f)
        else:
            train_pos = train_lab[train_lab['y']==1][['user_id','post_id']]
            uc_prof = cb.build_user_profiles(train_pos, content_df)
            with open(uprof_pkl,'wb') as f: joblib.dump(uc_prof,f, compress=3)
    except Exception:
        train_pos = train_lab[train_lab['y']==1][['user_id','post_id']]
        uc_prof = cb.build_user_profiles(train_pos, content_df)

    # Collaborative model (train only, ใช้ SVD แบบโปรดักชัน)
    try:
        if os.path.exists(collab_pkl):
            with open(collab_pkl,'rb') as f: collab = joblib.load(f)
        else:
            collab = Collaborative()
            collab.fit(train_ev)
            with open(collab_pkl,'wb') as f: joblib.dump(collab,f, compress=3)
    except Exception:
        collab = Collaborative()
        collab.fit(train_ev)

    seen_train = train_lab.groupby('user_id')['post_id'].apply(set).to_dict()

    def eval_part(labels_df: pd.DataFrame, part: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
        rows = []; per_user_scores = {}
        users = sorted(labels_df[labels_df['y']==1]['user_id'].unique().tolist())
        pos_map = labels_df[labels_df['y']==1].groupby('user_id')['post_id'].apply(set).to_dict()
        for uid in users:
            pos = pos_map.get(uid, set())
            if not pos: continue
            sc = _recommend_scores_for_user(int(uid), content_df, cb, uc_prof, collab, weights)
            seen = seen_train.get(uid, set())
            sc = sc[~sc['post_id'].isin(seen)]
            per_user_scores[int(uid)] = sc
            ranking = sc['post_id'].tolist()
            rel = [1 if p in pos else 0 for p in ranking]
            num_pos = len(pos)
            for K in K_LIST:
                rows.append({
                    'mode': part, 'K': K,
                    'precision': _precision_at_k(rel, K),
                    'recall':    _recall_at_k(rel, K, num_pos),
                    'ndcg':      _ndcg_at_k(rel, K),
                    'hitrate':   _hit_at_k(rel, K),
                    'num_pos':   num_pos,
                    'num_candidates': len(ranking)
                })
        df = pd.DataFrame(rows)
        macro = df.groupby('K').mean(numeric_only=True).reset_index() if len(df) else pd.DataFrame()
        return df, macro, per_user_scores

    val_df, val_macro, val_scores = eval_part(val_lab, f'{mode_tag}_val')
    test_df, test_macro, test_scores = eval_part(test_lab, f'{mode_tag}_test')

    return (pd.concat([val_df, test_df], ignore_index=True) if len(val_df) or len(test_df) else pd.DataFrame(),
            pd.concat([val_macro.assign(split='val'), test_macro.assign(split='test')], ignore_index=True) if len(val_macro) or len(test_macro) else pd.DataFrame(),
            val_lab, test_lab, val_scores, test_scores)

def train_tune_eval(run_mode: str):
    _ensure_dirs()
    eng = _db_connect()
    content_df = _load_content_eval(eng)
    ts_col = _guess_ts_column(eng)
    events = _load_events_eval(eng, ts_col)

    # split
    train_ev, val_ev, test_ev = _split_user_tvt(events, ts_col)

    # choose grids & params by mode
    tfidf_list = [TFIDF_FAST] if run_mode.upper()=='FAST' else TFIDF_FULL
    weight_grid = WEIGHT_GRID_FAST if run_mode.upper()=='FAST' else WEIGHT_GRID_FULL
    neg_per_pos = 3 if run_mode.upper()=='FAST' else 5

    # fingerprint for caching
    content_hash = _md5_of_df(content_df, ['post_id', TEXT_COL, ENGAGE_COL])
    sample_ev = events[['user_id','post_id','action_type']].head(5000).copy() if len(events)>5000 else events[['user_id','post_id','action_type']]
    events_hash  = _md5_of_df(sample_ev, ['user_id','post_id','action_type'])
    cache_key = f"{content_hash}_{events_hash}"

    # tuning loop
    best = None
    tune_rows = []
    for tfp in tfidf_list:
        for weights in weight_grid:
            # สร้าง/โหลด CB/CF ภายใต้พารามิเตอร์ tfidf + weights
            # หมายเหตุ: CB.fit จะอิง content_df ที่เตรียม prior ไว้แล้ว
            # แต่เราจะใช้ object ที่สร้างใหม่ทุกครั้งภายใต้ evaluate_split
            # เพื่อความถูกต้องของ cache
            df, macro, val_lab, test_lab, val_scores, _ = _evaluate_split(
                content_df, train_ev, val_ev, test_ev, tfp, weights, mode_tag='tune', cache_key=cache_key
            )
            if macro.empty or not (macro['split']=='val').any():
                score = -1.0
            else:
                row = macro[(macro['split']=='val') & (macro['K']==TUNE_K)]
                score = float(row[TUNE_METRIC].iloc[0]) if len(row) else -1.0
            tune_rows.append({'tfidf':str(tfp),'weights':weights,TUNE_METRIC:score})
            if best is None or score > best[0]:
                best = (score, tfp, weights, val_lab, val_scores)

    pd.DataFrame(tune_rows).to_csv(os.path.join(OUT_DIR,'tuning_results.csv'), index=False)
    if best is None or best[0] < 0:
        print('[ERROR] tuning failed (not enough validation positives).'); return

    best_tfidf, best_w, val_lab, val_scores = best[1], best[2], best[3], best[4]

    # Train re-ranker บน validation
    lr = _train_reranker(val_lab, val_scores, neg_per_pos=neg_per_pos)

    # Final evaluate on TEST with best settings
    df_final, macro_final, _, test_lab, _, test_scores = _evaluate_split(
        content_df, train_ev, val_ev, test_ev, best_tfidf, best_w, mode_tag='final', cache_key=cache_key
    )

    # Compare base vs rerank on TEST
    rows = []
    users_test = sorted(test_lab[test_lab['y']==1]['user_id'].unique().tolist())
    pos_map_test = test_lab[test_lab['y']==1].groupby('user_id')['post_id'].apply(set).to_dict()

    for uid in users_test:
        sc = test_scores.get(int(uid))
        if sc is None or sc.empty: continue
        base_rank = sc['post_id'].tolist()
        rerank = _apply_reranker(lr, sc)
        pos = pos_map_test.get(uid, set())
        for tag, rank in [('test_base', base_rank), ('test_rerank', rerank)]:
            rel = [1 if p in pos else 0 for p in rank]
            num_pos = len(pos)
            for K in K_LIST:
                rows.append({
                    'mode': tag, 'K': K,
                    'precision': _precision_at_k(rel, K),
                    'recall':    _recall_at_k(rel, K, num_pos),
                    'ndcg':      _ndcg_at_k(rel, K),
                    'hitrate':   _hit_at_k(rel, K),
                    'num_pos':   num_pos,
                    'num_candidates': len(rank),
                    'w_collab': best_w[0], 'w_item': best_w[1], 'w_user': best_w[2],
                })

    df_test_comp = pd.DataFrame(rows)
    macro_test_comp = df_test_comp.groupby(['mode','K']).mean(numeric_only=True).reset_index()

    # Save & plots
    if len(df_final): df_final.to_csv(os.path.join(OUT_DIR,'final_metrics_per_user.csv'), index=False)
    if len(macro_final): macro_final.to_csv(os.path.join(OUT_DIR,'final_metrics_macro.csv'), index=False)
    if len(df_test_comp): df_test_comp.to_csv(os.path.join(OUT_DIR,'final_test_compare_per_user.csv'), index=False)
    if len(macro_test_comp): macro_test_comp.to_csv(os.path.join(OUT_DIR,'final_test_compare_macro.csv'), index=False)

    for m in ['precision','recall','ndcg','hitrate']:
        if len(macro_test_comp):
            plt.figure()
            for tag in ['test_base','test_rerank']:
                sub = macro_test_comp[macro_test_comp['mode']==tag]
                if len(sub): plt.plot(sub['K'], sub[m], marker='o', label=tag)
            plt.xlabel('K'); plt.ylabel(m.upper()); plt.title(f'{m.upper()}@K (TEST, base vs rerank)')
            plt.legend(); plt.grid(True, linestyle='--', linewidth=0.5)
            plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f'test_{m}_base_vs_rerank.png')); plt.close()

    print('[INFO] Tuning complete. Best settings:')
    print(f'  tfidf = {best_tfidf}')
    print(f'  weights (collab,item,user) = {best_w}')
    if len(macro_final):
        print('\n=== FINAL (BASE) VAL/TEST ===')
        print(macro_final.round(4).to_string(index=False))
    if len(macro_test_comp):
        print('\n=== TEST: BASE vs RERANK ===')
        print(macro_test_comp.round(4).to_string(index=False))
    print(f'\nOutputs in: {OUT_DIR}')

# ================= CLI (รัน evaluation ได้จากไฟล์นี้เลย) =================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Bestpick Recsys evaluation (single-file)')
    parser.add_argument('--mode', default=RUN_MODE_DEFAULT, choices=['FAST','FULL'], help='evaluation mode')
    args = parser.parse_args()
    train_tune_eval(args.mode)
