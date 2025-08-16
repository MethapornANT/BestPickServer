# appai_eval_runtime_fixed.py
# ===================== EVALUATION (ใช้โค้ด+ตั้งค่าเดียวกับ Runtime) =====================
# - ใช้พารามิเตอร์ที่ “ล็อกแล้ว” จาก runtime: TFIDF_PARAMS, WEIGHTS, POP_ALPHA, SVD
# - สร้าง/โหลดบล็อกโมเดลด้วยแคชไฟล์แบบเดียวกัน (cache_key = md5(content slice) + md5(event slice))
# - Split แบบ per-user เป็น train/val/test เหมือนเดิม
# - ประเมิน base vs rerank (LogisticRegression) แล้วเซฟ CSV/รูปกราฟเหมือนเดิม

import os, math, json, pickle, hashlib, warnings
from typing import List, Dict, Tuple, Optional
from datetime import datetime

warnings.filterwarnings("ignore", category=UserWarning)

import numpy as np
import pandas as pd

from sqlalchemy import create_engine

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import normalize as sk_normalize
from sklearn.linear_model import LogisticRegression

from scipy.sparse import csr_matrix
from scipy.sparse import load_npz, save_npz

from surprise import SVD, Dataset, Reader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ===================== GLOBAL CONFIG (ตรงกับ runtime) =====================
DB_URI = os.getenv("BESTPICK_DB_URI", "mysql+mysqlconnector://root:1234@localhost/bestpick")

CONTENT_VIEW = "contentbasedview"
EVENT_TABLE  = "user_interactions"

TEXT_COL   = "Content"
ENGAGE_COL = "PostEngagement"

POS_ACTIONS    = {"like","comment","bookmark","share"}
NEG_ACTIONS    = {"unlike"}
IGNORE_ACTIONS = {"view_profile","follow","unfollow"}
VIEW_POS_MIN   = 3

ACTION_WEIGHT = {"view":1.0,"like":4.0,"comment":4.0,"bookmark":4.5,"share":5.0,"unlike":-3.0}
RATING_MIN, RATING_MAX = 0.5, 5.0

TFIDF_PARAMS = dict(analyzer="char_wb", ngram_range=(2,5), max_features=60000, min_df=2, max_df=0.95, stop_words=None)
WEIGHTS      = (0.3, 0.3, 0.4)   # (collab, item_content, user_content)
POP_ALPHA    = 5.0

OUT_DIR   = "./Recommend_Evaluation"
CACHE_DIR = os.path.join(OUT_DIR, "cache")

# Eval settings (แบบเดิม)
K_LIST      = [5, 10, 20]
NEG_PER_POS = 3          # สำหรับ reranker
TUNE_METRIC = "ndcg"     # ใช้แค่ในสรุปชื่อคอลัมน์

# ===================== UTILS =====================
def ensure_dirs():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

def _connect():
    return create_engine(DB_URI, pool_pre_ping=True, pool_recycle=1800)

def _guess_ts_column(eng) -> Optional[str]:
    try:
        one = pd.read_sql(f"SELECT * FROM {EVENT_TABLE} LIMIT 1", eng)
    except Exception:
        return None
    for c in ["created_at","updated_at","ts","timestamp","event_time","inserted_at"]:
        if c in one.columns:
            return c
    return None

def _normalize(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce").fillna(0.0).astype(np.float32)
    mn, mx = float(s.min()), float(s.max())
    return (s - mn) / (mx - mn + 1e-12)

def _md5_of_df(df: pd.DataFrame, cols: List[str]) -> str:
    snap = df[cols].copy().fillna(0)
    h = hashlib.md5(pd.util.hash_pandas_object(snap, index=False).values).hexdigest()
    return h

# ===================== DATA IO =====================
def _load_content(eng) -> pd.DataFrame:
    df = pd.read_sql(f"SELECT * FROM {CONTENT_VIEW}", eng)

    if "post_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id":"post_id"})
    df["post_id"] = pd.to_numeric(df["post_id"], errors="coerce").dropna().astype(int)

    if TEXT_COL not in df.columns: df[TEXT_COL] = ""
    if ENGAGE_COL not in df.columns: df[ENGAGE_COL] = 0.0

    eng = pd.to_numeric(df[ENGAGE_COL], errors="coerce").fillna(0.0).astype(np.float32)
    prior = (eng + POP_ALPHA) / (eng.max() + POP_ALPHA)
    df["PopularityPrior"]     = _normalize(pd.Series(prior))
    df["NormalizedEngagement"] = _normalize(pd.Series(eng))

    if "owner_id" not in df.columns:
        df["owner_id"] = -1
    else:
        df["owner_id"] = pd.to_numeric(df["owner_id"], errors="coerce").fillna(-1).astype(int)
    return df

def _load_events(eng, ts_col: Optional[str]) -> pd.DataFrame:
    base = "user_id, post_id, action_type"
    if ts_col: base += f", {ts_col} AS ts"
    ev = pd.read_sql(f"SELECT {base} FROM {EVENT_TABLE}", eng)
    ev = ev.dropna(subset=["user_id","post_id"])
    ev["user_id"] = pd.to_numeric(ev["user_id"], errors="coerce").dropna().astype(int)
    ev["post_id"] = pd.to_numeric(ev["post_id"], errors="coerce").dropna().astype(int)
    ev["action_type"] = ev["action_type"].astype(str).str.lower()
    if "ts" in ev.columns: ev = ev.dropna(subset=["ts"])
    ev = ev[~ev["action_type"].isin(IGNORE_ACTIONS)].copy()
    return ev

# ===================== CONTENT MODELS =====================
def _build_tfidf(content_df: pd.DataFrame, params: dict):
    tfidf = TfidfVectorizer(**params, dtype=np.float32)
    X = tfidf.fit_transform(content_df[TEXT_COL].fillna(""))
    return tfidf, X.astype(np.float32)

def _build_knn(X):
    knn = NearestNeighbors(n_neighbors=10, metric="cosine")
    knn.fit(X)
    return knn

def _precompute_item_content_scores(knn, content_df: pd.DataFrame, X: csr_matrix) -> np.ndarray:
    n = X.shape[0]
    scores = np.zeros(n, dtype=np.float32)
    n_neighbors = min(20, n)
    _, idxs = knn.kneighbors(X, n_neighbors=n_neighbors)
    eng = content_df["NormalizedEngagement"].to_numpy(dtype=np.float32)
    for i in range(n):
        scores[i] = float(np.mean(eng[idxs[i]])) if idxs[i].size else 0.0
    return scores

def _user_text_profiles(train_pos: pd.DataFrame, content_df: pd.DataFrame, X: csr_matrix) -> Dict[int, csr_matrix]:
    pid_to_idx = {int(pid): i for i, pid in enumerate(content_df["post_id"].tolist())}
    profiles: Dict[int, csr_matrix] = {}
    for uid, g in train_pos.groupby("user_id"):
        idxs = [pid_to_idx.get(p) for p in g["post_id"].tolist() if pid_to_idx.get(p) is not None]
        if not idxs:
            profiles[int(uid)] = csr_matrix((1, X.shape[1]), dtype=np.float32)
            continue
        mat = X[idxs]
        mean_vec = mat.mean(axis=0)
        if hasattr(mean_vec, "toarray"): mean_vec = mean_vec.toarray()
        else: mean_vec = np.asarray(mean_vec)
        prof = sk_normalize(mean_vec)
        profiles[int(uid)] = csr_matrix(prof, dtype=np.float32)
    return profiles

def _user_content_score(uid: int, profiles: Dict[int, csr_matrix], X: csr_matrix, idx: int) -> float:
    prof = profiles.get(int(uid))
    if prof is None or prof.nnz == 0: return 0.0
    v = X[idx]
    num = float(v.multiply(prof).sum())
    den = (np.linalg.norm(v.data) * np.linalg.norm(prof.data)) if prof.nnz>0 and v.nnz>0 else 0.0
    return float(num/den) if den>0 else 0.0

# ===================== COLLABORATIVE =====================
def _build_true_labels(events: pd.DataFrame) -> pd.DataFrame:
    t = events.groupby(["user_id","post_id","action_type"]).size().reset_index(name="cnt")
    if t.empty:
        return pd.DataFrame(columns=["user_id","post_id","y"])
    pvt = t.pivot_table(index=["user_id","post_id"], columns="action_type",
                        values="cnt", fill_value=0, aggfunc="sum").reset_index()
    pvt.columns = [str(c).lower() for c in pvt.columns]
    pos = np.zeros(len(pvt), dtype=bool)
    for a in POS_ACTIONS:
        if a in pvt.columns: pos |= (pvt[a].to_numpy(dtype=float) > 0)
    if "view" in pvt.columns:
        pos |= (pvt["view"].to_numpy(dtype=float) >= VIEW_POS_MIN)
    if NEG_ACTIONS:
        neg = np.zeros(len(pvt), dtype=bool)
        for a in NEG_ACTIONS:
            if a in pvt.columns: neg |= (pvt[a].to_numpy(dtype=float) > 0)
        pos = np.where(neg, False, pos)
    pvt["y"] = pos.astype(int)
    return pvt[["user_id","post_id","y"]]

def _build_collab_model(events: pd.DataFrame, post_ids: List[int]):
    e = events[events["post_id"].isin(post_ids)].copy()
    if e.empty: return None
    t = e.groupby(["user_id","post_id","action_type"]).size().reset_index(name="cnt")
    pvt = t.pivot_table(index=["user_id","post_id"], columns="action_type",
                        values="cnt", fill_value=0, aggfunc="sum").reset_index()
    rating = np.zeros(len(pvt), dtype=np.float32)
    for act, w in ACTION_WEIGHT.items():
        if act in pvt.columns:
            rating += np.float32(w) * pvt[act].to_numpy(dtype=np.float32)
    if "view" in pvt.columns:
        rating += np.where(pvt["view"].to_numpy(dtype=np.float32) >= VIEW_POS_MIN, np.float32(2.0), np.float32(0.0))
    rating = np.clip(rating, RATING_MIN, RATING_MAX)

    data = pvt[["user_id","post_id"]].copy()
    data["rating"] = rating
    data = data[data["rating"] > 0]
    if data.empty: return None

    reader   = Reader(rating_scale=(RATING_MIN, RATING_MAX))
    dset     = Dataset.load_from_df(data[["user_id","post_id","rating"]], reader)
    trainset = dset.build_full_trainset()
    model = SVD(n_factors=150, n_epochs=60, lr_all=0.005, reg_all=0.5)
    model.fit(trainset)
    return model

# ===================== RANKING =====================
def _recommend_scores_for_user(uid: int,
                               content_df: pd.DataFrame,
                               tfidf, X, knn,
                               uc_profiles: Dict[int, csr_matrix],
                               collab_model,
                               item_content_scores: np.ndarray,
                               weights: Tuple[float,float,float]) -> pd.DataFrame:
    wc, wi, wu = weights
    rows = []
    collab_pred_default = 0.5
    for i in range(len(content_df)):
        row = content_df.iloc[i]
        pid = int(row["post_id"])
        # กันโพสต์ของตัวเอง
        if int(row.get("owner_id", -1)) == int(uid):
            continue
        # collab
        collab = collab_pred_default
        if collab_model is not None:
            try:
                collab = float(collab_model.predict(int(uid), pid).est)
            except Exception:
                collab = collab_pred_default
        # item content
        ic = float(item_content_scores[i]) if i < len(item_content_scores) else 0.0
        # user-content
        uc = _user_content_score(uid, uc_profiles, X, i)
        # popularity prior
        pop = float(row.get("PopularityPrior", 0.0))
        final = wc*collab + wi*ic + wu*uc + 0.05*pop
        rows.append((pid, collab, ic, uc, pop, final))
    out = pd.DataFrame(rows, columns=["post_id","collab","item_content","user_content","pop","final"])
    out["final_norm"] = _normalize(out["final"])
    return out.sort_values(["final_norm","final"], ascending=[False, False])

# ===================== CACHE PATHS (แบบ runtime) =====================
def _get_cache_paths(cache_key: str, tfidf_params: dict):
    key = hashlib.md5(json.dumps(tfidf_params, sort_keys=True).encode()).hexdigest()
    base = os.path.join(CACHE_DIR, cache_key)
    return {
        "tfidf_pkl": base + f".tfidf_{key}.pkl",
        "X_npz"    : base + f".X_{key}.npz",
        "knn_pkl"  : base + ".knn.pkl",
        "ics_npy"  : base + ".item_scores.npy",
        "uc_pkl"   : base + ".ucprof.pkl",
        "svd_pkl"  : base + ".svd.pkl",
    }

def _build_or_load_blocks_for_train(content_df: pd.DataFrame, events_for_train: pd.DataFrame, use_cache: bool=True):
    """
    ใช้เฉพาะ TRAIN เพื่อสร้าง uc_profiles และ SVD จาก train เท่านั้น
    ส่วน TF-IDF/X/KNN/ItemScores ใช้ content ทั้งหมด (เหมือน runtime) และแคชตาม content_hash
    """
    content_hash = _md5_of_df(content_df, ["post_id", TEXT_COL, ENGAGE_COL])
    sample_ev = events_for_train[["user_id","post_id","action_type"]].head(5000).copy() if len(events_for_train)>5000 else events_for_train[["user_id","post_id","action_type"]]
    events_hash  = _md5_of_df(sample_ev, ["user_id","post_id","action_type"])
    cache_key = f"{content_hash}_{events_hash}"
    P = _get_cache_paths(cache_key, TFIDF_PARAMS)

    # TF-IDF
    if use_cache and os.path.exists(P["tfidf_pkl"]) and os.path.exists(P["X_npz"]):
        with open(P["tfidf_pkl"], "rb") as f: tfidf = pickle.load(f)
        X = load_npz(P["X_npz"]).astype(np.float32)
    else:
        tfidf, X = _build_tfidf(content_df, TFIDF_PARAMS)
        with open(P["tfidf_pkl"], "wb") as f: pickle.dump(tfidf, f)
        save_npz(P["X_npz"], X)

    # KNN + item content
    if use_cache and os.path.exists(P["knn_pkl"]) and os.path.exists(P["ics_npy"]):
        with open(P["knn_pkl"], "rb") as f: knn = pickle.load(f)
        item_scores = np.load(P["ics_npy"])
    else:
        knn = _build_knn(X)
        item_scores = _precompute_item_content_scores(knn, content_df, X)
        with open(P["knn_pkl"], "wb") as f: pickle.dump(knn, f)
        np.save(P["ics_npy"], item_scores)

    # User text profiles จาก TRAIN labels บวก
    train_labels = _build_true_labels(events_for_train)
    train_pos = train_labels[train_labels["y"]==1][["user_id","post_id"]]
    if use_cache and os.path.exists(P["uc_pkl"]):
        try:
            with open(P["uc_pkl"], "rb") as f: uc_prof = pickle.load(f)
        except Exception:
            uc_prof = _user_text_profiles(train_pos, content_df, X)
            with open(P["uc_pkl"], "wb") as f: pickle.dump(uc_prof, f)
    else:
        uc_prof = _user_text_profiles(train_pos, content_df, X)
        with open(P["uc_pkl"], "wb") as f: pickle.dump(uc_prof, f)

    # SVD จาก TRAIN
    if use_cache and os.path.exists(P["svd_pkl"]):
        try:
            with open(P["svd_pkl"], "rb") as f: svd = pickle.load(f)
        except Exception:
            svd = _build_collab_model(events_for_train, content_df["post_id"].tolist())
            with open(P["svd_pkl"], "wb") as f: pickle.dump(svd, f)
    else:
        svd = _build_collab_model(events_for_train, content_df["post_id"].tolist())
        with open(P["svd_pkl"], "wb") as f: pickle.dump(svd, f)

    return tfidf, X, knn, item_scores, uc_prof, svd, cache_key

# ===================== METRICS (เหมือนเดิม) =====================
def precision_at_k(rel: List[int], k: int) -> float:
    k = min(k, len(rel))
    return float(np.sum(rel[:k]))/k if k else 0.0

def recall_at_k(rel: List[int], k: int, num_pos: int) -> float:
    k = min(k, len(rel))
    return float(np.sum(rel[:k]))/float(max(1, num_pos)) if k else 0.0

def dcg_at_k(rel: List[int], k: int) -> float:
    return float(sum((2**r - 1)/math.log2(i+2) for i, r in enumerate(rel[:k])))

def ndcg_at_k(rel: List[int], k: int) -> float:
    ideal = sorted(rel, reverse=True); idcg = dcg_at_k(ideal, k)
    return (dcg_at_k(rel, k)/idcg) if idcg > 0 else 0.0

def hit_at_k(rel: List[int], k: int) -> float:
    return 1.0 if any(rel[:k]) else 0.0

# ===================== SPLIT =====================
def split_user_tvt(events: pd.DataFrame, ts_col: Optional[str], ratios=(0.6,0.2,0.2)):
    a, b, c = ratios
    ev = events.copy()
    if ts_col and "ts" in ev.columns:
        ev = ev.sort_values(["user_id","ts"])
    else:
        ev = ev.groupby("user_id", group_keys=False).apply(lambda g: g.sample(frac=1.0, random_state=42))
    tr, va, te = [], [], []
    for uid, g in ev.groupby("user_id"):
        n = len(g)
        n_tr = max(1, int(n*a)); n_va = max(1, int(n*b)); n_te = max(1, n - n_tr - n_va)
        tr.append(g.iloc[:n_tr]); va.append(g.iloc[n_tr:n_tr+n_va]); te.append(g.iloc[n_tr+n_va:])
    return pd.concat(tr, ignore_index=True), pd.concat(va, ignore_index=True), pd.concat(te, ignore_index=True)

# ===================== RERANKER =====================
def train_reranker(val_lab: pd.DataFrame,
                   scores_map: Dict[int, pd.DataFrame],
                   neg_per_pos: int) -> Optional[LogisticRegression]:
    X_rows, y_rows = [], []
    for uid, _ in val_lab.groupby("user_id"):
        pos_items = set(val_lab[(val_lab["user_id"]==uid) & (val_lab["y"]==1)]["post_id"].tolist())
        if not pos_items: continue
        sc = scores_map.get(int(uid))
        if sc is None or sc.empty: continue

        pos_feat = sc[sc["post_id"].isin(pos_items)][["collab","item_content","user_content","pop","final"]].values
        if len(pos_feat) == 0: continue
        X_rows.append(pos_feat); y_rows += [1]*len(pos_feat)

        cand_neg = sc[~sc["post_id"].isin(pos_items)]
        if len(cand_neg) == 0: continue
        sample_n = min(len(cand_neg), max(neg_per_pos*len(pos_feat), 8))
        neg_feat = cand_neg.sample(n=sample_n, random_state=42)[["collab","item_content","user_content","pop","final"]].values
        X_rows.append(neg_feat); y_rows += [0]*len(neg_feat)

    if not X_rows: return None
    Xtr = np.vstack(X_rows); ytr = np.array(y_rows, dtype=int)
    lr = LogisticRegression(max_iter=500, n_jobs=1)
    lr.fit(Xtr, ytr)
    return lr

def apply_reranker(lr: Optional[LogisticRegression], sc: pd.DataFrame) -> List[int]:
    if lr is None or sc is None or sc.empty:
        return sc["post_id"].tolist()
    feats = sc[["collab","item_content","user_content","pop","final"]].values
    sc = sc.copy()
    sc["lr_score"] = lr.predict_proba(feats)[:,1]
    return sc.sort_values("lr_score", ascending=False)["post_id"].tolist()

# ===================== EVALUATION LOOP (Fixed config) =====================
def evaluate_fixed():
    ensure_dirs()
    eng = _connect()
    content_df = _load_content(eng)
    ts_col = _guess_ts_column(eng)
    events   = _load_events(eng, ts_col)

    # split
    train_ev, val_ev, test_ev = split_user_tvt(events, ts_col, ratios=(0.6,0.2,0.2))

    # build/load blocks จาก TRAIN
    tfidf, X, knn, item_scores, uc_prof, svd, _ = _build_or_load_blocks_for_train(content_df, train_ev, use_cache=True)

    # labels
    train_lab = _build_true_labels(train_ev)
    val_lab   = _build_true_labels(val_ev)
    test_lab  = _build_true_labels(test_ev)

    seen_train = train_lab[train_lab["y"]==1].groupby("user_id")["post_id"].apply(set).to_dict()

    def eval_part(labels_df: pd.DataFrame, part: str) -> Tuple[pd.DataFrame, pd.DataFrame, Dict[int, pd.DataFrame]]:
        rows = []; per_user_scores: Dict[int, pd.DataFrame] = {}
        users = sorted(labels_df[labels_df["y"]==1]["user_id"].unique().tolist())
        pos_map = labels_df[labels_df["y"]==1].groupby("user_id")["post_id"].apply(set).to_dict()

        for uid in users:
            pos = pos_map.get(uid, set())
            if not pos: continue

            sc = _recommend_scores_for_user(int(uid), content_df, tfidf, X, knn, uc_prof, svd, item_scores, WEIGHTS)

            # กัน item ที่เคยอยู่ใน TRAIN positives ของ user
            seen = seen_train.get(uid, set())
            if len(seen):
                sc = sc[~sc["post_id"].isin(seen)]

            per_user_scores[int(uid)] = sc
            ranking = sc["post_id"].tolist()
            rel = [1 if p in pos else 0 for p in ranking]
            num_pos = len(pos)

            for K in K_LIST:
                rows.append({
                    "mode": part, "K": K,
                    "precision": precision_at_k(rel, K),
                    "recall":    recall_at_k(rel, K, num_pos),
                    "ndcg":      ndcg_at_k(rel, K),
                    "hitrate":   hit_at_k(rel, K),
                    "num_pos":   num_pos,
                    "num_candidates": len(ranking),
                    "w_collab": WEIGHTS[0], "w_item": WEIGHTS[1], "w_user": WEIGHTS[2],
                    "collab_variant": "svd"
                })
        df = pd.DataFrame(rows)
        macro = df.groupby("K").mean(numeric_only=True).reset_index() if len(df) else pd.DataFrame()
        return df, macro, per_user_scores

    # VAL evaluate (สำหรับฝึก reranker)
    _, val_macro, val_scores = eval_part(val_lab, "fixed_val")

    # ฝึก reranker บน VAL
    lr = train_reranker(val_lab, val_scores, neg_per_pos=NEG_PER_POS)

    # TEST evaluate (base + rerank)
    test_rows = []
    _, _, test_scores = eval_part(test_lab, "fixed_test_base_tmp")

    users_test = sorted(test_lab[test_lab["y"]==1]["user_id"].unique().tolist())
    pos_map_test = test_lab[test_lab["y"]==1].groupby("user_id")["post_id"].apply(set).to_dict()

    for uid in users_test:
        sc = test_scores.get(int(uid))
        if sc is None or sc.empty: continue
        base_rank = sc["post_id"].tolist()
        rerank    = apply_reranker(lr, sc)
        pos = pos_map_test.get(uid, set())

        for tag, rank in [("test_base", base_rank), ("test_rerank", rerank)]:
            rel = [1 if p in pos else 0 for p in rank]
            num_pos = len(pos)
            for K in K_LIST:
                test_rows.append({
                    "mode": tag, "K": K,
                    "precision": precision_at_k(rel, K),
                    "recall":    recall_at_k(rel, K, num_pos),
                    "ndcg":      ndcg_at_k(rel, K),
                    "hitrate":   hit_at_k(rel, K),
                    "num_pos":   num_pos,
                    "num_candidates": len(rank),
                    "w_collab": WEIGHTS[0], "w_item": WEIGHTS[1], "w_user": WEIGHTS[2],
                    "collab_variant": "svd"
                })

    df_test_comp = pd.DataFrame(test_rows)
    macro_test_comp = df_test_comp.groupby(["mode","K"]).mean(numeric_only=True).reset_index()

    # เซฟผล
    if len(val_macro):        val_macro.to_csv(os.path.join(OUT_DIR, "fixed_val_metrics_macro.csv"), index=False)
    if len(df_test_comp):     df_test_comp.to_csv(os.path.join(OUT_DIR, "fixed_test_compare_per_user.csv"), index=False)
    if len(macro_test_comp):  macro_test_comp.to_csv(os.path.join(OUT_DIR, "fixed_test_compare_macro.csv"), index=False)

    # กราฟ base vs rerank
    for m in ["precision","recall","ndcg","hitrate"]:
        if len(macro_test_comp):
            plt.figure()
            for tag in ["test_base","test_rerank"]:
                sub = macro_test_comp[macro_test_comp["mode"]==tag]
                if len(sub): plt.plot(sub["K"], sub[m], marker="o", label=tag)
            plt.xlabel("K"); plt.ylabel(m.upper()); plt.title(f"{m.upper()}@K (TEST, base vs rerank)")
            plt.legend(); plt.grid(True, linestyle="--", linewidth=0.5)
            plt.tight_layout(); plt.savefig(os.path.join(OUT_DIR, f"fixed_test_{m}_base_vs_rerank.png")); plt.close()

    print("[INFO] Fixed-config evaluation complete.")
    print(f"  TF-IDF = {TFIDF_PARAMS}")
    print(f"  weights (collab,item,user) = {WEIGHTS}")
    if len(val_macro):
        print("\n=== VAL (BASE) ===")
        print(val_macro.round(4).to_string(index=False))
    if len(macro_test_comp):
        print("\n=== TEST: BASE vs RERANK ===")
        print(macro_test_comp.round(4).to_string(index=False))
    print(f"\nOutputs in: {OUT_DIR}")

# ===================== CLI =====================
if __name__ == "__main__":
    evaluate_fixed()
