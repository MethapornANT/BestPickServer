# NSFW_Evaluation.py
# -*- coding: utf-8 -*-
# NSFW evaluation (OOF calibrated, progress + ETA, vectorized sweep, fast/bootstrap modes)
import os, json, warnings, math, time, pickle, random
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFile
import torch, torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import confusion_matrix, f1_score, accuracy_score, roc_curve, precision_recall_curve, auc
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import StratifiedKFold, train_test_split
import matplotlib.pyplot as plt
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

# ========== CONFIG ==========
DATA_DIR   = "data"  # subfolders: normal, hentai, porn, sexy, anime
MODEL_PATH = "./NSFW_Model/results/20250817-022748/efficientnetb0/models/efficientnetb0_best.pth"
OUT_DIR    = "NSFW_Evaluation"
LABELS     = ['normal', 'hentai', 'porn', 'sexy', 'anime']

# inference tuning
BATCH_SIZE = 64
USE_AMP    = True
SEED       = 1337

# sweep / calibration / reliability
SINGLE_SWEEP_START = 20
SINGLE_SWEEP_END   = 40
BASELINE_PAIRS = [(20,20), (20,25), (25,20), (30,25)]

N_FOLDS = 5                # stratified K for OOF calibration
BOOTSTRAP_ITERS = 200      # bootstrap iterations (reduce for speed)
BOOTSTRAP_MODE = "fast"    # "fast" -> bootstrap metrics at fixed best_pair; "full" -> re-sweep each iter (very slow)
USE_VECTORIZED_SWEEP = True  # use vectorized sweep_pair_thresholds implementation (faster)
CALIBRATION_DIR = os.path.join(OUT_DIR, "calibration_models_oof")
RANDOM_SEED = SEED

# ========== SAFETY & UTILS ==========
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Palette images.*transparency")

def set_seed(seed=SEED):
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)

def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(CALIBRATION_DIR, exist_ok=True)

def save_plot(path):
    plt.tight_layout(); plt.savefig(path); plt.close()

def make_json_serializable(x):
    if isinstance(x, dict):
        return {make_json_serializable(k): make_json_serializable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [make_json_serializable(v) for v in x]
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    try:
        import pandas as _pd
        if isinstance(x, _pd.Timestamp):
            return x.isoformat()
    except Exception:
        pass
    return x

# ========== MODEL ==========
class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes=len(LABELS)):
        super().__init__()
        try:
            self.efficientnet = models.efficientnet_b0(weights=None)
        except TypeError:
            self.efficientnet = models.efficientnet_b0(pretrained=False)
        try:
            in_f = self.efficientnet.classifier[1].in_features
            self.efficientnet.classifier[1] = nn.Linear(in_f, num_classes)
        except Exception:
            try:
                in_f2 = self.efficientnet.classifier.in_features
            except Exception:
                in_f2 = 1280
            self.efficientnet.classifier = nn.Sequential(nn.Dropout(0.2), nn.Linear(in_f2, num_classes))
    def forward(self, x):
        return self.efficientnet(x)

def load_checkpoint(model, ckpt_path, log_path):
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    new_state = {}
    for k, v in state.items():
        nk = k
        for pref in ["module.", "model.", "efficientnet."]:
            if nk.startswith(pref):
                nk = nk[len(pref):]
        if not nk.startswith("efficientnet.") and (nk.startswith("classifier") or nk.startswith("features") or "conv" in nk or "fc" in nk):
            nk = f"efficientnet.{nk}"
        new_state[nk] = v
    try:
        load_res = model.load_state_dict(new_state, strict=False)
        missing = getattr(load_res, "missing_keys", None)
        unexpected = getattr(load_res, "unexpected_keys", None)
        if missing is None or unexpected is None:
            if isinstance(load_res, (tuple, list)) and len(load_res) == 2:
                missing, unexpected = load_res
            else:
                missing, unexpected = [], []
    except Exception:
        model_state = model.state_dict()
        intersect = {k: v for k, v in new_state.items() if k in model_state and model_state[k].shape == v.shape}
        model_state.update(intersect)
        model.load_state_dict(model_state, strict=False)
        missing = list(set(model.state_dict().keys()) - set(intersect.keys()))
        unexpected = [k for k in new_state.keys() if k not in intersect]
    try:
        with open(log_path, "w", encoding="utf-8") as f:
            f.write("== load_state_log ==\n")
            f.write(f"missing_keys: {list(missing)}\n")
            f.write(f"unexpected_keys: {list(unexpected)}\n")
    except Exception:
        pass

# ========== PREPROCESS ==========
preprocess = transforms.Compose([
    transforms.Resize((224,224), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])
ALLOWED_EXTS = {'.jpg','.jpeg','.png','.webp'}

def load_image_paths(root):
    paths, labels = [], []
    for cls in LABELS:
        d = Path(root)/cls
        if not d.is_dir(): continue
        for p in d.rglob("*"):
            if p.suffix.lower() in ALLOWED_EXTS:
                paths.append(str(p)); labels.append(cls)
    return paths, labels

def load_and_preprocess(fp):
    im = Image.open(fp)
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    im = ImageOps.exif_transpose(im).convert("RGB")
    return preprocess(im)

@torch.inference_mode()
def run_inference_batched(model, device, paths):
    rows=[]; kept_paths=[]
    pbar = tqdm(total=len(paths), desc="Inference", unit="img")
    for i in range(0,len(paths),BATCH_SIZE):
        batch_paths = paths[i:i+BATCH_SIZE]
        tensors=[]; valid_paths=[]
        for fp in batch_paths:
            try:
                tensors.append(load_and_preprocess(fp)); valid_paths.append(fp)
            except Exception:
                pass
            finally:
                pbar.update(1)
        if not tensors: continue
        x = torch.stack(tensors).to(device, non_blocking=True)
        try:
            if device.type == 'cuda' and USE_AMP:
                with torch.amp.autocast(device_type='cuda'):
                    logits = model(x)
            else:
                logits = model(x)
        except Exception:
            if device.type == 'cuda' and USE_AMP:
                with torch.cuda.amp.autocast():
                    logits = model(x)
            else:
                logits = model(x)
        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        for j,fp in enumerate(valid_paths):
            rows.append((fp, *probs[j])); kept_paths.append(fp)
    pbar.close()
    cols = ["path"] + [f"prob_{c}" for c in LABELS]
    return pd.DataFrame(rows, columns=cols), kept_paths

# ========== METRICS / CALIBRATION HELPERS ==========
def metrics_from_cm(y_true, y_pred):
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    except Exception:
        y_true = np.asarray(y_true).astype(int); y_pred = np.asarray(y_pred).astype(int)
        tp = int(np.sum((y_true==1)&(y_pred==1))); tn = int(np.sum((y_true==0)&(y_pred==0)))
        fp = int(np.sum((y_true==0)&(y_pred==1))); fn = int(np.sum((y_true==1)&(y_pred==0)))
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = f1_score(y_true, y_pred) if (tp+fp+fn)>0 else 0.0
    acc  = accuracy_score(y_true, y_pred)
    fpr  = fp/(fp+tn) if (fp+tn)>0 else 0.0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp, precision=prec, recall=rec, f1=f1, accuracy=acc, fpr=fpr, specificity=spec)

def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, float); y_prob = np.asarray(y_prob, float)
    return np.mean((y_prob - y_true)**2)

def calibration_bins(y_true, y_prob, n_bins=10):
    df = pd.DataFrame({"y":y_true, "p":y_prob})
    df["bin"] = np.minimum((df["p"]*n_bins).astype(int), n_bins-1)
    g = df.groupby("bin")
    return g.agg(predicted=("p","mean"), observed=("y","mean"), n=("y","size")).reset_index(drop=True)

def decile_hist(series):
    s = np.clip(np.array(series)*100.0, 0, 100)
    bins = np.arange(0,110,10)
    hist,_ = np.histogram(s, bins=bins)
    return pd.DataFrame({"bin_start":bins[:-1], "bin_end":bins[1:], "count":hist})

def expected_calibration_error(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true); y_prob = np.asarray(y_prob)
    bins = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        lo,hi = bins[i], bins[i+1]
        mask = (y_prob>=lo)&(y_prob<hi)
        if mask.sum()==0: continue
        acc = y_true[mask].mean()
        conf = y_prob[mask].mean()
        ece += (mask.sum()/len(y_prob)) * abs(acc - conf)
    return ece

# ========== SWEEPS ==========
def pair_rule_predict(hh, pp, th_h, th_p):
    return ((hh*100.0 >= th_h) | (pp*100.0 >= th_p)).astype(int)

def sweep_pair_thresholds_naive(hh, pp, y_true, h_range=range(0,101), p_range=range(0,101)):
    rows=[]
    for th in (h_range if not isinstance(h_range, range) else range(h_range.start, h_range.stop)):
        H = (hh*100.0 >= th)
        for tpv in (p_range if not isinstance(p_range, range) else range(p_range.start, p_range.stop)):
            P = (pp*100.0 >= tpv)
            y_pred = (H|P).astype(int)
            m = metrics_from_cm(y_true, y_pred)
            rows.append((th, tpv, m["precision"], m["recall"], m["f1"], m["accuracy"], m["recall"], m["fpr"], m["specificity"], m["tp"], m["fp"], m["tn"], m["fn"]))
    return pd.DataFrame(rows, columns=["hentai_thresh_pct","porn_thresh_pct","precision","recall","f1","accuracy","tpr","fpr","specificity","tp","fp","tn","fn"])

def sweep_pair_thresholds_vectorized(hh, pp, y_true, h_range=range(0,101), p_range=range(0,101)):
    # Vectorized across p for each h row to speed up massively vs nested loops.
    # hh, pp: arrays (n,), y_true: array (n,) of 0/1
    H_vals = np.array(list(h_range))  # e.g. 0..100
    P_vals = np.array(list(p_range))
    n = len(y_true)
    y_pos = (y_true == 1).astype(np.int64)
    rows = []
    # Precompute P_matrix boolean per p threshold as ints: shape (P_len, n)
    P_mat = (pp[np.newaxis,:]*100.0 >= P_vals[:,np.newaxis]).astype(np.int8)  # (P_len, n)
    # For each h threshold, vectorize combine with all P rows at once
    t0 = time.perf_counter()
    for i, h_t in enumerate(tqdm(H_vals, desc="Sweep hentai thresholds", unit="h")):
        H_row = (hh*100.0 >= h_t).astype(np.int8)  # (n,)
        # broadcast OR: combined (P_len, n)
        combined = (P_mat | H_row[np.newaxis,:]).astype(np.int8)  # (P_len, n)
        # predicted positives per (h,p)
        pred_pos = combined.sum(axis=1)  # (P_len,)
        tp = (combined * y_pos[np.newaxis,:]).sum(axis=1)  # (P_len,)
        fp = pred_pos - tp
        fn = y_pos.sum() - tp
        tn = n - pred_pos - fn
        # compute metrics vectorized
        with np.errstate(divide='ignore', invalid='ignore'):
            precision = np.where((tp+fp)>0, tp/(tp+fp), 0.0)
            recall = np.where((tp+fn)>0, tp/(tp+fn), 0.0)
            f1 = np.where((precision+recall)>0, 2*precision*recall/(precision+recall), 0.0)
            accuracy = (tp + tn) / n
            fpr = np.where((fp+tn)>0, fp/(fp+tn), 0.0)
            specificity = np.where((tn+fp)>0, tn/(tn+fp), 0.0)
        for j, p_t in enumerate(P_vals):
            rows.append((int(h_t), int(p_t), float(precision[j]), float(recall[j]), float(f1[j]), float(accuracy[j]),
                         float(recall[j]), float(fpr[j]), float(specificity[j]), int(tp[j]), int(fp[j]), int(tn[j]), int(fn[j])))
    df = pd.DataFrame(rows, columns=["hentai_thresh_pct","porn_thresh_pct","precision","recall","f1","accuracy","tpr","fpr","specificity","tp","fp","tn","fn"])
    return df

def sweep_pair_thresholds(hh, pp, y_true, h_range=range(0,101), p_range=range(0,101)):
    if USE_VECTORIZED_SWEEP:
        return sweep_pair_thresholds_vectorized(hh, pp, y_true, h_range=h_range, p_range=p_range)
    else:
        return sweep_pair_thresholds_naive(hh, pp, y_true, h_range=h_range, p_range=p_range)

def sweep_single_class(series_prob, y_true_bin, t_start=SINGLE_SWEEP_START, t_end=SINGLE_SWEEP_END):
    rows=[]
    s = np.asarray(series_prob)*100.0
    for t in range(t_start, t_end+1):
        y_pred = (s >= t).astype(int)
        m = metrics_from_cm(y_true_bin, y_pred); m.update({'threshold_pct': t}); rows.append(m)
    return pd.DataFrame(rows)

def choose_best_pair(df_pairs):
    df = df_pairs.sort_values(by=["f1","precision","fpr"], ascending=[False, False, True]).head(1)
    r = df.iloc[0]
    return dict(hentai_thresh_pct=int(r["hentai_thresh_pct"]), porn_thresh_pct=int(r["porn_thresh_pct"]),
                f1=float(r["f1"]), precision=float(r["precision"]), recall=float(r["recall"]),
                fpr=float(r["fpr"]), accuracy=float(r["accuracy"]), tp=int(r["tp"]), fp=int(r["fp"]), tn=int(r["tn"]), fn=int(r["fn"]))

# ========== BOOTSTRAP UTILITIES ==========
def bootstrap_metrics_at_thresholds(h_probs, p_probs, y_true, th_h, th_p, iters=100, rng_seed=RANDOM_SEED):
    rng = np.random.RandomState(rng_seed)
    n = len(y_true)
    rows=[]
    for i in tqdm(range(iters), desc="Bootstrap (fast)"):
        idx = rng.randint(0,n,size=n)
        y_pred = ((h_probs[idx]*100.0 >= th_h) | (p_probs[idx]*100.0 >= th_p)).astype(int)
        m = metrics_from_cm(y_true[idx], y_pred)
        m['iter']=int(i); rows.append(m)
    return pd.DataFrame(rows)

def bootstrap_best_pair_full(h_probs, p_probs, y_true, iters=100, rng_seed=RANDOM_SEED, save_every=10):
    rng = np.random.RandomState(rng_seed)
    n = len(y_true)
    rows=[]
    for i in range(iters):
        idx = rng.randint(0,n,size=n)
        dfb = sweep_pair_thresholds(h_probs[idx], p_probs[idx], y_true[idx], h_range=range(0,101), p_range=range(0,101))
        best = choose_best_pair(dfb)
        best['iter']=int(i)
        rows.append(best)
        if (i+1) % save_every == 0:
            pd.DataFrame(rows).to_csv(os.path.join(OUT_DIR,"bootstrap_progress_partial.csv"), index=False)
    return pd.DataFrame(rows)

# ========== MAIN ==========
def main():
    set_seed(RANDOM_SEED); ensure_outdir()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == 'cuda': torch.backends.cudnn.benchmark = True
    print(f"Device: {device} | AMP:{USE_AMP and device.type=='cuda'} | Batch:{BATCH_SIZE}")
    print("USE_VECTORIZED_SWEEP:", USE_VECTORIZED_SWEEP, "BOOTSTRAP_MODE:", BOOTSTRAP_MODE, "BOOTSTRAP_ITERS:", BOOTSTRAP_ITERS)

    if not os.path.isdir(DATA_DIR):
        raise SystemExit("Data folder missing")
    if not os.path.isfile(MODEL_PATH):
        raise SystemExit("Model checkpoint missing")

    paths, labels = load_image_paths(DATA_DIR)
    if not paths: raise SystemExit("No images found")
    labels_by_path = dict(zip(paths, labels))

    model = CustomEfficientNetB0(len(LABELS)).to(device)
    load_checkpoint(model, MODEL_PATH, os.path.join(OUT_DIR,"load_state_log.txt"))
    model.eval()

    start_t = time.perf_counter()
    df_probs, kept_paths = run_inference_batched(model, device, paths)
    infer_elapsed = time.perf_counter() - start_t
    df_probs = df_probs.reset_index(drop=True)
    df_probs["true_label"] = df_probs["path"].map(labels_by_path)
    n = len(df_probs)
    print(f"Kept: {n} (inference {infer_elapsed:.1f}s)")

    df_probs["hentai_prob"] = df_probs["prob_hentai"]
    df_probs["porn_prob"]   = df_probs["prob_porn"]
    y_true_bin = df_probs["true_label"].isin(["hentai","porn"]).astype(int).values

    skf = StratifiedKFold(n_splits=min(N_FOLDS, max(2, int(n/10))), shuffle=True, random_state=RANDOM_SEED)
    oof_h = np.zeros(n, dtype=float); oof_p = np.zeros(n, dtype=float)
    fold_info = []
    for fold, (train_idx, val_idx) in enumerate(skf.split(np.zeros(n), y_true_bin)):
        h_train_y = (df_probs["true_label"].values[train_idx] == "hentai").astype(int)
        p_train_y = (df_probs["true_label"].values[train_idx] == "porn").astype(int)
        h_train_probs = df_probs["hentai_prob"].values[train_idx]
        p_train_probs = df_probs["porn_prob"].values[train_idx]

        ir_h = IsotonicRegression(out_of_bounds='clip')
        ir_p = IsotonicRegression(out_of_bounds='clip')

        h_ok = (h_train_y.sum()>0) and ((h_train_y==0).sum()>0)
        p_ok = (p_train_y.sum()>0) and ((p_train_y==0).sum()>0)

        if h_ok:
            ir_h.fit(h_train_probs, h_train_y)
            oof_h[val_idx] = ir_h.transform(df_probs["hentai_prob"].values[val_idx])
        else:
            oof_h[val_idx] = df_probs["hentai_prob"].values[val_idx]

        if p_ok:
            ir_p.fit(p_train_probs, p_train_y)
            oof_p[val_idx] = ir_p.transform(df_probs["porn_prob"].values[val_idx])
        else:
            oof_p[val_idx] = df_probs["porn_prob"].values[val_idx]

        fold_info.append({"fold":int(fold), "h_fitted":bool(h_ok), "p_fitted":bool(p_ok), "n_train":int(len(train_idx)), "n_val":int(len(val_idx))})

    df_probs["hentai_oof"] = oof_h; df_probs["porn_oof"] = oof_p
    df_probs["nsfw_max_raw"] = df_probs[["hentai_prob","porn_prob"]].max(axis=1)
    df_probs["nsfw_max_oof"] = np.maximum(df_probs["hentai_oof"], df_probs["porn_oof"])

    br_raw = brier_score(y_true_bin, df_probs["nsfw_max_raw"].values)
    br_oof = brier_score(y_true_bin, df_probs["nsfw_max_oof"].values)
    ece_raw = expected_calibration_error(y_true_bin, df_probs["nsfw_max_raw"].values, n_bins=10)
    ece_oof = expected_calibration_error(y_true_bin, df_probs["nsfw_max_oof"].values, n_bins=10)
    print(f"Brier raw={br_raw:.6f}  oof_cal={br_oof:.6f} | ECE raw={ece_raw:.6f} oof={ece_oof:.6f}")

    # final calibrators on full data
    try:
        h_full_y = (df_probs["true_label"].values == "hentai").astype(int)
        p_full_y = (df_probs["true_label"].values == "porn").astype(int)
        if (h_full_y.sum()>0) and ((h_full_y==0).sum()>0):
            final_ir_h = IsotonicRegression(out_of_bounds='clip'); final_ir_h.fit(df_probs["hentai_prob"].values, h_full_y)
            with open(os.path.join(CALIBRATION_DIR,"final_isotonic_hentai.pkl"),"wb") as f: pickle.dump(final_ir_h, f)
        if (p_full_y.sum()>0) and ((p_full_y==0).sum()>0):
            final_ir_p = IsotonicRegression(out_of_bounds='clip'); final_ir_p.fit(df_probs["porn_prob"].values, p_full_y)
            with open(os.path.join(CALIBRATION_DIR,"final_isotonic_porn.pkl"),"wb") as f: pickle.dump(final_ir_p, f)
    except Exception as e:
        print("Warning saving final calibrators:", e)

    df_probs.to_csv(os.path.join(OUT_DIR,"per_image_scores_oof.csv"), index=False)
    payload_folds = {"folds": make_json_serializable(fold_info), "brier_raw": float(br_raw), "brier_oof": float(br_oof), "ece_raw": float(ece_raw), "ece_oof": float(ece_oof)}
    with open(os.path.join(OUT_DIR,"calibration_folds.json"), "w", encoding="utf-8") as f:
        json.dump(payload_folds, f, ensure_ascii=False, indent=2)

    # Hist + ROC/PR/Calibration
    h_hist_all = decile_hist(df_probs["hentai_oof"]); p_hist_all = decile_hist(df_probs["porn_oof"])
    h_hist_all.to_csv(os.path.join(OUT_DIR,"hist_hentai_all_oof.csv"), index=False)
    p_hist_all.to_csv(os.path.join(OUT_DIR,"hist_porn_all_oof.csv"), index=False)

    fpr, tpr, _ = roc_curve(y_true_bin, df_probs["nsfw_max_oof"].values)
    precs, recs, _ = precision_recall_curve(y_true_bin, df_probs["nsfw_max_oof"].values)
    roc_auc = auc(fpr, tpr); pr_auc = auc(recs, precs)
    pd.DataFrame({"fpr":fpr,"tpr":tpr}).to_csv(os.path.join(OUT_DIR,"roc_points_oof.csv"), index=False)
    pd.DataFrame({"recall":recs,"precision":precs}).to_csv(os.path.join(OUT_DIR,"pr_points_oof.csv"), index=False)
    plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.4f})"); save_plot(os.path.join(OUT_DIR,"roc_oof.png"))
    plt.figure(); plt.plot(recs,precs); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AUC={pr_auc:.4f})"); save_plot(os.path.join(OUT_DIR,"pr_oof.png"))

    cal_oof = calibration_bins(y_true_bin, df_probs["nsfw_max_oof"].values, n_bins=10)
    cal_oof.to_csv(os.path.join(OUT_DIR,"calibration_bins_oof.csv"), index=False)
    plt.figure(); plt.plot(cal_oof["predicted"], cal_oof["observed"], marker="o"); plt.plot([0,1],[0,1],'--'); plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title(f"Calibration (Brier raw={br_raw:.4f} oof={br_oof:.4f})"); save_plot(os.path.join(OUT_DIR,"calibration_oof.png"))

    # Sweep (vectorized if enabled)
    print("Running sweep to find best thresholds (this may take a short while)...")
    sweep_start = time.perf_counter()
    pair = sweep_pair_thresholds(df_probs["hentai_oof"].values, df_probs["porn_oof"].values, y_true_bin, h_range=range(0,101), p_range=range(0,101))
    sweep_elapsed = time.perf_counter() - sweep_start
    pair.to_csv(os.path.join(OUT_DIR,"sweep_pair_hentai_porn_oof.csv"), index=False)
    best_pair = choose_best_pair(pair)
    with open(os.path.join(OUT_DIR,"best_threshold_pair_oof.json"),"w",encoding="utf-8") as f:
        json.dump(make_json_serializable(best_pair), f, ensure_ascii=False, indent=2)
    print("Best pair (OOF):", best_pair, f" (sweep {sweep_elapsed:.1f}s)")

    # Bootstrap (fast or full)
    print("Bootstrap starting (mode=%s). If it's full and your dataset is large, expect this to take long." % BOOTSTRAP_MODE)
    boot_start = time.perf_counter()
    boot_df = None
    try:
        if BOOTSTRAP_MODE == "fast":
            # fast bootstrap: only evaluate metrics at best_pair (much faster)
            boot_df = bootstrap_metrics_at_thresholds(df_probs["hentai_oof"].values, df_probs["porn_oof"].values, y_true_bin, best_pair['hentai_thresh_pct'], best_pair['porn_thresh_pct'], iters=BOOTSTRAP_ITERS, rng_seed=RANDOM_SEED)
            boot_df.to_csv(os.path.join(OUT_DIR,"bootstrap_metrics_fixed_threshold_fast.csv"), index=False)
        else:
            # full bootstrap: re-sweep each sample (very slow). show progress & ETA via tqdm
            boot_df = bootstrap_best_pair_full(df_probs["hentai_oof"].values, df_probs["porn_oof"].values, y_true_bin, iters=BOOTSTRAP_ITERS, rng_seed=RANDOM_SEED, save_every=10)
            boot_df.to_csv(os.path.join(OUT_DIR,"bootstrap_best_pairs.csv"), index=False)
        boot_elapsed = time.perf_counter() - boot_start
        print(f"Bootstrap done (elapsed {boot_elapsed:.1f}s). Saved results to {OUT_DIR}")
        if boot_df is not None and not boot_df.empty:
            def ci_series(s): return (float(np.percentile(s,2.5)), float(np.percentile(s,97.5)))
            if BOOTSTRAP_MODE == "fast":
                ci = {
                    "f1_ci": ci_series(boot_df["f1"].values),
                    "precision_ci": ci_series(boot_df["precision"].values),
                    "recall_ci": ci_series(boot_df["recall"].values),
                    "fpr_ci": ci_series(boot_df["fpr"].values)
                }
            else:
                ci = {
                    "hentai_thresh_ci": ci_series(boot_df["hentai_thresh_pct"].values),
                    "porn_thresh_ci": ci_series(boot_df["porn_thresh_pct"].values),
                    "f1_ci": ci_series(boot_df["f1"].values),
                    "precision_ci": ci_series(boot_df["precision"].values),
                    "recall_ci": ci_series(boot_df["recall"].values),
                    "fpr_ci": ci_series(boot_df["fpr"].values)
                }
            with open(os.path.join(OUT_DIR,"bootstrap_CI.json"),"w",encoding="utf-8") as f:
                json.dump(make_json_serializable(ci), f, ensure_ascii=False, indent=2)
            print("Bootstrap CI saved.")
    except KeyboardInterrupt:
        print("Bootstrap interrupted by user. Partial results may be in OUT_DIR (bootstrap_progress_partial.csv).")
    except Exception as e:
        print("Bootstrap failed/skipped:", e)

    # Baseline & single sweeps
    baseline = []
    for th_h, th_p in BASELINE_PAIRS:
        y_pred = pair_rule_predict(df_probs["hentai_oof"].values, df_probs["porn_oof"].values, th_h, th_p)
        m = metrics_from_cm(y_true_bin, y_pred); m.update({"hentai_thresh_pct":int(th_h),"porn_thresh_pct":int(th_p)}); baseline.append(m)
    pd.DataFrame(baseline).to_csv(os.path.join(OUT_DIR,"confusion_baseline_pairs_oof.csv"), index=False)

    h_single = sweep_single_class(df_probs["hentai_oof"].values, y_true_bin, SINGLE_SWEEP_START, SINGLE_SWEEP_END)
    p_single = sweep_single_class(df_probs["porn_oof"].values,   y_true_bin, SINGLE_SWEEP_START, SINGLE_SWEEP_END)
    h_single.to_csv(os.path.join(OUT_DIR,"sweep_hentai_20_40_independent_oof.csv"), index=False)
    p_single.to_csv(os.path.join(OUT_DIR,"sweep_porn_20_40_independent_oof.csv"), index=False)

    # README
    with open(os.path.join(OUT_DIR,"README_summary.md"),"w",encoding="utf-8") as f:
        f.write("# NSFW Eval Summary (OOF calibrated)\n")
        f.write(f"- Device: {device}, Batch: {BATCH_SIZE}\n")
        f.write(f"- Images evaluated: {n}\n")
        f.write(f"- ROC_AUC={roc_auc:.6f}, PR_AUC={pr_auc:.6f}\n")
        f.write(f"- Brier_raw={br_raw:.6f}, Brier_oof={br_oof:.6f}\n")
        f.write(f"- ECE_raw={ece_raw:.6f}, ECE_oof={ece_oof:.6f}\n\n")
        f.write("## Best thresholds (OOF)\n")
        f.write(json.dumps(make_json_serializable(best_pair), ensure_ascii=False, indent=2))
        if 'ci' in locals():
            f.write("\n\n## Bootstrap 95% CI\n")
            f.write(json.dumps(make_json_serializable(ci), ensure_ascii=False, indent=2))
    print("Done. Results in", OUT_DIR)

if __name__ == "__main__":
    main()
