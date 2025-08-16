# -*- coding: utf-8 -*-
# NSFW evaluation: full metrics, sweeps, histograms, calibration, progress %, CUDA/AMP, and best-threshold suggestions.

import os, json, warnings, math
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps, ImageFile
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics import (
    confusion_matrix, f1_score, accuracy_score,
    roc_curve, precision_recall_curve, auc
)
import matplotlib.pyplot as plt
from tqdm import tqdm

# ========== CONFIG (แก้ตรงนี้ถ้าจำเป็น) ==========
DATA_DIR   = "data"  # ต้องมี subfolders: normal, hentai, porn, sexy, anime
MODEL_PATH = "./NSFW_Model/results/20250816-001345/efficientnetb0/models/efficientnetb0_best.pth"
OUT_DIR    = "NSFW_Evaluation"
LABELS     = ['normal', 'hentai', 'porn', 'sexy', 'anime']

# inference tuning
BATCH_SIZE = 64          # ใช้ batch เพื่อกิน CUDA ให้คุ้ม
USE_AMP    = True        # เปิด AMP บน CUDA เพื่อความไว
SEED       = 1337

# baseline ที่อยากโชว์เทียบ
BASELINE_PAIRS = [(20,20), (20,25), (25,20), (30,25)]

# sweep แบบแยกตัวเดียวช่วง 20..40
SINGLE_SWEEP_START = 20
SINGLE_SWEEP_END   = 40

# ========== PIL SAFETY ==========
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=Image.DecompressionBombWarning)
warnings.filterwarnings("ignore", category=UserWarning, message="Palette images.*transparency")

# ========== UTIL ==========
def set_seed(seed=SEED):
    try:
        torch.manual_seed(seed)
        np.random.seed(seed)
    except Exception:
        pass

def ensure_outdir():
    os.makedirs(OUT_DIR, exist_ok=True)

def save_plot(path):
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ========== MODEL ==========
class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes=len(LABELS)):
        super().__init__()
        # รองรับทั้ง API เก่า/ใหม่ของ torchvision
        try:
            self.efficientnet = models.efficientnet_b0(weights=None)
        except TypeError:
            self.efficientnet = models.efficientnet_b0(pretrained=False)
        in_f = self.efficientnet.classifier[1].in_features
        self.efficientnet.classifier[1] = nn.Linear(in_f, num_classes)
    def forward(self, x):
        return self.efficientnet(x)

def load_checkpoint(model, ckpt_path, log_path):
    state = torch.load(ckpt_path, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]

    # แก้ prefix ที่เจอบ่อย
    new_state = {}
    for k, v in state.items():
        nk = k
        for pref in ["module.", "model.", "efficientnet."]:
            if nk.startswith(pref):
                nk = nk[len(pref):]
        # เติม prefix ให้ตรงกับโครงสร้างปัจจุบัน
        if not nk.startswith("efficientnet."):
            nk = f"efficientnet.{nk}"
        new_state[nk] = v

    missing, unexpected = model.load_state_dict(new_state, strict=False)
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("== load_state_log ==\n")
        f.write(f"missing_keys: {list(missing)}\n")
        f.write(f"unexpected_keys: {list(unexpected)}\n")

# ========== PREPROCESS ==========
preprocess = transforms.Compose([
    transforms.Resize((224, 224), interpolation=Image.BILINEAR),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

ALLOWED_EXTS = {'.jpg', '.jpeg', '.png', '.webp'}

# ========== IO HELPERS ==========
def load_image_paths(root):
    paths, labels = [], []
    for cls in LABELS:
        d = Path(root) / cls
        if not d.is_dir():
            continue
        for p in d.rglob("*"):
            if p.suffix.lower() in ALLOWED_EXTS:
                paths.append(str(p))
                labels.append(cls)
    return paths, labels

def load_and_preprocess(fp):
    im = Image.open(fp)
    # แก้เคส PNG โหมด P + transparency
    if im.mode == "P" and "transparency" in im.info:
        im = im.convert("RGBA")
    im = ImageOps.exif_transpose(im).convert("RGB")
    x = preprocess(im)
    return x

@torch.inference_mode()
def run_inference_batched(model, device, paths):
    rows = []
    kept_paths = []

    pbar = tqdm(total=len(paths), desc="Inference", unit="img")
    for i in range(0, len(paths), BATCH_SIZE):
        batch_paths = paths[i:i+BATCH_SIZE]
        tensors, valid_paths = [], []
        for fp in batch_paths:
            try:
                tensors.append(load_and_preprocess(fp))
                valid_paths.append(fp)
            except Exception:
                # ภาพเสีย/อ่านไม่ได้ ข้าม
                pass
            finally:
                pbar.update(1)

        if not tensors:
            continue

        x = torch.stack(tensors).to(device, non_blocking=True)

        if device.type == "cuda" and USE_AMP:
            with torch.cuda.amp.autocast(dtype=torch.float16):
                logits = model(x)
        else:
            logits = model(x)

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        for j, fp in enumerate(valid_paths):
            rows.append((fp, *probs[j]))
            kept_paths.append(fp)

    pbar.close()
    cols = ["path"] + [f"prob_{c}" for c in LABELS]
    return pd.DataFrame(rows, columns=cols), kept_paths

# ========== METRICS ==========
def metrics_from_cm(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    prec = tp/(tp+fp) if (tp+fp)>0 else 0.0
    rec  = tp/(tp+fn) if (tp+fn)>0 else 0.0
    f1   = f1_score(y_true, y_pred) if (tp+fp+fn)>0 else 0.0
    acc  = accuracy_score(y_true, y_pred)
    fpr  = fp/(fp+tn) if (fp+tn)>0 else 0.0
    spec = tn/(tn+fp) if (tn+fp)>0 else 0.0
    return dict(tn=tn, fp=fp, fn=fn, tp=tp,
                precision=prec, recall=rec, f1=f1,
                accuracy=acc, fpr=fpr, specificity=spec)

def brier_score(y_true, y_prob):
    y_true = np.asarray(y_true, float)
    y_prob = np.asarray(y_prob, float)
    return np.mean((y_prob - y_true)**2)

def calibration_bins(y_true, y_prob, n_bins=10):
    df = pd.DataFrame({"y": y_true, "p": y_prob})
    df["bin"] = np.minimum((df["p"] * n_bins).astype(int), n_bins-1)
    g = df.groupby("bin")
    return g.agg(predicted=("p","mean"), observed=("y","mean"), n=("y","size")).reset_index(drop=True)

def decile_hist(series):
    s = np.clip(np.array(series)*100.0, 0, 100)
    bins = np.arange(0, 110, 10)
    hist, _ = np.histogram(s, bins=bins)
    return pd.DataFrame({"bin_start": bins[:-1], "bin_end": bins[1:], "count": hist})

# ========== SWEEPS ==========
def pair_rule_predict(hh, pp, th_h, th_p):
    return ((hh*100.0 >= th_h) | (pp*100.0 >= th_p)).astype(int)

def sweep_pair_thresholds(hh, pp, y_true, h_range=range(0,101), p_range=range(0,101)):
    rows = []
    for th in tqdm(h_range, desc="Sweep pair: hentai", unit="%"):
        H = (hh*100.0 >= th)
        for tpv in p_range:
            P = (pp*100.0 >= tpv)
            y_pred = (H | P).astype(int)
            m = metrics_from_cm(y_true, y_pred)
            rows.append((th, tpv, m["precision"], m["recall"], m["f1"], m["accuracy"],
                         m["recall"], m["fpr"], m["specificity"], m["tp"], m["fp"], m["tn"], m["fn"]))
    return pd.DataFrame(rows, columns=[
        "hentai_thresh_pct","porn_thresh_pct","precision","recall","f1","accuracy",
        "tpr","fpr","specificity","tp","fp","tn","fn"
    ])

def sweep_single_class(series_prob, y_true_bin, t_start=SINGLE_SWEEP_START, t_end=SINGLE_SWEEP_END):
    rows = []
    s = np.asarray(series_prob)*100.0
    for t in tqdm(range(t_start, t_end+1), desc="Sweep single", unit="%"):
        y_pred = (s >= t).astype(int)
        m = metrics_from_cm(y_true_bin, y_pred)
        m.update({'threshold_pct': t})
        rows.append(m)
    return pd.DataFrame(rows)

def sweep_vary_one_fix_other(h_prob, p_prob, y_true_bin, vary='hentai', fixed_other=20,
                             t_start=SINGLE_SWEEP_START, t_end=SINGLE_SWEEP_END):
    rows = []
    H = np.asarray(h_prob)*100.0
    P = np.asarray(p_prob)*100.0
    desc = f"Sweep vary {vary} fix {fixed_other}"
    for t in tqdm(range(t_start, t_end+1), desc=desc, unit="%"):
        if vary == 'hentai':
            y_pred = ((H >= t) | (P >= fixed_other)).astype(int)
            row = {'vary':'hentai', 'threshold_pct':t, 'fixed_other_pct':fixed_other}
        else:
            y_pred = ((H >= fixed_other) | (P >= t)).astype(int)
            row = {'vary':'porn', 'threshold_pct':t, 'fixed_other_pct':fixed_other}
        m = metrics_from_cm(y_true_bin, y_pred)
        row.update(m)
        rows.append(row)
    return pd.DataFrame(rows)

def choose_best_pair(df_pairs):
    df = df_pairs.sort_values(by=["f1","precision","fpr"], ascending=[False, False, True]).head(1)
    r = df.iloc[0]
    return dict(
        hentai_thresh_pct=int(r["hentai_thresh_pct"]),
        porn_thresh_pct=int(r["porn_thresh_pct"]),
        f1=float(r["f1"]),
        precision=float(r["precision"]),
        recall=float(r["recall"]),
        fpr=float(r["fpr"]),
        accuracy=float(r["accuracy"]),
        tp=int(r["tp"]), fp=int(r["fp"]), tn=int(r["tn"]), fn=int(r["fn"])
    )

def choose_best_from_sweep(df, vary_name):
    df2 = df.sort_values(by=["f1","precision","fpr"], ascending=[False, False, True]).head(1)
    r = df2.iloc[0]
    return dict(
        vary=vary_name,
        threshold_pct=int(r["threshold_pct"]),
        f1=float(r["f1"]),
        precision=float(r["precision"]),
        recall=float(r["recall"]),
        fpr=float(r["fpr"]),
        accuracy=float(r["accuracy"]),
        tp=int(r["tp"]), fp=int(r["fp"]), tn=int(r["tn"]), fn=int(r["fn"])
    )

# ========== MAIN ==========
def main():
    set_seed()
    ensure_outdir()

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device} | AMP: {USE_AMP and device.type=='cuda'} | Batch: {BATCH_SIZE}")

    # Load list
    paths, labels = load_image_paths(DATA_DIR)
    if not paths:
        raise SystemExit("No images found under data/ with subfolders: " + ", ".join(LABELS))
    labels_by_path = dict(zip(paths, labels))

    # Load model
    model = CustomEfficientNetB0(len(LABELS)).to(device)
    load_checkpoint(model, MODEL_PATH, os.path.join(OUT_DIR, "load_state_log.txt"))
    model.eval()

    # Inference (batched) with progress, keep only readable images
    df_probs, kept_paths = run_inference_batched(model, device, paths)

    # Align truth ONLY for kept paths
    df_probs = df_probs.reset_index(drop=True)
    df_probs["true_label"] = df_probs["path"].map(labels_by_path)
    y_true_bin = df_probs["true_label"].isin(["hentai","porn"]).astype(int).values

    # NSFW scores
    df_probs["hentai_prob"] = df_probs["prob_hentai"]
    df_probs["porn_prob"]   = df_probs["prob_porn"]
    df_probs["nsfw_max"]    = df_probs[["hentai_prob","porn_prob"]].max(axis=1)

    # Save per-image scores
    df_probs[["path","true_label","hentai_prob","porn_prob","nsfw_max"]].to_csv(
        os.path.join(OUT_DIR,"per_image_scores.csv"), index=False
    )

    # Quick info
    total, kept = len(paths), len(df_probs)
    dropped = total - kept
    print(f"Inference done. Kept {kept}/{total} images, dropped {dropped} unreadable.")

    # Histograms
    h_hist_all = decile_hist(df_probs["hentai_prob"])
    p_hist_all = decile_hist(df_probs["porn_prob"])
    safe_mask = ~df_probs["true_label"].isin(["hentai","porn"])
    h_hist_safe = decile_hist(df_probs.loc[safe_mask,"hentai_prob"])
    p_hist_safe = decile_hist(df_probs.loc[safe_mask,"porn_prob"])
    h_hist_all.to_csv(os.path.join(OUT_DIR,"hist_hentai_all.csv"), index=False)
    p_hist_all.to_csv(os.path.join(OUT_DIR,"hist_porn_all.csv"), index=False)
    h_hist_safe.to_csv(os.path.join(OUT_DIR,"hist_hentai_on_safe_classes.csv"), index=False)
    p_hist_safe.to_csv(os.path.join(OUT_DIR,"hist_porn_on_safe_classes.csv"), index=False)

    plt.figure()
    plt.bar(h_hist_all["bin_start"], h_hist_all["count"], width=9, align="edge", alpha=0.8, label="hentai (all)")
    plt.bar(p_hist_all["bin_start"], p_hist_all["count"], width=9, align="edge", alpha=0.5, label="porn (all)")
    plt.xticks(range(0,101,10)); plt.xlabel("score bin start (%)"); plt.ylabel("count"); plt.legend(); plt.title("Hesitation histogram (all images)")
    save_plot(os.path.join(OUT_DIR,"hist_all.png"))

    plt.figure()
    plt.bar(h_hist_safe["bin_start"], h_hist_safe["count"], width=9, align="edge", alpha=0.8, label="hentai (safe)")
    plt.bar(p_hist_safe["bin_start"], p_hist_safe["count"], width=9, align="edge", alpha=0.5, label="porn (safe)")
    plt.xticks(range(0,101,10)); plt.xlabel("score bin start (%)"); plt.ylabel("count"); plt.legend(); plt.title("Hesitation on SAFE classes")
    save_plot(os.path.join(OUT_DIR,"hist_safe.png"))

    # ROC / PR on max(hentai,porn)
    fpr, tpr, _ = roc_curve(y_true_bin, df_probs["nsfw_max"].values)
    precs, recs, _ = precision_recall_curve(y_true_bin, df_probs["nsfw_max"].values)
    roc_auc = auc(fpr, tpr)
    pr_auc  = auc(recs, precs)
    pd.DataFrame({"fpr":fpr, "tpr":tpr}).to_csv(os.path.join(OUT_DIR,"roc_points.csv"), index=False)
    pd.DataFrame({"recall":recs, "precision":precs}).to_csv(os.path.join(OUT_DIR,"pr_points.csv"), index=False)
    with open(os.path.join(OUT_DIR,"auc.txt"),"w") as f:
        f.write(f"ROC_AUC={roc_auc:.6f}\nPR_AUC={pr_auc:.6f}\n")

    plt.figure(); plt.plot(fpr,tpr); plt.plot([0,1],[0,1],'--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC (AUC={roc_auc:.4f})")
    save_plot(os.path.join(OUT_DIR,"roc.png"))
    plt.figure(); plt.plot(recs,precs); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR (AUC={pr_auc:.4f})")
    save_plot(os.path.join(OUT_DIR,"pr.png"))

    # Calibration (max score)
    br = brier_score(y_true_bin, df_probs["nsfw_max"].values)
    cal = calibration_bins(y_true_bin, df_probs["nsfw_max"].values, n_bins=10)
    cal.to_csv(os.path.join(OUT_DIR,"calibration_bins.csv"), index=False)
    with open(os.path.join(OUT_DIR,"brier.txt"),"w") as f:
        f.write(f"Brier={br:.6f}\n")
    plt.figure(); plt.plot(cal["predicted"], cal["observed"], marker="o"); plt.plot([0,1],[0,1],'--'); plt.xlabel("Predicted"); plt.ylabel("Observed"); plt.title(f"Calibration (Brier={br:.4f})")
    save_plot(os.path.join(OUT_DIR,"calibration.png"))

    # Baseline pairs quick check
    baseline = []
    for th_h, th_p in BASELINE_PAIRS:
        y_pred = pair_rule_predict(df_probs["hentai_prob"].values, df_probs["porn_prob"].values, th_h, th_p)
        m = metrics_from_cm(y_true_bin, y_pred)
        m.update({"hentai_thresh_pct":th_h, "porn_thresh_pct":th_p})
        baseline.append(m)
    pd.DataFrame(baseline).to_csv(os.path.join(OUT_DIR,"confusion_baseline_pairs.csv"), index=False)

    # Pair sweep full 0..100
    pair = sweep_pair_thresholds(df_probs["hentai_prob"].values, df_probs["porn_prob"].values, y_true_bin,
                                 h_range=range(0,101), p_range=range(0,101))
    pair.to_csv(os.path.join(OUT_DIR,"sweep_pair_hentai_porn.csv"), index=False)
    best_pair = choose_best_pair(pair)
    with open(os.path.join(OUT_DIR,"best_threshold_pair.json"),"w",encoding="utf-8") as f:
        json.dump(best_pair, f, ensure_ascii=False, indent=2)

    # Single-class sweeps 20..40
    h_single = sweep_single_class(df_probs["hentai_prob"].values, y_true_bin, SINGLE_SWEEP_START, SINGLE_SWEEP_END)
    p_single = sweep_single_class(df_probs["porn_prob"].values,   y_true_bin, SINGLE_SWEEP_START, SINGLE_SWEEP_END)
    h_single.to_csv(os.path.join(OUT_DIR, "sweep_hentai_20_40_independent.csv"), index=False)
    p_single.to_csv(os.path.join(OUT_DIR, "sweep_porn_20_40_independent.csv"),   index=False)

    # Vary-one-fix-other (ใกล้เคียงการใช้จริง OR-rule)
    h_fixp20 = sweep_vary_one_fix_other(df_probs["hentai_prob"].values, df_probs["porn_prob"].values, y_true_bin,
                                        vary='hentai', fixed_other=20,
                                        t_start=SINGLE_SWEEP_START, t_end=SINGLE_SWEEP_END)
    p_fixh25 = sweep_vary_one_fix_other(df_probs["hentai_prob"].values, df_probs["porn_prob"].values, y_true_bin,
                                        vary='porn', fixed_other=25,
                                        t_start=SINGLE_SWEEP_START, t_end=SINGLE_SWEEP_END)
    h_fixp20.to_csv(os.path.join(OUT_DIR, "sweep_hentai_20_40_with_porn_fixed20.csv"), index=False)
    p_fixh25.to_csv(os.path.join(OUT_DIR, "sweep_porn_20_40_with_hentai_fixed25.csv"), index=False)

    # Best suggestion for single sweeps
    best_h_single = choose_best_from_sweep(h_single.assign(vary='hentai'), 'hentai')
    best_p_single = choose_best_from_sweep(p_single.assign(vary='porn'), 'porn')

    # Best suggestion for vary-one-fix-other
    best_h_fixp20 = choose_best_from_sweep(h_fixp20.assign(vary='hentai_fix_porn20'), 'hentai_fix_porn20')
    best_p_fixh25 = choose_best_from_sweep(p_fixh25.assign(vary='porn_fix_hentai25'), 'porn_fix_hentai25')

    # Summary file
    with open(os.path.join(OUT_DIR,"README_summary.md"),"w",encoding="utf-8") as f:
        f.write("# NSFW Eval Summary\n")
        f.write(f"- Device: {device}, AMP: {USE_AMP and device.type=='cuda'}, Batch: {BATCH_SIZE}\n")
        f.write(f"- Images evaluated: {len(df_probs)} (dropped {dropped} / total {total})\n")
        f.write(f"- ROC_AUC={roc_auc:.6f}, PR_AUC={pr_auc:.6f}, Brier={br:.6f}\n")
        f.write("- Outputs:\n")
        f.write("  - per_image_scores.csv\n")
        f.write("  - hist_* (all, safe)\n")
        f.write("  - roc.png / pr.png / calibration.png\n")
        f.write("  - confusion_baseline_pairs.csv\n")
        f.write("  - sweep_pair_hentai_porn.csv\n")
        f.write("  - sweep_hentai_20_40_independent.csv / sweep_porn_20_40_independent.csv\n")
        f.write("  - sweep_hentai_20_40_with_porn_fixed20.csv / sweep_porn_20_40_with_hentai_fixed25.csv\n")
        f.write("  - best_threshold_pair.json\n\n")

        f.write("## Recommended thresholds\n")
        f.write("- Best pair (OR-rule, by F1 then precision then low FPR):\n")
        f.write(f"  - hentai>={best_pair['hentai_thresh_pct']}% OR porn>={best_pair['porn_thresh_pct']}%\n")
        f.write(f"  - F1={best_pair['f1']:.4f}, Precision={best_pair['precision']:.4f}, Recall={best_pair['recall']:.4f}, FPR={best_pair['fpr']:.4f}\n\n")

        f.write("- Best single sweeps (independent):\n")
        f.write(f"  - hentai single @ {best_h_single['threshold_pct']}% -> F1={best_h_single['f1']:.4f}, P={best_h_single['precision']:.4f}, R={best_h_single['recall']:.4f}, FPR={best_h_single['fpr']:.4f}\n")
        f.write(f"  - porn   single @ {best_p_single['threshold_pct']}% -> F1={best_p_single['f1']:.4f}, P={best_p_single['precision']:.4f}, R={best_p_single['recall']:.4f}, FPR={best_p_single['fpr']:.4f}\n\n")

        f.write("- Best vary-one-fix-other (OR-rule):\n")
        f.write(f"  - hentai varying, porn fixed 20% -> hentai @ {best_h_fixp20['threshold_pct']}% (F1={best_h_fixp20['f1']:.4f}, P={best_h_fixp20['precision']:.4f}, R={best_h_fixp20['recall']:.4f}, FPR={best_h_fixp20['fpr']:.4f})\n")
        f.write(f"  - porn   varying, hentai fixed 25% -> porn @ {best_p_fixh25['threshold_pct']}% (F1={best_p_fixh25['f1']:.4f}, P={best_p_fixh25['precision']:.4f}, R={best_p_fixh25['recall']:.4f}, FPR={best_p_fixh25['fpr']:.4f})\n\n")

        f.write("## Notes\n")
        f.write("- Labels = folder names. If your dataset has mislabels, metrics will follow the mess. Fix labels for best truth.\n")
        f.write("- 'sexy' counted as SAFE by default. Change mapping here if policy differs.\n")

    print(f"\nDone. Results saved in: {OUT_DIR}")
    print("Best pair:", best_pair)
    print("Best hentai single:", best_h_single)
    print("Best porn single:", best_p_single)
    print("Best hentai (vary, porn=20):", best_h_fixp20)
    print("Best porn (vary, hentai=25):", best_p_fixh25)

if __name__ == "__main__":
    main()
