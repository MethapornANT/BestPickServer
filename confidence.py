# nsfw_confidence_audit_v2.py
# ---------------------------------------------
# หัวข้อหลัก: CONFIG / MODEL / PREPROCESS / SCAN / STATS / EVAL / CSV / MAIN
# ---------------------------------------------

# ============== CONFIG ==============
from pathlib import Path
import os, sys, csv
from collections import defaultdict
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

DATA_DIR   = Path("./data")
MODEL_PATH = Path("./NSFW_Model_archived_20250803-132149/results/20250801-100650 Best Model/efficientnetb0/models/efficientnetb0_best.pth")

LABELS = ['normal','hentai','porn','sexy','anime']
THRESHOLDS = [10,20,30,40,50,60,70,80,90]  # ตารางสรุปแยก H/P/Max
POLICIES = {  # ประเมินนโยบาย OR ระหว่างสองคลาส
    "sym20 (H>=20 or P>=20)": (20, 20),
    "sym25 (H>=25 or P>=25)": (25, 25),
    "sym30 (H>=30 or P>=30)": (30, 30),
    "asym (H>=25 or P>=20)":  (25, 20),
}

OUT_DETAILS_CSV   = Path("nsfw_details.csv")
OUT_BINS_CSV      = Path("nsfw_bins_summary.csv")
OUT_THRESH_CSV    = Path("nsfw_thresholds_summary.csv")
OUT_POLICIES_CSV  = Path("nsfw_policies_summary.csv")
SUPPORTED_EXTS    = {".jpg",".jpeg",".png",".webp",".bmp"}

# ============== MODEL ==============
class CustomEfficientNetB0(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        self.efficientnet = models.efficientnet_b0(weights=None)
        self.efficientnet.classifier[1] = nn.Linear(
            self.efficientnet.classifier[1].in_features, num_classes
        )
    def forward(self, x):
        return self.efficientnet(x)

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CustomEfficientNetB0(num_classes=len(LABELS))
    state = torch.load(MODEL_PATH, map_location=device)
    try:
        model.efficientnet.load_state_dict(state)
    except Exception:
        model.load_state_dict(state)
    model.eval().to(device)
    return model, device

# ============== PREPROCESS ==============
processor = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225]),
])

# ============== SCAN ==============
def list_images(root: Path):
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in SUPPORTED_EXTS:
            yield p

def predict_probs(model, device, img_path: Path):
    img = Image.open(img_path).convert("RGB")
    x = processor(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1).squeeze().cpu().tolist()
    return [round(p*100, 2) for p in probs]

# ============== STATS ==============
def bin_label(v):  # 0-10, 10-20, ..., 90-100
    b = min(9, int(v // 10))
    return f"{b*10:02d}-{(b+1)*10:02d}"

def init_bins():
    return {f"{i*10:02d}-{(i+1)*10:02d}": 0 for i in range(10)}

def run_audit():
    if not DATA_DIR.exists() or not MODEL_PATH.exists():
        print("Path ไม่ถูกต้อง: ตรวจ DATA_DIR / MODEL_PATH")
        sys.exit(1)

    model, device = load_model()
    images = list(list_images(DATA_DIR))
    total = len(images)
    if total == 0:
        print("ไม่พบไฟล์รูปภาพใน data/")
        sys.exit(0)

    bins_hentai = init_bins()
    bins_porn   = init_bins()
    bins_maxhp  = init_bins()

    flagged_h = {t:0 for t in THRESHOLDS}
    flagged_p = {t:0 for t in THRESHOLDS}
    flagged_m = {t:0 for t in THRESHOLDS}

    label_counts = {k:0 for k in LABELS}
    policy_flagged_by_folder = {name: defaultdict(int) for name in POLICIES}
    policy_flagged_total = {name: 0 for name in POLICIES}

    fails = 0
    with OUT_DETAILS_CSV.open("w", newline="", encoding="utf-8") as fdet:
        w = csv.writer(fdet)
        w.writerow(["path","folder_label","pred_label",
                    "prob_normal","prob_hentai","prob_porn","prob_sexy","prob_anime",
                    "hentai_score","porn_score","max_hp"])

        last_pct = 0
        for i, img_path in enumerate(images):
            pct = int(((i+1) / total) * 100)
            if pct > last_pct:
                print(f"\rProgress: {pct:3d}%", end="", flush=True)
                last_pct = pct

            folder_label = img_path.parent.name if img_path.parent.name in LABELS else "unknown"
            try:
                probs = predict_probs(model, device, img_path)
            except Exception:
                fails += 1
                continue

            record = dict(zip(LABELS, probs))
            hentai_score = record["hentai"]
            porn_score   = record["porn"]
            max_hp       = max(hentai_score, porn_score)

            bins_hentai[bin_label(hentai_score)] += 1
            bins_porn[bin_label(porn_score)]     += 1
            bins_maxhp[bin_label(max_hp)]        += 1

            for t in THRESHOLDS:
                if hentai_score >= t: flagged_h[t] += 1
                if porn_score   >= t: flagged_p[t] += 1
                if max_hp       >= t: flagged_m[t] += 1

            for name, (th_h, th_p) in POLICIES.items():
                if (hentai_score >= th_h) or (porn_score >= th_p):
                    policy_flagged_total[name] += 1
                    policy_flagged_by_folder[name][folder_label] += 1

            pred_idx = int(max(range(len(probs)), key=lambda j: probs[j]))
            pred_label = LABELS[pred_idx]
            if folder_label in label_counts: label_counts[folder_label] += 1

            w.writerow([
                str(img_path), folder_label, pred_label,
                record["normal"], record["hentai"], record["porn"], record["sexy"], record["anime"],
                hentai_score, porn_score, max_hp
            ])

    print()
    return {
        "total": total,
        "fails": fails,
        "label_counts": label_counts,
        "bins_h": bins_hentai,
        "bins_p": bins_porn,
        "bins_m": bins_maxhp,
        "flag_h": flagged_h,
        "flag_p": flagged_p,
        "flag_m": flagged_m,
        "policy_flagged_total": policy_flagged_total,
        "policy_flagged_by_folder": policy_flagged_by_folder,
    }

# ============== EVAL (พิมพ์สรุปสำคัญ) ==============
def pct(x, total):
    return 0.0 if total == 0 else round(x/total*100, 2)

def print_bins(title, bins, total):
    print(f"\n-- {title} (counts / %) --")
    for i in range(10):
        b = f"{i*10:02d}-{(i+1)*10:02d}"
        print(f"{b:>7}% : {bins[b]:6d}  ({pct(bins[b], total):5.2f}%)")

def print_threshold_table(title, flagged_dict, total):
    print(f"\n-- {title}: flagged when score >= T --")
    for t in THRESHOLDS:
        print(f"T={t:>2}% : {flagged_dict[t]:6d}  ({pct(flagged_dict[t], total):5.2f}%)")

def print_policy_eval(stats):
    total = stats["total"]
    pos_total = stats["label_counts"].get("hentai",0) + stats["label_counts"].get("porn",0)
    neg_total = total - pos_total

    print("\n================ POLICY EVALUATION ================")
    print(f"Positives (hentai+porn) ~ {pos_total} | Negatives (others) ~ {neg_total}")
    for name, (th_h, th_p) in POLICIES.items():
        flagged_total = stats["policy_flagged_total"][name]
        by_folder = stats["policy_flagged_by_folder"][name]
        tp = by_folder.get("hentai",0) + by_folder.get("porn",0)
        fp = flagged_total - tp
        tpr = pct(tp, pos_total)
        fpr = pct(fp, neg_total)
        print(f"\n{name}")
        print(f"  H-th={th_h} / P-th={th_p} | Flagged: {flagged_total} ({pct(flagged_total,total)}%)")
        print(f"  -> TP on (hentai+porn): {tp} ({tpr}%) | FP on (normal+sexy+anime): {fp} ({fpr}%)")
        print(f"  Breakdown flagged by folder: "
              f"H={by_folder.get('hentai',0)}, P={by_folder.get('porn',0)}, "
              f"N={by_folder.get('normal',0)}, S={by_folder.get('sexy',0)}, A={by_folder.get('anime',0)}")

# ============== CSV (สรุป) ==============
def save_bins_summary(bins_h, bins_p, bins_m):
    with OUT_BINS_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["bin","hentai_count","porn_count","max(h,p)_count"])
        for i in range(10):
            b = f"{i*10:02d}-{(i+1)*10:02d}"
            w.writerow([b, bins_h[b], bins_p[b], bins_m[b]])

def save_thresholds_summary(total, flagged_h, flagged_p, flagged_m):
    with OUT_THRESH_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["threshold","flag_hentai","flag_hentai_%","flag_porn","flag_porn_%","flag_max(h,p)","flag_max(h,p)_%"])
        for t in THRESHOLDS:
            w.writerow([
                t,
                flagged_h[t], pct(flagged_h[t], total),
                flagged_p[t], pct(flagged_p[t], total),
                flagged_m[t], pct(flagged_m[t], total),
            ])

def save_policies_summary(stats):
    total = stats["total"]
    pos_total = stats["label_counts"].get("hentai",0) + stats["label_counts"].get("porn",0)
    neg_total = total - pos_total
    with OUT_POLICIES_CSV.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["policy","H_th","P_th","flagged","flagged_%","TP","TP_%_of_pos","FP","FP_%_of_neg",
                    "flag_H","flag_P","flag_N","flag_S","flag_A"])
        for name, (th_h, th_p) in POLICIES.items():
            flagged_total = stats["policy_flagged_total"][name]
            by_folder = stats["policy_flagged_by_folder"][name]
            tp = by_folder.get("hentai",0) + by_folder.get("porn",0)
            fp = flagged_total - tp
            w.writerow([
                name, th_h, th_p,
                flagged_total, pct(flagged_total,total),
                tp, pct(tp,pos_total),
                fp, pct(fp,neg_total),
                by_folder.get("hentai",0),
                by_folder.get("porn",0),
                by_folder.get("normal",0),
                by_folder.get("sexy",0),
                by_folder.get("anime",0),
            ])

# ============== MAIN ==============
if __name__ == "__main__":
    stats = run_audit()

    # พิมพ์ LOG สำคัญสำหรับวิเคราะห์
    print("\n================ DATASET SUMMARY ================")
    print(f"Scanned: {stats['total']} | Unreadable: {stats['fails']}")
    print("By folder:", ", ".join([f"{k}={stats['label_counts'][k]}" for k in LABELS]))

    print_bins("HENTAI score distribution (per 10%)", stats["bins_h"], stats["total"])
    print_bins("PORN   score distribution (per 10%)", stats["bins_p"], stats["total"])
    print_bins("MAX(H,P) distribution (per 10%)",     stats["bins_m"], stats["total"])

    print_threshold_table("Threshold table (HENTAI)", stats["flag_h"], stats["total"])
    print_threshold_table("Threshold table (PORN)",   stats["flag_p"], stats["total"])
    print_threshold_table("Threshold table (MAX(H,P))", stats["flag_m"], stats["total"])

    print_policy_eval(stats)

    # เซฟ CSV
    save_bins_summary(stats["bins_h"], stats["bins_p"], stats["bins_m"])
    save_thresholds_summary(stats["total"], stats["flag_h"], stats["flag_p"], stats["flag_m"])
    save_policies_summary(stats)

    print("\nCSV saved:",
          OUT_DETAILS_CSV.name, ",",
          OUT_BINS_CSV.name, ",",
          OUT_THRESH_CSV.name, ",",
          OUT_POLICIES_CSV.name)
