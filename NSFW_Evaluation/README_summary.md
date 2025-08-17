# NSFW Eval Summary (OOF calibrated)
- Device: cuda, Batch: 64
- Images evaluated: 16013
- ROC_AUC=0.992668, PR_AUC=0.986470
- Brier_raw=0.030577, Brier_oof=0.026111
- ECE_raw=0.030804, ECE_oof=0.011454

## Best thresholds (OOF)
{
  "hentai_thresh_pct": 37,
  "porn_thresh_pct": 41,
  "f1": 0.9469071639661673,
  "precision": 0.9575432811211871,
  "recall": 0.9365047369481959,
  "fpr": 0.01863916033297141,
  "accuracy": 0.9674639355523637,
  "tp": 4646,
  "fp": 206,
  "tn": 10846,
  "fn": 315
}

## Bootstrap 95% CI
{
  "f1_ci": [
    0.9418684964666547,
    0.9509726545320817
  ],
  "precision_ci": [
    0.951469638089807,
    0.9632078719952114
  ],
  "recall_ci": [
    0.9281383573766112,
    0.9423455073959843
  ],
  "fpr_ci": [
    0.016073539106437787,
    0.02118365317432591
  ]
}