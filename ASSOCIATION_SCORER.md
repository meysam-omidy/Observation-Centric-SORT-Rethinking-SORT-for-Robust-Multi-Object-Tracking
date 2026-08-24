# Learned no-ReID association scorer

This experiment learns a small residual for OC-SORT's existing association
cost.  It is not a detector, a ReID model, or a new gate.  The normal IoU gate,
Byte phase split, Kalman predict/update path and Hungarian assignment stay in
place.  The model only ranks pairs that the original tracker already accepts as
valid candidates.

Features are IoU, normalised centre/scale residuals, detector confidence,
direction cost, age/observation state, local detector crowding and bounded KF
uncertainty.  The network is `14 -> 64 -> 32 -> 1` and is evaluated in one
batched call per association phase.

## 1. Export causal candidate data

The exporter can make a deterministic sequence-level 80/20 split for every
dataset. Do not use final reporting/evaluation sequences as the held-out set
while choosing hyperparameters. The exporter runs the real tracker causally; GT
labels only tell it, after the frame, which existing candidate was the source
track's correct detection.

```powershell
python .\export_association_candidates.py `
  --datasets MOT17 MOT20 DanceTrack SportsMOT `
  --split train `
  --output_dir .\association_data `
  --combine `
  --no_use_oru `
  --weights_path ..\motion-predictor\checkpoints\kf_adaptive_kalman_real_low_data_light_transformer_lr_new_kftrackoff2\best_model.pth
```

This writes a reproducible `sequence_split_seed42.json`, two files per dataset,
and—because of `--combine`—`all_datasets_train.npz` and
`all_datasets_heldout.npz`. Use `--seed` to intentionally create a new
split. If a dataset does not provide the requested split, add
`--skip_missing_datasets` and it will be reported then skipped.

Use exactly the motion/Kalman switches of the tracking baseline being improved.
For example, append `--use_confidence_r` if that is the baseline, or
`--no_motion` to train a pure fixed-KF association scorer. The exporter prints
how many positive pairs were not offered or were blocked by the existing gate;
the learned scorer cannot repair those cases.

For compatibility with historical OC-SORT result files, append
`--legacy_post_assignment_iou_gate` to both export and tracking. This uses the
old assign-then-filter IoU behavior; it must be consistent between training and
inference.

## 2. Train

```powershell
python .\train_association_model.py `
  --train_data .\association_data\all_datasets_train.npz `
  --val_data .\association_data\all_datasets_heldout.npz `
  --save_dir .\checkpoints\association_multidataset_v1 `
  --device cuda
```

The objective is listwise softmax cross-entropy: for every track/frame/phase,
the oracle detector for its causal GT identity must rank above all other valid
candidate detections. `training_history.json` reports held-out top-1 candidate
accuracy; it must improve before spending time on full HOTA runs.

## 3. Evaluate as a bounded residual

Start with a conservative weight and tune it only on the held-out sequences:

```powershell
python .\run_tracker.py `
  --dataset DanceTrack --evaluate --no_use_oru `
  --weights_path ..\motion-predictor\checkpoints\kf_adaptive_kalman_real_low_data_light_transformer_lr_new_kftrackoff2\best_model.pth `
  --use_learned_association `
  --association_weights_path .\checkpoints\association_dancetrack_v1\best_model.pth `
  --association_cost_weight 0.05 `
  --association_residual_clip 0.50
```

Sweep `association_cost_weight` over `0.02, 0.05, 0.10, 0.15`; keep the clip at
`0.50` initially. This limits the model's maximum effect to `weight * 0.50`, so
it cannot silently replace OC-SORT's geometric association. Once one setting
wins on held-out sequences, run it once on the final evaluation set and report
HOTA/AssA/IDF1/IDSW plus FPS against the identical baseline.
