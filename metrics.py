import motmetrics as mm
import pandas as pd
import os
import numpy as np

# ——— NumPy≥2.0 compatibility for motmetrics ———
# motmetrics.distances.iou_matrix calls np.asfarray, removed in NumPy 2.0
if not hasattr(np, "asfarray"):
    np.asfarray = lambda x: np.asarray(x, dtype=float)


# Paths
mot16_sequences = [
    "MOT16/train/MOT16-02/gt/gt.txt",
    "MOT16/train/MOT16-04/gt/gt.txt",
    "MOT16/train/MOT16-05/gt/gt.txt",
    "MOT16/train/MOT16-09/gt/gt.txt",
    "MOT16/train/MOT16-10/gt/gt.txt",
    "MOT16/train/MOT16-11/gt/gt.txt",
    "MOT16/train/MOT16-13/gt/gt.txt"
]
res_files = [
    "outputs/MOT16-02.txt",
    "outputs/MOT16-04.txt",
    "outputs/MOT16-05.txt",
    "outputs/MOT16-09.txt",
    "outputs/MOT16-10.txt",
    "outputs/MOT16-11.txt",
    "outputs/MOT16-13.txt"
]



# Initialize metrics accumulator
mh = mm.metrics.create()
accs = []

# Process each sequence
for gt_file, res_file in zip(mot16_sequences, res_files):
    gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=0.5)
    res = mm.io.loadtxt(res_file, fmt="mot15-2D")
    acc = mm.utils.compare_to_groundtruth(gt, res, 'iou', distth=0.5)
    accs.append(acc)

# Compute metrics for all sequences
summary = mh.compute_many(
    accs, 
    metrics=mm.metrics.motchallenge_metrics, 
    names=[f"MOT16-{seq.split('-')[1]}" for seq in mot16_sequences]
)

# Display results
print(mm.io.render_summary(summary,
    formatters=mh.formatters,
    namemap=mm.io.motchallenge_metric_names))
