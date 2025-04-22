# vim: expandtab:ts=4:sw=4
import argparse
import os
import deep_sort_app
import numpy as np
from bayes_opt import BayesianOptimization
import motmetrics as mm

# ——— NumPy≥2.0 compatibility for motmetrics ———
if not hasattr(np, "asfarray"):
    np.asfarray = lambda x: np.asarray(x, dtype=float)

def run_sort(deltas):
    sequence = "MOT16-13"
    mot_dir = "MOT16/train"
    detection_dir = "resources/detections/MOT16_POI_train"
    output_dir = "outputs"

    sequence_dir = os.path.join(mot_dir, sequence)
    detection_file = os.path.join(detection_dir, "%s.npy" % sequence)
    output_file = os.path.join(output_dir, "%s.txt" % sequence)
    
    deep_sort_app.run(
        sequence_dir, detection_file, output_file, min_confidence=0.5,
        nms_max_overlap=1.0, min_detection_height=0,
        max_cosine_distance=0.2, nn_budget=None, display=False,
        delta=deltas
    )

def compute_metrics():
    mot16_sequences = ["MOT16/train/MOT16-13/gt/gt.txt"]
    res_files = ["outputs/MOT16-13.txt"]
    
    mh = mm.metrics.create()
    accs = []
    
    for gt_file, res_file in zip(mot16_sequences, res_files):
        gt = mm.io.loadtxt(gt_file, fmt="mot15-2D", min_confidence=0.5)
        res = mm.io.loadtxt(res_file, fmt="mot15-2D")
        acc = mm.utils.compare_to_groundtruth(gt, res, 'iou', distth=0.5)
        accs.append(acc)
    
    summary = mh.compute_many(
        accs,
        metrics=mm.metrics.motchallenge_metrics,
        names=["MOT16-13"]
    )
    return summary



deltas = np.array([1.489, 0.1412, 5.202, 9.831])
run_sort(deltas=deltas)

