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
    sequence = "MOT16-02"
    mot_dir = "C:/Users/nalsa/Desktop/DeepSort/deep_sort/MOT16/train"
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
    mot16_sequences = ["MOT16/train/MOT16-02/gt/gt.txt"]
    res_files = ["outputs/MOT16-02.txt"]
    
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
        names=["MOT16-02"]
    )
    return summary

def eval_sort(delta1, delta2, delta3, delta4):
    deltas = np.array([delta1, delta2, delta3, delta4])
    run_sort(deltas=deltas)
    summary = compute_metrics()
    mota = summary.loc['MOT16-02']['mota']
    return mota

# Set up Bayesian optimization
optimizer = BayesianOptimization(
    f=eval_sort,
    pbounds={
        "delta1": (0.1, 10.0),
        "delta2": (0.1, 10.0),
        "delta3": (0.1, 10.0),
        "delta4": (0.1, 10.0),
    },
    verbose=2,
    random_state=42
)

# Run optimization
optimizer.maximize(init_points=5, n_iter=25)

# Get and display results
best_params = optimizer.max['params']
best_mota = optimizer.max['target']

# Final run with best parameters
final_deltas = np.array([
    best_params['delta1'],
    best_params['delta2'],
    best_params['delta3'],
    best_params['delta4']
])
run_sort(deltas=final_deltas)
final_summary = compute_metrics()

print("\n✅ Best parameters found:")
print(f"delta1={best_params['delta1']:.4f}")
print(f"delta2={best_params['delta2']:.4f}")
print(f"delta3={best_params['delta3']:.4f}")
print(f"delta4={best_params['delta4']:.4f}")
print(f"Best MOTA: {best_mota:.4f}")

# Display final metrics
print("\nFinal metrics:")
print(final_summary)
