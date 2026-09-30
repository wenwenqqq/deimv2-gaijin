#!/usr/bin/env python3
"""
Scan each experiment folder under /workspace/cpfs-data/deimv2/outputs/,
extract the top-3 epochs by (AP) @[ IoU=0.50:0.95 | area=all | maxDets=100 ],
and write a summary to top3_ap_results.txt.
"""

import json
import os
import re
from pathlib import Path

OUTPUTS_DIR = Path("/mnt/e/Experiment/huya-deimv2/deimv2/outputs")
OUTPUT_FILE = OUTPUTS_DIR / "top3_ap_results.txt"

# COCO eval bbox index mapping
COCO_KEYS = [
    "AP @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
    "AP @[ IoU=0.50      | area=   all | maxDets=100 ]",
    "AP @[ IoU=0.75      | area=   all | maxDets=100 ]",
    "AP @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
    "AP @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
    "AP @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
    "AR @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ]",
    "AR @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ]",
    "AR @[ IoU=0.50:0.95 | area=   all | maxDets=100 ]",
    "AR @[ IoU=0.50:0.95 | area= small | maxDets=100 ]",
    "AR @[ IoU=0.50:0.95 | area=medium | maxDets=100 ]",
    "AR @[ IoU=0.50:0.95 | area= large | maxDets=100 ]",
]

lines_out = []

# Discover experiment folders (directories only, skip non-folder files like .sh)
exp_dirs = sorted(
    [d for d in OUTPUTS_DIR.iterdir() if d.is_dir()],
    key=lambda d: d.name,
)

for exp_dir in exp_dirs:
    log_path = exp_dir / "log.txt"
    if not log_path.exists():
        continue

    # Parse all JSON lines, collect (ap, epoch, eval_array) tuples
    entries = []
    with open(log_path, "r") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            eval_bbox = obj.get("test_coco_eval_bbox")
            if eval_bbox is None or not isinstance(eval_bbox, list) or len(eval_bbox) < 1:
                continue
            epoch = obj.get("epoch", "?")
            ap = eval_bbox[0]  # main AP metric
            entries.append((ap, epoch, eval_bbox))

    if not entries:
        lines_out.append(f"Experiment: {exp_dir.name}  -->  NO EVAL DATA\n")
        continue

    # Sort descending by AP, take top 3
    entries.sort(key=lambda x: x[0], reverse=True)
    top3 = entries[:3]

    lines_out.append(f"{'='*120}")
    lines_out.append(f"Experiment: {exp_dir.name}")
    lines_out.append(f"{'='*120}")

    for rank, (ap, epoch, eval_arr) in enumerate(top3, 1):
        lines_out.append(f"  --- Top {rank} (epoch={epoch}, AP={ap:.6f}) ---")
        for idx, key in enumerate(COCO_KEYS):
            if idx < len(eval_arr):
                lines_out.append(f"    {key} = {eval_arr[idx]:.6f}")
            else:
                lines_out.append(f"    {key} = N/A")
        lines_out.append("")

    # Also print top AP summary
    lines_out.append(f"  >> Top-3 AP summary: "
                     f"{'  |  '.join(f'#{i+1}: {top3[i][0]:.6f} (ep {top3[i][1]})' for i in range(len(top3)))}")
    lines_out.append("")

# Write output
with open(OUTPUT_FILE, "w") as f:
    f.write("\n".join(lines_out))

print(f"Done! Results written to: {OUTPUT_FILE}")
print(f"Processed {sum(1 for d in exp_dirs if (d / 'log.txt').exists())} experiments.")
