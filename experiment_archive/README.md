# 2026-09-30 experiment snapshot

This snapshot contains the local DEIMv2 experiment code, configs, launch scripts,
text logs, and visualizations. The original repository history is retained.
Local IDE/agent state and Python caches are excluded.

## Dataset and checkpoint assets

The matching Release is `experiments-2026-09-30`:
https://github.com/wenwenqqq/deimv2-gaijin/releases/tag/experiments-2026-09-30

Each numbered ZIP is independent. Download and extract every required ZIP into
the repository root; do not concatenate the ZIP files. `annotations-*.zip`
restores the four COCO JSON files, `uavdt-images-*.zip` restores all 40,735 UAVDT
images under `datasets/UAVDT/UAV-benchmark-M`, and `experiments-*.zip` restores
all saved model checkpoints, evaluation objects, pretrained weights, and
TensorBoard events at their original relative paths. No intermediate training
checkpoints were intentionally dropped. See `inventory.json` for the full
experiment list and asset counts. Release files include SHA-256 checksums and
member manifests for verification.

The original YAML paths are preserved as experiment provenance. Before training,
change the dataset YAML `img_folder` paths to
`./datasets/UAVDT/UAV-benchmark-M`, and `ann_file` paths to the restored JSON
files in the repository root. The main split uses `uavdt_realtrain.json` for
training and `uavdt_test.json` for validation/testing. Both Windows and Linux
can use these relative paths when launched from the repository root.

The code commit and Release asset upload are separate steps. Consult the Release
asset list and its completion status before assuming the archive is complete.
