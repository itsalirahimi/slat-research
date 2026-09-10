# Incremental mapping implementation

## Run

Use the same fused directory shown in the sample tree:

```bash
python3 launch/map.py --src data/usegeo/fusion/test/fused/ --save
```

The default `position_mode` is `relative`. To test without positions:

```bash
python3 launch/map.py --src data/usegeo/fusion/test/fused/ --save --position-mode off
```

Do **not** use `--on-video` in the earlier projection/fusion run. The fused PCDs
are already rotated by OPK into the project world axes, but remain centered at
their camera. Mapping therefore places frame `i` with:

```text
translation_i = pose_utm_i - pose_utm_0
```

and estimates only a residual rigid SE(3) correction. This avoids applying the
IMU rotation twice and keeps large UTM values out of registration calculations.

## Algorithm

For every frame, the mapper performs these steps:

1. Load `fusion/<sequence>/fused/<name>.pcd`, the optional sibling
   `ground/<name>.pcd`, `projection/<sequence>/rgb/<name>.png`, and the raw
   three-value `pose_utm` from `data.json`.
2. Apply the relative position prior and measure nearest-neighbor overlap with
   the cumulative local map.
3. Match the current image against nearby accepted keyframes using SIFT,
   mutual Lowe-ratio filtering, 2D fundamental/homography RANSAC, then lift
   valid pixels into the organized PCD and run a second 3D rigid RANSAC.
4. Use the generated ground (GEP) surfaces only for their observable terms:
   residual roll/pitch and translation normal to the ground. GEP is not allowed
   to invent yaw or translation parallel to a flat surface.
5. Refine the result with robust coarse-to-fine point-to-plane ICP against a
   cropped cumulative submap.
6. Accept the frame only when overlap, ICP fitness/RMSE, sensor-prior limits,
   and visual/geometry agreement pass. A failed registration does not modify
   the trusted map. With positions enabled, a true non-overlapping frame is
   added as a position-only connected component; with positions disabled it is
   rejected.
7. Fuse accepted observations into a voxel map. Geometry is confidence
   weighted. Colors are averaged in linear RGB, downweighted when inconsistent,
   and rejected after repeated large color disagreement. A rigid correction is
   applied per frame; confidence is per observation/voxel, so geometry is not
   torn by per-point rotations.

## Outputs

With `--save`, results are written to `data/usegeo/mapping/test/`:

```text
map.pcd                 latest cumulative colored map
state.npz               voxel statistics and all frame transforms
trajectory.jsonl        one registration record per processed frame
origin.json             UTM origin/coordinate metadata
summary.json            accepted/rejected counts
steps/*.pcd             cumulative map after every frame
reports/*.json          per-frame evidence and rejection reason
```

Inspect `summary.json` first. In each frame report, `accepted` must be true;
`status`, `overlap_after`, `rmse_after`, and `message` explain the decision.
`position_only_component` is valid, but means that frame had no measured
overlap and was placed only from `pose_utm`.

## Optional tuning

Defaults work without changing the supplied `config.yaml`. Overrides can be
added under a `mapping` section, for example:

```yaml
mapping:
  position_mode: relative
  voxel_size: 0.5
  min_overlap: 0.05
  max_icp_rmse: 1.25
  max_residual_translation: 10.0
  max_residual_rotation_deg: 12.0
```

The eight supplied poses span about 41.05 m, with consecutive moves from 3.52 m
to 11.50 m. Relative positions are therefore the recommended mode for this
sequence. Start with the defaults; tune thresholds only after inspecting the
JSON reports and map scale.

## Validation performed

`tests/test_mapping.py` covers rigid RANSAC with outliers, GEP plane alignment,
voxel/color fusion, two-frame ICP accumulation, and an end-to-end eight-frame
CLI run using the supplied relative pose trajectory that writes and reloads all
checkpoints. All five tests pass with Open3D 0.19 and OpenCV 4.12:

```bash
python3 -m unittest discover -s tests -v
```

The supplied attachment contains the real eight-frame directory tree and
metadata, but not the PCD/image bytes themselves. Therefore the exact real
sequence could not be executed here; its names, paths, pose origin, and motion
were validated, while registration/output behavior was exercised on generated
organized colored PCDs through the real launcher.
