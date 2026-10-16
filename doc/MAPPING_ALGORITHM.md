# Incremental mapping and pose-graph optimization

## Run

Baseline incremental mapping:

```bash
python3 launch/map.py --src data/usegeo/fusion/test/fused/ --save
```

This is the fast default. It saves a cumulative PCD after every frame, but
builds the mesh, occupancy files, orthographic map, persistent state, and final
trajectory only once during finalization. To reproduce the older, much slower
behavior and save every derived product at every frame, add
`--save-derived-steps`.

Interactive processing (`A` advances one frame, `Q` closes):

```bash
python3 launch/map.py --src data/usegeo/fusion/test/fused/ --save --visualize
```

Also run the separate loop-closure/global-optimization pass:

```bash
python3 launch/map.py \
  --src data/usegeo/fusion/test/fused/ \
  --save \
  --loop-closure
```

The incremental outputs stay in `mapping/test/`. The second pass writes only to
`mapping/test/optimized/`, so the two results can be compared directly.

Do not use `--on-video` during the earlier projection/fusion run. Each fused PCD
is already OPK-rotated into world/NWU axes but remains camera-centered. Mapping
therefore initializes frame `i` with:

```text
translation_i = pose_utm_i - pose_utm_0
```

and estimates a residual rigid SE(3) correction. This avoids applying IMU
rotation or geographic translation twice and avoids large UTM values inside
ICP.

## Incremental algorithm

For every frame:

1. Load the colored fused PCD, sibling generated-ground PCD, projected RGB, and
   raw `pose_utm` entry.
2. Place it using the relative-position prior and measure overlap with a cropped
   cumulative submap.
3. Visually register against nearby accepted keyframes.
4. Use the generated ground only to constrain residual roll, pitch, and motion
   normal to the ground. A nearly planar surface cannot determine yaw or
   translation along the plane.
5. Refine with robust coarse-to-fine point-to-plane ICP.
6. Accept only when overlap, ICP fitness/RMSE, prior-deviation limits, and
   visual/geometry agreement pass. A rejected frame never changes the trusted
   map.
7. Confidence-fuse accepted geometry and color into one surfel per map voxel.
   Colors are averaged in linear RGB and persistent large disagreements are
   rejected.
8. Update 3D occupancy rays and save the cumulative PCD plus registration
   reports. Build the colored surface mesh, orthographic RGB/mask, trajectory,
   and persistent state once at the end.

The mapper retains only a bounded voxel sample plus lifted SIFT points for each
keyframe. It does not retain every full-resolution PCD/image, which keeps the
eight large images from exhausting memory.

## How SIFT is matched

SIFT itself detects keypoints and computes descriptors. The mapper turns those
features into a geometric match as follows:

1. Perform brute-force two-nearest-neighbor descriptor matching in both image
   directions.
2. Apply Lowe's ratio test in both directions.
3. Keep only mutual source/target pairs.
4. Fit both a fundamental matrix and homography with robust 2D RANSAC and retain
   the model with more inliers.
5. Lift surviving image pixels to their corresponding organized PCD 3D points.
6. Run a second 3D RANSAC to estimate one rigid SE(3) correction and reject
   inconsistent monocular depths.
7. Use that correction only as the seed for ground alignment and ICP; it is not
   accepted on image evidence alone.

Per-frame visual diagnostics are in `reports/visual/`:

- `*_matches.png`: green 2D RANSAC inliers and red rejected mutual matches;
- `*_rgb_overlay.png`: the current RGB warped onto the chosen keyframe;
- `*_mask_overlay.png`: cyan current match footprint, magenta target footprint,
  and white overlap.

The report images are capped at 1600 pixels on their longest side even when the
source photographs are much larger.

## Timing and performance

Every frame prints a monotonic wall-clock breakdown with six decimal places:

```text
[timing] frame 000003: load=0.081234s mapping=2.345678s products=0.000000s saving=0.042100s reports=0.190200s total=2.659212s
```

- `load`: reading the PCD, RGB, ground cloud, and metadata;
- `mapping`: SIFT/RANSAC, ground/ICP registration, occupancy ray carving,
  voxel/color fusion, and keyframe update;
- `products`: mesh and orthographic RGB-array construction (zero per frame in
  the fast default);
- `saving`: non-report file serialization, including the cumulative step PCD;
- `reports`: visual overlays, geometry diagnostics, and frame JSON.

Exact per-frame values are saved to `timings.jsonl`; totals, means, maxima,
final-output time, loop-closure time, and phase definitions are saved to
`timing_summary.json` and embedded in `summary.json`. Viewer waiting, console
printing, and writing the timing log itself are excluded. The two principal
speed changes are eliminating duplicate `map.pcd` writes during each step and
not regenerating large derived products until finalization. Occupancy sampling
also defaults to 5,000 endpoint rays at one-voxel steps; increase it only when
the timing log shows that the extra ray density is affordable.

## View PCD and PLY output

The standalone viewer supports normal point-cloud PCD/PLY files, triangle-mesh
PLY files, this mapper's trajectory PLY, and occupancy PCDs:

```bash
python3 launch/view3d.py data/usegeo/mapping/test/map.pcd
python3 launch/view3d.py data/usegeo/mapping/test/map_mesh.ply
python3 launch/view3d.py data/usegeo/mapping/test/trajectory.ply
python3 launch/view3d.py data/usegeo/mapping/test/occupancy/occupancy_grid_3d.pcd
```

For an occupancy PCD, the sibling JSON supplies the voxel size and the viewer
renders every visible center as a cube. Use `--points` to inspect its raw point
representation, or use `--voxels --voxel-size 0.75` for any other PCD.

## Geometry report

Each `reports/geometry/*_geometry.json` records ICP pyramid levels, fitness,
RMSE, overlap, median error, correction magnitude, and acceptance status.
`*_alignment.ply` overlays the target in light orange with the registered source
colored green-to-red by nearest-neighbor error. This is separate from the main
per-frame JSON so geometry can be inspected without parsing visual evidence.

## Loop closure and global pose-graph optimization

A pose graph represents every accepted keyframe as a node containing its global
SE(3) pose. Consecutive frames create odometry edges. Non-adjacent frames become
loop candidates only when their current camera centers are within the configured
search radius. A candidate becomes a loop edge only after the same visual 2D/3D
RANSAC and geometric ICP checks pass.

The optimizer fixes the first node, then jointly adjusts every other node to
minimize all relative-edge errors. Translation and rotation have separate noise
scales, and a robust loss limits the influence of a bad loop. The map, mesh,
occupancy, RGB map, and trajectory are rebuilt from the optimized poses under
`optimized/`; the incremental map is not overwritten.

Inspect `optimized/comparison.json`:

- `accepted_loops` must be greater than zero for optimization to have new
  information;
- compare `loop_residual_before` and `loop_residual_after`;
- `loop_residual_improved` says whether verified-loop consistency decreased;
- maximum translation/rotation changes show how much the trajectory moved.

If no loop is independently verified, the optimized output intentionally equals
the baseline and the status is `no_verified_loop`. Your supplied eight camera
positions do not return to the starting position; useful non-adjacent constraints
therefore depend on the aerial footprints actually overlapping. Even when the
loop residual improves, this is an internal-consistency result, not proof of
lower absolute error without control points or ground truth.

## Outputs

```text
mapping/test/
├── map.pcd
├── map_mesh.ply
├── trajectory.ply
├── trajectory.jsonl
├── state.npz
├── origin.json
├── summary.json
├── timings.jsonl
├── timing_summary.json
├── steps/*.pcd
├── map_2d/
│   ├── map_rgb.png
│   ├── map_mask.png
│   ├── map_trajectory.png
│   └── map_data.npz
├── occupancy/
│   ├── occupancy_grid_3d.npz
│   ├── occupancy_grid_3d.pcd
│   └── occupancy_grid_3d.json
├── reports/
│   ├── *.json
│   ├── visual/*_matches.png, *_rgb_overlay.png, *_mask_overlay.png
│   └── geometry/*_geometry.json, *_alignment.ply
└── optimized/
    ├── map.pcd
    ├── map_mesh.ply
    ├── map_rgb.png, map_mask.png, map_trajectory.png
    ├── occupancy_grid_3d.npz, .pcd, .json
    ├── trajectory.ply
    ├── poses.npz
    ├── pose_graph.json
    └── comparison.json
```

With `--save-derived-steps`, `mesh_steps/*.ply`, `map_2d/steps/*`, and
`occupancy/steps/*.{npz,pcd,json}` are additionally generated.

## Tuning

Defaults can be overridden under `mapping` in `config.yaml`:

```yaml
mapping:
  position_mode: relative
  voxel_size: 0.5
  min_overlap: 0.05
  max_icp_rmse: 1.25

  save_mesh: true
  save_derived_steps: false
  mesh_grid_size: 0.5
  mesh_max_height_jump: 3.0

  save_occupancy: true
  occupancy_voxel_size: 1.0
  occupancy_max_rays_per_frame: 5000
  occupancy_max_ray_length: 250.0
  occupancy_ray_step_factor: 1.0

  save_map_2d: true
  map_2d_resolution: 0.5

  run_loop_closure: false
  loop_min_separation: 3
  loop_search_radius: 25.0
  loop_min_overlap: 0.12
  loop_min_inliers: 15
```

The supplied poses span about 41 m, with consecutive motion of roughly
3.5–11.5 m. Start with `position_mode: relative`. For a strong registration,
prefer overlapping frames with `overlap_after > 0.30`, RMSE below one map voxel,
well-distributed visual inliers, and agreement between visual and ICP transforms.
`position_only_component` is usable GNSS placement but contains no measured
cross-frame alignment.

## Fusion decision and suggested improvements

Fusion was intentionally left unchanged. Its current output already satisfies
mapping's coordinate contract: metric-scaled, world-oriented, camera-centered,
colored, image-organized, and accompanied by generated ground. Applying position
inside fusion would risk scaling the translation and mapping would then apply it
again.

The highest-value future fusion improvements are:

1. Replace the current single extreme `nanmin` scale anchor with a masked robust
   median/percentile or Huber scale estimate tied to AGL.
2. Preserve real low-frequency terrain slope; remove only estimated depth bias
   instead of forcing every fitted ground surface perfectly flat.
3. Export per-pixel depth/ground confidence and use it in 3D RANSAC, ICP, voxel
   fusion, and occupancy log-odds.
4. Reject dynamic objects and depth discontinuities before surface fitting.
5. Calibrate camera-to-IMU extrinsics and camera/GNSS timestamps, then validate
   on withheld depth/control points.

These can improve the map more than tighter ICP thresholds, but they should be
benchmarked against your `eval/gt` data before replacing the working fusion
output.

## Validation

`tests/test_mapping.py` contains nine tests: robust rigid RANSAC, ground-plane
alignment, confidence/color voxel fusion, 3D ray occupancy states, mesh and RGB
map generation, nonblocking viewer behavior, two-frame ICP, and a complete
eight-frame CLI run. The CLI test reloads all main outputs, checks timing logs,
renders the occupancy PCD as voxels, loads mesh/trajectory PLY files, checks the
diagnostic images, verifies a loop edge, and verifies that the optimized loop
residual is lower. It passes with Open3D 0.19 and OpenCV 5.0 (the declared
minimum remains OpenCV 4.5):

```bash
python3 -m unittest discover -s tests -v
```

The attachments provide the real directory tree, configuration, and metadata,
but not the PCD/image bytes. The exact real reconstruction therefore still needs
to be run by you; the supplied names, paths, UTM trajectory, launcher behavior,
and output contracts were exercised with an eight-frame organized synthetic
sequence.
