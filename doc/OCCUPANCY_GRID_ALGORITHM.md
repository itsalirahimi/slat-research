# Sparse 3D occupancy-grid algorithm

## Representation and colors

The mapper maintains a sparse probabilistic voxel grid in full XYZ space. The
voxel size is configurable and defaults to 1 m. Only observed voxels are stored;
space that no ray has visited is implicit unknown.

| Stored value | State | Meaning | PCD/voxel-view appearance |
| ---: | --- | --- | --- |
| `0` | free | One or more camera rays traversed the voxel with enough evidence. | Transparent: omitted from the PCD. |
| `50` | known/uncertain | The voxel was observed, but accumulated evidence is not decisive. | Light orange. |
| `100` | occupied | A reconstructed surface endpoint has enough evidence. | Light red. |
| `-1` | unknown | No observation exists; this value is conceptual and not stored as a dense array. | Not rendered. |

This distinguishes the requested **known** state from truly **unknown** space.
Free transparency is implemented by leaving free voxels out of the visualization
PCD; the exact free entries remain present in the NPZ.

## Ray update

For every accepted frame:

1. Transform the camera origin and fused surface points into the corrected map
   frame.
2. Group endpoints by occupancy voxel and keep one deterministic ray per unique
   endpoint voxel. This stops millions of pixels in one image from overwhelming
   later views.
3. Reject zero-length rays and rays longer than
   `occupancy_max_ray_length`.
4. Sample each camera-to-surface segment at a fraction of the voxel size.
5. Mark traversed voxels as free evidence and the final voxel as occupied
   evidence. If a voxel is both traversed and an endpoint in one frame, the
   occupied endpoint wins.
6. Weight each update by the registration confidence and clamp the accumulated
   log-odds so repeated observations cannot grow without bound.

The update is:

```text
L(v) = clamp(L(v) + confidence * evidence)
```

with default free evidence `-0.40`, occupied evidence `+0.85`, free threshold
`-0.20`, and occupied threshold `+0.40`. Values between the two thresholds are
known/uncertain.

## Files

```text
occupancy/
├── occupancy_grid_3d.npz
├── occupancy_grid_3d.pcd
└── occupancy_grid_3d.json
```

The NPZ contains:

- `occupancy3d_keys`: signed integer XYZ voxel keys;
- `occupancy3d_states`: `0`, `50`, or `100` for every stored key;
- `occupancy3d_log_odds`: accumulated probability evidence;
- `occupancy3d_observations` and `occupancy3d_last_seen`;
- voxel size, observed bounds, and the RGBA palette.

The PCD contains colored voxel centers for known/uncertain and occupied cells.
Its sibling JSON summarizes counts, bounds, state values, color policy, and
voxel size. `launch/view3d.py` reads both files and displays the centers as full
cubes. `state.npz` also contains the latest occupancy arrays so mapping state
and trajectory remain in one checkpoint. Add `--save-derived-steps` only if an
occupancy PCD/NPZ/JSON is also needed after every input frame.

## Run and configure

Occupancy is produced automatically when saving:

```bash
python3 launch/map.py --src data/usegeo/fusion/test/fused/ --save
```

Render occupied/known cells as actual cubes:

```bash
python3 launch/view3d.py \
  data/usegeo/mapping/test/occupancy/occupancy_grid_3d.pcd
```

Command-line tuning:

```bash
python3 launch/map.py \
  --src data/usegeo/fusion/test/fused/ \
  --save \
  --occupancy-voxel-size 0.75 \
  --occupancy-max-rays 10000
```

Equivalent `config.yaml` settings:

```yaml
mapping:
  save_occupancy: true
  occupancy_voxel_size: 1.0
  occupancy_free_log_odds: -0.40
  occupancy_occupied_log_odds: 0.85
  occupancy_free_threshold: -0.20
  occupancy_occupied_threshold: 0.40
  occupancy_clamp_min: -2.0
  occupancy_clamp_max: 3.5
  occupancy_max_rays_per_frame: 5000
  occupancy_max_ray_length: 250.0
  occupancy_ray_step_factor: 1.0
  occupancy_visual_max_points: 250000
```

Smaller voxels preserve detail but use more time and memory. Start near the
reliable 3D point spacing, not the RGB ground-sampling distance. The 5,000-ray,
one-voxel-step defaults are intended to keep aerial mapping practical. Increase
ray count or reduce `occupancy_ray_step_factor` only when thin objects are being
missed, and check `timings.jsonl` to measure the cost.

## Important limitation for aerial data

Ray carving says that the volume between an aerial camera and a visible surface
was unobstructed at capture time. Most of that free volume is air. It does **not**
mean a ground robot can drive there. Use the 3D grid for visibility/collision
reasoning, and derive a height-filtered navigation layer for a specific robot
only after choosing its ground surface, height band, clearance, and footprint.

Monocular depth errors can carve through a real obstacle or place a false
surface. For safety-sensitive use, raise the evidence required for a decisive
state, incorporate per-pixel depth confidence, and validate against measured
geometry. The automated tests verify ray traversal, all stored states, color
encoding, checkpoint reload, and optimized-map regeneration; real-scene
thresholds still require your PCD/image data.
