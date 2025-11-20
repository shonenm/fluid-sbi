# IBPM Quick Reference Card

## Resolution Selection

| Resolution | Command | Grid | Cylinder | Use Case |
|------------|---------|------|----------|----------|
| **Minimum** | `-nx 64 -ny 64` | 63×63 | 16 cells | Quick tests |
| **Recommended** | `-nx 128 -ny 128` | 127×127 | 32 cells | Development ⭐ |
| **High Quality** | `-nx 200 -ny 200` | 199×199 | 50 cells | Production |
| **Ultra** | `-nx 256 -ny 256` | 255×255 | 64 cells | Publication |

---

## Basic Commands

### 1. Generate IBPM Data (128×128 example)

```bash
ibpm -nx 128 -ny 128 \
     -length 4 -xoffset -2 -yoffset -2 \
     -geom cylinder.geom \
     -Re 100 -dt 0.02 -nsteps 250 \
     -tecplot 1 \
     -outdir /workspace/data/ibpm_128
```

### 2. Convert to HDF5 (No Resize)

```bash
python /workspace/scripts/convert_ibpm_to_sda.py \
    --input /workspace/data/ibpm_128 \
    --output /workspace/data/ibpm_h5_128 \
    --window 64 \
    --stride 8
```

### 3. Train Model

```bash
cd /workspace/sda/experiments/ibpm
python train.py
```

### 4. Evaluate Reconstruction

```bash
# Coarse observation experiment
python eval_coarse.py

# Sparse observation experiment
python eval_sparse.py
```

---

## Key Parameters

### IBPM Simulation

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `-nx`, `-ny` | 128 | Grid size (127 points) |
| `-length` | 4 | Domain size (4×4) |
| `-xoffset`, `-yoffset` | -2 | Domain origin [-2,2]×[-2,2] |
| `-Re` | 100 | Reynolds number |
| `-dt` | 0.02 | Time step |
| `-nsteps` | 250 | Number of steps |
| `-tecplot` | 1 | Output frequency |

### Data Conversion

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `--coarsen` | 1 | Coarsening factor (1 = no compression) |
| `--window` | 64 | Time window size |
| `--stride` | 8 | Sliding window stride |
| `--train-ratio` | 0.7 | Training split |
| `--valid-ratio` | 0.15 | Validation split |

### Training Configuration

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `window` | 3 | Time context (must ≤ min timesteps) |
| `batch_size` | 32 | Training batch size |
| `learning_rate` | 2e-4 | AdamW learning rate |
| `epochs` | 1024 | Total training epochs |

---

## Resolution Formulas

### Grid Points from Cell Count
```
grid_points = nx - 1
```
Example: `-nx 128` → 127×127 grid

### Cylinder Diameter in Pixels
```
dx = domain_size / (nx - 1)
cylinder_pixels = cylinder_diameter / dx
```
Example: nx=128, domain=4, D=1 → D ≈ 31.75 pixels

### CFL Condition
```
CFL = U∞ × dt / dx ≤ 1.0
```
Example: U∞=1.0, dt=0.02, dx=0.0315 → CFL=0.635 ✓

---

## File Structure

```
/workspace/
├── data/
│   ├── ibpm_128/              # IBPM output (Tecplot)
│   │   ├── ibpm00000.plt
│   │   ├── ibpm00001.plt
│   │   └── ...
│   └── ibpm_h5_128/           # HDF5 converted
│       ├── train.h5
│       ├── valid.h5
│       └── test.h5
├── sda/experiments/ibpm/
│   ├── train.py               # Training script
│   ├── eval_utils.py          # Common utilities
│   ├── eval_coarse.py         # Coarse observation eval
│   └── eval_sparse.py         # Sparse observation eval
└── scripts/
    └── convert_ibpm_to_sda.py # Conversion script
```

---

## Data Shapes

### IBPM Tecplot Output
- Format: ASCII, structured grid
- Variables: x, y, u, v, vorticity
- Grid: (I, J) structured points

### Intermediate Timeseries
- Shape: `(T, 2, H, W)`
- T: Number of timesteps (251)
- 2: Channels (u, v)
- H, W: Spatial resolution (nx-1, ny-1)

### HDF5 Storage Format
- Shape: `(T, N, C, H, W)`
- T: Number of time windows
- N: Samples per window (64)
- C: Channels (2)
- H, W: Spatial resolution

### Training Data Format
- Shape: `(N, T, C, H, W)` after transposition
- Created by `IBPMDataset` class
- Flattened: `(N, T*C, H, W)` for convolution

---

## Cylinder Constraint

### Mask Creation (64×64 example)
```python
center = (32.0, 37.0)  # Slightly off-center
radius = 7.5           # ~15 pixel diameter
mask = distance > radius  # True outside cylinder
```

### Applying Constraint
```python
velocity = velocity * mask  # Zero inside cylinder
```

### Resolution Scaling
For different resolutions:
```python
scale_factor = new_resolution / 64
center_x = 32.0 * scale_factor
center_y = 37.0 * scale_factor
radius = 7.5 * scale_factor
```

---

## Common Workflows

### 1. Quick Test (5 minutes)
```bash
# Low resolution, short simulation
ibpm -nx 64 -ny 64 -nsteps 50 -outdir data/test_64
python convert_ibpm_to_sda.py --input data/test_64 --output data/test_h5
# Verify shape
python -c "import h5py; print(h5py.File('data/test_h5/train.h5')['x'].shape)"
```

### 2. Development Workflow (20 minutes)
```bash
# Recommended resolution
ibpm -nx 128 -ny 128 -nsteps 250 -tecplot 1 -outdir data/ibpm_128
python convert_ibpm_to_sda.py \
    --input data/ibpm_128 \
    --output data/ibpm_h5_128 \
    --window 64 --stride 8
cd sda/experiments/ibpm
python train.py  # Check it runs
```

### 3. Production Run (1+ hours)
```bash
# High resolution, more timesteps
ibpm -nx 200 -ny 200 -nsteps 500 -tecplot 2 -outdir data/ibpm_prod
python convert_ibpm_to_sda.py \
    --input data/ibpm_prod \
    --output data/ibpm_h5_prod \
    --window 64 --stride 8
cd sda/experiments/ibpm
sbatch train_job.sh  # Submit to cluster
```

---

## Troubleshooting

### Data Shape Mismatch
**Symptom**: "Train and valid shapes don't match"
**Fix**: Use `IBPMDataset` class (handles transposition)

### Window Too Large
**Symptom**: "Time series length < window size"
**Fix**: Reduce `window` parameter in train.py

### Cylinder Blocky
**Symptom**: Cylinder appears pixelated
**Fix**: Increase IBPM resolution (`-nx 256`)

### Out of Memory
**Symptom**: CUDA out of memory during training
**Fix**: Reduce `batch_size` or use lower resolution

### CFL Violation
**Symptom**: Simulation diverges
**Fix**: Reduce `dt` (e.g., 0.01 for high resolution)

---

## Verification Commands

```bash
# Check IBPM output
ls -lh data/ibpm_128/*.plt | wc -l  # Should be 251

# Check HDF5 shapes
python3 << EOF
import h5py
for split in ['train', 'valid', 'test']:
    with h5py.File(f'data/ibpm_h5_128/{split}.h5', 'r') as f:
        print(f"{split}: {f['x'].shape}")
EOF

# Check cylinder mask
python3 << EOF
import sys; sys.path.append('sda/experiments/ibpm')
from eval_utils import create_cylinder_mask
mask = create_cylinder_mask(127)
print(f"Cylinder points: {(~mask).sum()} / {127*127}")
print(f"Cylinder fraction: {(~mask).sum() / (127*127):.2%}")
EOF

# Visualize data
python scripts/plot_ibpm_raw.py \
    --input data/ibpm_128/ibpm00100.plt \
    --output viz/snapshot_t100.png
```

---

## Performance Estimates

| Resolution | IBPM Time | Conversion | Training/Epoch | Total |
|------------|-----------|------------|----------------|-------|
| 64×64      | 5 min     | 1 min      | 10 sec         | ~20 min |
| 128×128    | 15 min    | 2 min      | 30 sec         | ~1 hour |
| 199×199    | 40 min    | 5 min      | 2 min          | ~3 hours |
| 256×256    | 90 min    | 10 min     | 5 min          | ~8 hours |

*Estimates for 250 timesteps, 1024 epochs on single GPU

---

## Key Design Decisions

1. **No Resize**: Preserve physical features by generating at target resolution
2. **Non-Periodic Boundaries**: Use `replicate` padding, not `circular`
3. **Cylinder Constraint**: Hard constraint (multiply by mask) in evaluation
4. **Data Transposition**: `IBPMDataset` handles (T,N,C,H,W) → (N,T,C,H,W)
5. **Time Window**: Limited by validation data (minimum 3 timesteps)

---

**Quick Links**:
- Full guide: `ibpm_resolution_guide.md`
- Workflow examples: `ibpm_resolution_workflow_example.md`
- Verification report: `ibpm_data_generation_verification.md`
- Strategy document: `sda/experiments/ibpm/reconstruction_experiment_strategy.md`

**Last Updated**: 2024-10-29
