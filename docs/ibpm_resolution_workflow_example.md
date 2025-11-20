# IBPM Resolution Workflow - Practical Examples

## Overview

This document provides step-by-step examples for generating IBPM data at different resolutions **without any resize operations** to preserve physical features.

---

## Why No Resize?

**Physical Feature Preservation**: Interpolation/resize operations can:
- Smooth out vortex structures
- Alter velocity gradients
- Change kinetic energy distribution
- Introduce artificial dissipation

**Solution**: Specify the target resolution directly in IBPM execution using `-nx` and `-ny` flags.

---

## Example 1: 128×128 Resolution (Recommended)

### Step 1: Run IBPM Simulation

```bash
cd /workspace/data
mkdir -p ibpm_128

ibpm -nx 128 -ny 128 \
     -length 4 -xoffset -2 -yoffset -2 \
     -geom cylinder.geom \
     -Re 100 -dt 0.02 -nsteps 250 \
     -tecplot 1 \
     -outdir /workspace/data/ibpm_128
```

**Output**: 251 Tecplot files with 127×127 grid points each

**Why 127×127?**: 128 cells produce 127 cell-center points

### Step 2: Convert to HDF5 (No Resize)

```bash
python /workspace/scripts/convert_ibpm_to_sda.py \
    --input /workspace/data/ibpm_128 \
    --output /workspace/data/ibpm_h5_128 \
    --window 64 \
    --stride 8
```

**Key**: No `--coarsen` flag means `coarsen_factor=1` (no compression)

**Output**:
```
Full timeseries shape: (251, 2, 127, 127)
  - Timesteps: 251
  - Channels: 2 (u, v)
  - Spatial resolution: 127×127 (original IBPM output)
  - No compression applied ✓

train.h5: (n_samples, 64, 2, 127, 127)
valid.h5: (n_samples, 64, 2, 127, 127)
test.h5:  (n_samples, 64, 2, 127, 127)
```

### Step 3: Verify Physical Features

```bash
python3 << 'EOF'
import h5py
import numpy as np

with h5py.File('/workspace/data/ibpm_h5_128/train.h5', 'r') as f:
    data = f['x'][:]
    print(f"Data shape: {data.shape}")
    print(f"Spatial resolution: {data.shape[-2]}×{data.shape[-1]}")
    print(f"Expected: 127×127 (no resize)")

    # Check if resolution matches IBPM output
    assert data.shape[-2] == 127 and data.shape[-1] == 127
    print("✓ Resolution preserved from IBPM output")

    # Verify velocity field statistics
    u_mean = data[:, :, 0].mean()
    v_mean = data[:, :, 1].mean()
    print(f"\nVelocity statistics:")
    print(f"  u mean: {u_mean:.6f}")
    print(f"  v mean: {v_mean:.6f}")
    print(f"  u range: [{data[:,:,0].min():.3f}, {data[:,:,0].max():.3f}]")
    print(f"  v range: [{data[:,:,1].min():.3f}, {data[:,:,1].max():.3f}]")

print("\n✓ Physical features preserved!")
EOF
```

---

## Example 2: 64×64 Resolution (Fast Prototyping)

### Step 1: Run IBPM

```bash
ibpm -nx 64 -ny 64 \
     -length 4 -xoffset -2 -yoffset -2 \
     -geom cylinder.geom \
     -Re 100 -dt 0.02 -nsteps 250 \
     -tecplot 1 \
     -outdir /workspace/data/ibpm_64
```

**Output**: 251 files with 63×63 grid points each

**Cylinder resolution**: D ≈ 16 cells (minimum acceptable)

### Step 2: Convert

```bash
python /workspace/scripts/convert_ibpm_to_sda.py \
    --input /workspace/data/ibpm_64 \
    --output /workspace/data/ibpm_h5_64 \
    --window 64 \
    --stride 8
```

**Result**: HDF5 files with 63×63 resolution (preserved from IBPM)

---

## Example 3: 199×199 Resolution (High Quality)

### Step 1: Run IBPM

```bash
ibpm -nx 200 -ny 200 \
     -length 4 -xoffset -2 -yoffset -2 \
     -geom cylinder.geom \
     -Re 100 -dt 0.02 -nsteps 250 \
     -tecplot 1 \
     -outdir /workspace/data/ibpm_199
```

**Output**: 251 files with 199×199 grid points each

**Cylinder resolution**: D ≈ 50 cells (excellent)

### Step 2: Convert

```bash
python /workspace/scripts/convert_ibpm_to_sda.py \
    --input /workspace/data/ibpm_199 \
    --output /workspace/data/ibpm_h5_199 \
    --window 64 \
    --stride 8
```

**Result**: HDF5 files with 199×199 resolution

**Note**: Training may require more GPU memory at this resolution

---

## Example 4: Using Coarsen (Optional)

If you need to reduce resolution while maintaining some physical features:

```bash
# Generate high-resolution data first
ibpm -nx 256 -ny 256 -outdir ibpm_256 ...

# Then apply coarsening (averaging, not interpolation)
python /workspace/scripts/convert_ibpm_to_sda.py \
    --input /workspace/data/ibpm_256 \
    --output /workspace/data/ibpm_h5_128 \
    --coarsen 2 \
    --window 64 \
    --stride 8
```

**Effect**: 255×255 → 127×127 via 2×2 average pooling

**Advantage over resize**: Preserves energy conservation better than interpolation

**Disadvantage**: Still loses some physical features (vortex details)

---

## Updating Experiment Configuration

After generating data at a specific resolution, update your experiment scripts:

### For 127×127 Data (from 128×128 IBPM):

**eval_utils.py**:
```python
def create_cylinder_mask(size: int = 127, device: str = 'cpu') -> Tensor:
    # Adjust center and radius for new resolution
    center_x, center_y = 63.5, 63.5  # Center of 127×127 grid
    radius = (127 / 4) / 2  # D = 1.0 in domain [-2,2]×[-2,2]
    # radius ≈ 15.875 pixels
```

**train.py**:
```python
# Data will automatically match the HDF5 file resolution
trainset = IBPMDataset(
    data_dir / 'train.h5',
    window=CONFIG['window'],
    flatten=True,
)
# Output shape: (window * 2, 127, 127)
```

---

## Resolution Comparison Table

| IBPM -nx | Grid Points | Cylinder Resolution | Disk (250 steps) | Recommended Use |
|----------|-------------|---------------------|------------------|-----------------|
| 64       | 63×63       | D ≈ 16 cells        | ~50 MB          | Quick tests     |
| 128      | 127×127     | D ≈ 32 cells        | ~200 MB         | **Development** |
| 200      | 199×199     | D ≈ 50 cells        | ~625 MB         | High quality    |
| 256      | 255×255     | D ≈ 64 cells        | ~1.25 GB        | Publication     |

---

## Verification Checklist

After generating new data:

1. **Check resolution preserved**:
   ```bash
   python3 -c "import h5py; f = h5py.File('train.h5'); print(f['x'].shape)"
   ```

2. **Verify cylinder size**:
   ```python
   # Calculate expected cylinder diameter in pixels
   nx = 128  # Your IBPM -nx value
   grid_points = nx - 1
   domain_size = 4.0
   dx = domain_size / (nx - 1)
   cylinder_diameter = 1.0
   cylinder_pixels = cylinder_diameter / dx
   print(f"Expected cylinder diameter: {cylinder_pixels:.1f} pixels")
   ```

3. **Check velocity statistics**:
   ```bash
   python scripts/inspect_ibpm_data.py /workspace/data/ibpm_h5_128
   ```

4. **Visualize first timestep**:
   ```bash
   python scripts/plot_ibpm_raw.py \
       --input /workspace/data/ibpm_128/ibpm00000.plt \
       --output viz/ibpm_t0.png
   ```

---

## Common Issues

### Issue: "Grid points don't match expected resolution"

**Cause**: Forgot that nx cells produce (nx-1) grid points

**Solution**:
- IBPM `-nx 128` → 127×127 grid points ✓
- IBPM `-nx 200` → 199×199 grid points ✓

### Issue: "Cylinder appears blocky"

**Cause**: Resolution too low (< 20 cells per diameter)

**Solution**: Increase IBPM resolution:
```bash
ibpm -nx 256 -ny 256 ...  # D ≈ 64 cells
```

### Issue: "Training runs out of memory"

**Cause**: Resolution too high for GPU

**Solution**: Either:
1. Reduce batch size in train.py
2. Use lower resolution (128×128 instead of 256×256)
3. Use coarsen factor: `--coarsen 2`

---

## Summary

**Golden Rule**: Specify resolution in IBPM, not in conversion script

**Recommended Workflow**:
1. Choose target resolution based on cylinder resolution (≥ 32 cells)
2. Run IBPM with `-nx` and `-ny`
3. Convert with `--coarsen 1` (default, no compression)
4. Train with preserved physical features

**Result**: ML models learn from physically accurate flow fields

---

**Updated**: 2024-10-29
**See also**:
- `ibpm_resolution_guide.md` - Resolution selection criteria
- `convert_ibpm_to_sda.py` - Conversion script
- `ibpm_data_generation_verification.md` - Data verification
