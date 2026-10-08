"""
Generate synthetic example datasets for the tetramer structural comparison app.

Three datasets: X-ray (reference + noise), MD (scaled + bias + noise),
CGDNA (different scale + bias + noise). Fixed seed for reproducibility.
"""

from itertools import product
import numpy as np
import pandas as pd
import os

SEED = 42
N_COORDS = 18
rng = np.random.default_rng(SEED)

# ------------------------------------------------------------------
# All 256 tetramers (A,C,G,T)^4
# ------------------------------------------------------------------
tetramers = ["".join(x) for x in product("ACGT", repeat=4)]
assert len(tetramers) == 256

# ------------------------------------------------------------------
# Coordinate physical ranges (approximate, inspired by real DNA params)
# Intra  (odd columns 1,3,5,7,9,11,13,15,17 → 0-indexed 0,2,4,…)
# Inter  (even columns 2,4,6,8,10,12,14,16,18 → 0-indexed 1,3,5,…)
#
# Typical inter-bp: Shift ~0, Slide ~-0.3..0.3, Rise ~3.3,
#                   Tilt ~0, Roll ~3..6, Twist ~34..36
# Typical intra-bp: Buckle ~0, Propeller ~-15..-5, Opening ~0..2,
#                   Shear ~0, Stretch ~0, Stagger ~0
# We create 3 intra groups × 6 params + 3 inter groups × 6 params = 18
# ------------------------------------------------------------------

# Base reference vectors – smooth variation across tetramers
# Use a combination of sinusoids of tetramer index to create structure
idx = np.arange(256)

coord_ranges = [
    # (center, amplitude, noise_std)  — approximate physical units
    (0.0,   0.8,  0.05),   # Buckle (deg)
    (-0.2,  0.4,  0.03),   # Shift (Å)
    (-10.0, 8.0,  0.5),    # Propeller (deg)
    (-0.3,  0.5,  0.03),   # Slide (Å)
    (1.0,   1.0,  0.08),   # Opening (deg)
    (3.36,  0.05, 0.01),   # Rise (Å)
    (0.0,   0.5,  0.04),   # Shear (Å)
    (2.0,   4.0,  0.3),    # Roll (deg)
    (0.0,   0.4,  0.03),   # Stretch (Å)
    (35.5,  2.5,  0.2),    # Twist (deg)
    (0.0,   0.3,  0.02),   # Stagger (Å)
    (0.0,   3.5,  0.25),   # Tilt (deg)
    (0.0,   0.6,  0.04),   # Buckle2 (deg)
    (-0.1,  0.3,  0.02),   # Shift2 (Å)
    (-8.0,  7.0,  0.4),    # Propeller2 (deg)
    (-0.25, 0.4,  0.03),   # Slide2 (Å)
    (0.8,   0.9,  0.07),   # Opening2 (deg)
    (3.38,  0.04, 0.01),   # Rise2 (Å)
]

coord_names = [
    "Buckle", "Shift", "Propeller", "Slide", "Opening", "Rise",
    "Shear", "Roll", "Stretch", "Twist", "Stagger", "Tilt",
    "Buckle_2", "Shift_2", "Propeller_2", "Slide_2", "Opening_2", "Rise_2",
]
assert len(coord_names) == N_COORDS

def make_reference(seed_offset=0):
    """Create a 256 x 18 reference matrix with sinusoidal structure."""
    ref = np.zeros((256, N_COORDS))
    for c, (center, amp, _) in enumerate(coord_ranges):
        # Deterministic smooth variation: superposition of harmonics
        base = (
            amp * 0.5 * np.sin(2 * np.pi * idx / 64 + c * 0.7 + seed_offset)
            + amp * 0.3 * np.sin(2 * np.pi * idx / 16 + c * 1.3 + seed_offset)
            + amp * 0.2 * np.cos(2 * np.pi * idx / 4  + c * 0.4 + seed_offset)
        )
        ref[:, c] = center + base
    return ref

reference = make_reference()

def add_noise(ref, scale=1.0, bias=None, noise_multiplier=1.0, seed=0):
    """Perturb the reference to simulate a different data source."""
    rng_local = np.random.default_rng(seed)
    out = ref.copy()
    for c, (_, _, noise_std) in enumerate(coord_ranges):
        noise = rng_local.normal(0, noise_std * noise_multiplier, size=256)
        out[:, c] = scale * ref[:, c] + noise
        if bias is not None:
            out[:, c] += bias[c]
    return out

# X-ray: reference + small Gaussian noise
bias_xray = np.zeros(N_COORDS)
xray_data = add_noise(reference, scale=1.0,  bias=bias_xray, noise_multiplier=1.0, seed=1)

# MD: slight scale + small systematic bias per coordinate + moderate noise
bias_md = rng.normal(0, 0.02, size=N_COORDS)
# Scale each coordinate slightly (0.93–0.98 range)
scale_md = rng.uniform(0.93, 1.02, size=N_COORDS)
md_data = np.zeros_like(reference)
rng_md = np.random.default_rng(2)
for c, (_, _, noise_std) in enumerate(coord_ranges):
    noise = rng_md.normal(0, noise_std * 1.5, size=256)
    md_data[:, c] = scale_md[c] * reference[:, c] + bias_md[c] + noise

# CGDNA: different scale + different bias + different noise level
bias_cg = rng.normal(0, 0.03, size=N_COORDS)
scale_cg = rng.uniform(0.96, 1.06, size=N_COORDS)
cg_data = np.zeros_like(reference)
rng_cg = np.random.default_rng(3)
for c, (_, _, noise_std) in enumerate(coord_ranges):
    noise = rng_cg.normal(0, noise_std * 1.2, size=256)
    cg_data[:, c] = scale_cg[c] * reference[:, c] + bias_cg[c] + noise

# Cryo-EM: separate seeded stream (does not disturb the other datasets)
rng_ce = np.random.default_rng(7)
scale_ce = rng_ce.uniform(0.94, 1.04, size=N_COORDS)
bias_ce  = rng_ce.normal(0, 0.025, size=N_COORDS)
ce_data = np.zeros_like(reference)
rng_ce2 = np.random.default_rng(4)
for c, (_, _, noise_std) in enumerate(coord_ranges):
    noise = rng_ce2.normal(0, noise_std * 1.3, size=256)
    ce_data[:, c] = scale_ce[c] * reference[:, c] + bias_ce[c] + noise

# ------------------------------------------------------------------
# Save CSVs
# ------------------------------------------------------------------
# ------------------------------------------------------------------
# Variance data (placeholder): per-coordinate variance for every tetramer.
# Shared sequence-dependent structure + dataset-specific scale and noise.
# ------------------------------------------------------------------
rng_v = np.random.default_rng(11)
phase = rng_v.uniform(0, 2 * np.pi, size=(3, N_COORDS))
var_struct = np.zeros((256, N_COORDS))
for c in range(N_COORDS):
    var_struct[:, c] = (
        0.5 * np.sin(2 * np.pi * idx / 32 + phase[0, c])
        + 0.3 * np.sin(2 * np.pi * idx / 8 + phase[1, c])
        + 0.2 * np.cos(2 * np.pi * idx / 4 + phase[2, c])
    )
sd_c = np.array([ns for (_, _, ns) in coord_ranges]) * 3.0      # typical std per coordinate
var_reference = (sd_c ** 2)[None, :] * np.exp(0.5 * var_struct)  # strictly positive

def make_variance(scale, noise, seed):
    r = np.random.default_rng(seed)
    return scale * var_reference * np.exp(r.normal(0, noise, size=var_reference.shape))

var_xray   = make_variance(1.00, 0.10, 21)
var_cryoem = make_variance(1.06, 0.14, 22)
var_md     = make_variance(0.90, 0.12, 23)
var_cgna   = make_variance(1.12, 0.10, 24)

# Cryo-EM + X-ray (placeholder): average of the two shape sets, pooled (reduced) variance.
# Own RNG so the other datasets are unchanged.  Replace data/cryoem_xray.csv with real data.
_rng_cx = np.random.default_rng(31)
cx_data = 0.5 * (xray_data + ce_data) + _rng_cx.normal(0, 0.01, size=xray_data.shape) * sd_c[None, :]
var_cx  = 0.5 * (var_xray + var_cryoem) * 0.75


def _block(values, info):
    """Tetramer rows followed by dimer rows (mean over the 16 tetramers sharing the
    central dinucleotide) for one Info type."""
    tet = pd.DataFrame(values, columns=coord_names)
    tet.insert(0, "Tetramer", tetramers)
    centre = tet["Tetramer"].str[1:3]
    dim = tet.groupby(centre)[coord_names].mean().reset_index()
    dim.columns = ["Tetramer"] + coord_names
    out = pd.concat([tet, dim], ignore_index=True)
    out.insert(1, "Info", info)
    return out


def save_csv(shape, variance, filename):
    """Columns: Tetramer, Info, <18 coordinates>.
    Row order: Shape tetramers, Variance tetramers, Shape dimers, Variance dimers."""
    s, v = _block(shape, "Shape"), _block(variance, "Variance")
    is_t = lambda d: d["Tetramer"].str.len() == 4
    df = pd.concat([s[is_t(s)], v[is_t(v)], s[~is_t(s)], v[~is_t(v)]], ignore_index=True)
    df.to_csv(filename, index=False, float_format="%.8f")
    print(f"Saved {filename}  shape={df.shape}")

out_dir = os.path.join(os.path.dirname(__file__), "data")
os.makedirs(out_dir, exist_ok=True)

save_csv(xray_data, var_xray,   os.path.join(out_dir, "xray.csv"))
save_csv(md_data,   var_md,     os.path.join(out_dir, "md.csv"))
save_csv(ce_data,   var_cryoem, os.path.join(out_dir, "cryoem.csv"))
save_csv(cg_data,   var_cgna,   os.path.join(out_dir, "cgnaplus.csv"))
save_csv(cx_data,   var_cx,     os.path.join(out_dir, "cryoem_xray.csv"))

# Quick sanity check
from scipy.stats import pearsonr
for c_idx, name in enumerate(coord_names):
    r, _ = pearsonr(xray_data[:, c_idx], md_data[:, c_idx])
    print(f"  Xray vs MD  {name:20s}  r={r:.4f}")
print("Example datasets generated successfully.")
