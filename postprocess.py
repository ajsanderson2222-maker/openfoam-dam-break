#!/usr/bin/env python3
"""
Dam break post-processing: validation against Koshizuka & Oka (1996),
snapshots, and animation. Usage: python3 postprocess.py [coarse|fine|adaptive]
"""
import sys, os, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.interpolate import LinearNDInterpolator
import pyvista as pv

SUFFIX = sys.argv[1] if len(sys.argv) > 1 else "coarse"
OUT_DIR = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# Physical parameters (Koshizuka & Oka 1996)
a = 0.146        # dam width / column width [m]
g = 9.81         # gravity [m/s²]

# Reference data: Koshizuka & Oka (1996) MPS simulation, open-channel tank
# (8.2a × 4.1a, no obstacle).  t* = t·√(2g/a),  x_f/a,  h/a
# NOTE: many online sources mis-cite data from the 4a×4a step-baffle tutorial
# geometry (step at x=2a).  These values are for the correct open-channel case.
ref_wavefront = np.array([
    [0.0,  1.00],
    [0.5,  1.30],
    [1.0,  1.80],
    [1.5,  2.32],
    [2.0,  2.84],
    [2.5,  3.31],
    [3.0,  3.74],
    [3.5,  4.13],
    [4.0,  4.49],
    [4.5,  4.82],
    [5.0,  5.10],
    [5.5,  5.35],
    [6.0,  5.56],
    [6.5,  5.74],
    [7.0,  5.89],
])
ref_colheight = np.array([
    [0.0,  2.0],
    [0.5,  1.95],
    [1.0,  1.83],
    [1.5,  1.68],
    [2.0,  1.48],
    [2.5,  1.25],
    [3.0,  1.00],
    [3.5,  0.80],
    [4.0,  0.68],
    [4.5,  0.59],
    [5.0,  0.52],
    [5.5,  0.46],
    [6.0,  0.42],
])

# ── Load case via native OpenFOAM reader ─────────────────────────────────────
foam_file = "case.foam"
if not os.path.exists(foam_file):
    open(foam_file, "w").close()

reader = pv.OpenFOAMReader(foam_file)
times = np.array(reader.time_values)
times = times[times > 0]   # skip t=0
print(f"Found {len(times)} time steps: {times[0]:.3f} s to {times[-1]:.3f} s")

def get_slice(t):
    """Return (pts_cc, alpha) for a z-midplane slice at time t."""
    reader.set_active_time_value(t)
    mesh = reader.read()
    # The internal mesh is in mesh["internalMesh"]
    block = mesh["internalMesh"]
    zmid = (block.bounds[4] + block.bounds[5]) / 2
    sl = block.slice(normal="z", origin=(0, 0, zmid))
    sl_cc = sl.cell_centers()
    pts   = np.array(sl_cc.points)
    alpha = np.array(sl["alpha.water"]).ravel()
    return pts, alpha

# ── Time series: wave front and column height ─────────────────────────────────
print("Computing wave front and column height time series...")
sim_tstar      = []
sim_wavefront  = []
sim_colheight  = []

for t in times:
    pts, alpha = get_slice(t)
    wet = alpha > 0.5
    if wet.sum() == 0:
        continue
    front_x    = pts[wet, 0].max()
    left_wet   = wet & (pts[:, 0] < a)
    col_height = pts[left_wet, 1].max() if left_wet.sum() > 0 else 0.0

    t_star = t * np.sqrt(2 * g / a)
    sim_tstar.append(t_star)
    sim_wavefront.append(front_x / a)
    sim_colheight.append(col_height / a)

sim_tstar     = np.array(sim_tstar)
sim_wavefront = np.array(sim_wavefront)
sim_colheight = np.array(sim_colheight)

# Clip wave front at right-wall impact (xf/a within one cell of 6.849a)
wall_hit = np.where(sim_wavefront > 6.70)[0]
tstar_wall = sim_tstar[wall_hit[0]] if len(wall_hit) > 0 else sim_tstar[-1]
wf_mask = sim_tstar < tstar_wall
col_mask = sim_tstar <= ref_colheight[-1, 0] + 0.5

# ── Validation plot ───────────────────────────────────────────────────────────
print("Plotting validation...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Dam Break Validation — {SUFFIX} mesh (Koshizuka & Oka 1996)", fontsize=13)

ax = axes[0]
ax.plot(ref_wavefront[:, 0], ref_wavefront[:, 1], "ko", ms=5, label="K&O (1996) MPS")
ax.plot(sim_tstar[wf_mask], sim_wavefront[wf_mask], "r-", lw=2, label=f"OpenFOAM ({SUFFIX})")
ax.set_xlabel("t* = t√(2g/a)")
ax.set_ylabel("x_f / a")
ax.set_title("Wave Front Position")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(ref_colheight[:, 0], ref_colheight[:, 1], "ko", ms=5, label="K&O (1996) MPS")
ax.plot(sim_tstar[col_mask], sim_colheight[col_mask], "b-", lw=2, label=f"OpenFOAM ({SUFFIX})")
ax.set_xlabel("t* = t√(2g/a)")
ax.set_ylabel("h / a")
ax.set_title("Water Column Height")
ax.legend(); ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/validation_{SUFFIX}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR}/validation_{SUFFIX}.png")

# ── Volume conservation ───────────────────────────────────────────────────────
print("Computing volume conservation...")
vols = []
for t in times:
    pts, alpha = get_slice(t)
    reader.set_active_time_value(t)
    mesh = reader.read()
    block = mesh["internalMesh"]
    zmid = (block.bounds[4] + block.bounds[5]) / 2
    sl = block.slice(normal="z", origin=(0, 0, zmid))
    sl = sl.compute_cell_sizes(area=True, length=False, volume=False)
    alpha_v = np.array(sl["alpha.water"]).ravel()
    area_v  = np.array(sl["Area"]).ravel()
    vols.append(np.sum(alpha_v * area_v))

vols = np.array(vols)
v0 = vols[0]
vol_err = 100 * (vols - v0) / v0

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(times, vol_err, "g-", lw=2)
ax.axhline(0, color="k", lw=0.8, ls="--")
ax.set_xlabel("Time [s]")
ax.set_ylabel("Volume error [%]")
ax.set_title(f"Water Volume Conservation — {SUFFIX}")
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/volume_{SUFFIX}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR}/volume_{SUFFIX}.png")

# ── Snapshots at four times ───────────────────────────────────────────────────
print("Generating snapshots...")
snap_targets = [0.25, 0.5, 1.0, 2.0]
snap_times   = [times[np.argmin(np.abs(times - t))] for t in snap_targets]

fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Dam Break — alpha.water snapshots ({SUFFIX})", fontsize=13)

reader.set_active_time_value(snap_times[0])
mesh0 = reader.read()["internalMesh"]
xmin, xmax = mesh0.bounds[0], mesh0.bounds[1]
ymin, ymax = mesh0.bounds[2], mesh0.bounds[3]
xi = np.linspace(xmin, xmax, 400)
yi = np.linspace(ymin, ymax, 400)
Xi, Yi = np.meshgrid(xi, yi)

for ax, t_snap in zip(axes.ravel(), snap_times):
    pts, alpha = get_slice(t_snap)
    interp = LinearNDInterpolator(pts[:, :2], alpha, fill_value=0.0)
    Ai = interp(Xi, Yi)
    cf = ax.contourf(Xi, Yi, Ai, levels=50, cmap="Blues", vmin=0, vmax=1)
    t_star = t_snap * np.sqrt(2 * g / a)
    ax.set_title(f"t = {t_snap:.2f}s  (t*={t_star:.1f})")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="α water")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/snapshots_{SUFFIX}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR}/snapshots_{SUFFIX}.png")

# ── Animation ─────────────────────────────────────────────────────────────────
print("Generating animation...")
fig2, ax2 = plt.subplots(figsize=(8, 8))
frames = []
for t in times[::2]:   # every other frame
    pts, alpha = get_slice(t)
    interp = LinearNDInterpolator(pts[:, :2], alpha, fill_value=0.0)
    Ai = interp(Xi, Yi)
    ax2.clear()
    ax2.contourf(Xi, Yi, Ai, levels=50, cmap="Blues", vmin=0, vmax=1)
    ax2.set_xlim(xmin, xmax); ax2.set_ylim(ymin, ymax)
    ax2.set_aspect("equal")
    ax2.set_title(f"Dam Break ({SUFFIX})  t = {t:.3f}s", fontsize=12)
    ax2.set_xlabel("x [m]"); ax2.set_ylabel("y [m]")
    fig2.canvas.draw()
    buf = fig2.canvas.buffer_rgba()
    frames.append(np.asarray(buf)[..., :3])
plt.close(fig2)

import imageio
out_mp4 = f"{OUT_DIR}/dam_break_{SUFFIX}.mp4"
with imageio.get_writer(out_mp4, fps=15, codec="libx264",
                        output_params=["-pix_fmt", "yuv420p"]) as writer:
    for frame in frames:
        writer.append_data(frame)
print(f"  Saved {out_mp4}")
print(f"\nDone. All outputs in {OUT_DIR}/")
