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
VTK_DIR = "VTK"
OUT_DIR  = "results"
os.makedirs(OUT_DIR, exist_ok=True)

# Physical parameters (Koshizuka & Oka 1996)
a = 0.146        # dam width / column width [m]
g = 9.81         # gravity [m/s²]

# Experimental reference data (Koshizuka & Oka 1996, Table 1)
# Non-dimensional: t* = t*sqrt(2g/a), x_f/a, h/a
ref_wavefront = np.array([
    [0.0,  1.0],
    [0.5,  1.20],
    [1.0,  1.56],
    [1.5,  2.00],
    [2.0,  2.45],
    [2.5,  2.84],
    [3.0,  3.22],
    [3.5,  3.61],
    [4.0,  3.91],
    [4.5,  4.23],
    [5.0,  4.52],
    [5.5,  4.79],
    [6.0,  5.10],
    [6.5,  5.37],
    [7.0,  5.61],
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

# ── Load VTK files ────────────────────────────────────────────────────────────
vtk_files = sorted(glob.glob(f"{VTK_DIR}/openfoam-dam-break_*.vtk"))
vtk_files = [f for f in vtk_files if not f.endswith("_0.vtk")]
if not vtk_files:
    print("No VTK files found — run foamToVTK first"); sys.exit(1)
print(f"Found {len(vtk_files)} VTK files")

# Extract time from filename: openfoam-dam-break_N.vtk, map N → sim time
# We need actual time values — read from time directories
time_dirs = sorted(
    [d for d in os.listdir(".") if d.replace(".", "").isdigit() and d != "0"],
    key=float
)
times = [float(d) for d in time_dirs]

# Match VTK files to times (both should be same count)
if len(vtk_files) != len(times):
    print(f"WARNING: {len(vtk_files)} VTK files vs {len(times)} time dirs — using min")
    n = min(len(vtk_files), len(times))
    vtk_files = vtk_files[:n]
    times = times[:n]

# ── Time series: wave front and column height ─────────────────────────────────
print("Computing wave front and column height time series...")
sim_tstar = []
sim_wavefront = []
sim_colheight = []

for vtk_path, t in zip(vtk_files, times):
    mesh = pv.read(vtk_path)
    zmid = (mesh.bounds[4] + mesh.bounds[5]) / 2
    sl = mesh.slice(normal="z", origin=(0, 0, zmid))
    if sl.n_points == 0:
        continue

    sl_cc = sl.cell_centers()
    pts   = np.array(sl_cc.points)
    alpha = np.array(sl["alpha.water"]).ravel()

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

# ── Validation plot ───────────────────────────────────────────────────────────
print("Plotting validation...")
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(f"Dam Break Validation — {SUFFIX} mesh (Koshizuka & Oka 1996)", fontsize=13)

ax = axes[0]
ax.plot(ref_wavefront[:, 0], ref_wavefront[:, 1], "ko", ms=5, label="Experiment")
ax.plot(sim_tstar, sim_wavefront, "r-", lw=2, label=f"OpenFOAM ({SUFFIX})")
ax.set_xlabel("t* = t√(2g/a)")
ax.set_ylabel("x_f / a")
ax.set_title("Wave Front Position")
ax.legend(); ax.grid(True, alpha=0.3)

ax = axes[1]
ax.plot(ref_colheight[:, 0], ref_colheight[:, 1], "ko", ms=5, label="Experiment")
ax.plot(sim_tstar, sim_colheight, "b-", lw=2, label=f"OpenFOAM ({SUFFIX})")
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
for vtk_path in vtk_files:
    mesh = pv.read(vtk_path)
    zmid = (mesh.bounds[4] + mesh.bounds[5]) / 2
    sl = mesh.slice(normal="z", origin=(0, 0, zmid))
    if sl.n_points == 0:
        vols.append(np.nan); continue
    sl = sl.compute_cell_sizes(area=True, length=False, volume=False)
    alpha = np.array(sl["alpha.water"]).ravel()
    area  = np.array(sl["Area"]).ravel()
    vols.append(np.sum(alpha * area))

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
snap_times = [0.25, 0.5, 1.0, 2.0]
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Dam Break — alpha.water snapshots ({SUFFIX})", fontsize=13)

for ax, t_target in zip(axes.ravel(), snap_times):
    idx = np.argmin(np.abs(np.array(times) - t_target))
    mesh = pv.read(vtk_files[idx])
    zmid = (mesh.bounds[4] + mesh.bounds[5]) / 2
    sl = mesh.slice(normal="z", origin=(0, 0, zmid))

    sl_cc = sl.cell_centers()
    pts   = np.array(sl_cc.points)
    alpha = np.array(sl["alpha.water"]).ravel()

    # Interpolate onto regular grid
    xmin, xmax = mesh.bounds[0], mesh.bounds[1]
    ymin, ymax = mesh.bounds[2], mesh.bounds[3]
    xi = np.linspace(xmin, xmax, 400)
    yi = np.linspace(ymin, ymax, 400)
    Xi, Yi = np.meshgrid(xi, yi)
    interp = LinearNDInterpolator(pts[:, :2], alpha, fill_value=0.0)
    Ai = interp(Xi, Yi)

    cf = ax.contourf(Xi, Yi, Ai, levels=50, cmap="Blues", vmin=0, vmax=1)
    ax.set_title(f"t = {times[idx]:.2f}s (t*={times[idx]*np.sqrt(2*g/a):.1f})")
    ax.set_xlabel("x [m]"); ax.set_ylabel("y [m]")
    ax.set_aspect("equal")
    plt.colorbar(cf, ax=ax, label="α water")

plt.tight_layout()
plt.savefig(f"{OUT_DIR}/snapshots_{SUFFIX}.png", dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved {OUT_DIR}/snapshots_{SUFFIX}.png")

# ── Animation ─────────────────────────────────────────────────────────────────
print("Generating animation (this may take a minute)...")
fig2, ax2 = plt.subplots(figsize=(8, 8))

mesh0 = pv.read(vtk_files[0])
xmin, xmax = mesh0.bounds[0], mesh0.bounds[1]
ymin, ymax = mesh0.bounds[2], mesh0.bounds[3]
xi = np.linspace(xmin, xmax, 400)
yi = np.linspace(ymin, ymax, 400)
Xi, Yi = np.meshgrid(xi, yi)

frames = []
for vtk_path, t in zip(vtk_files[::2], times[::2]):   # every 2nd frame for speed
    mesh = pv.read(vtk_path)
    zmid = (mesh.bounds[4] + mesh.bounds[5]) / 2
    sl = mesh.slice(normal="z", origin=(0, 0, zmid))
    if sl.n_points == 0:
        continue
    sl_cc = sl.cell_centers()
    pts   = np.array(sl_cc.points)
    alpha = np.array(sl["alpha.water"]).ravel()
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
    frame = np.asarray(buf)[..., :3]
    frames.append(frame)

plt.close(fig2)

# Write mp4
import imageio
out_mp4 = f"{OUT_DIR}/dam_break_{SUFFIX}.mp4"
with imageio.get_writer(out_mp4, fps=15, codec="libx264",
                        output_params=["-pix_fmt", "yuv420p"]) as writer:
    for frame in frames:
        writer.append_data(frame)
print(f"  Saved {out_mp4}")

print(f"\nDone. All outputs in {OUT_DIR}/")
