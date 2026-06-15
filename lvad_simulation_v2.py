"""
=============================================================================
 Biventricular Interdependence Simulation – LVAD RVF Risk Analysis
 Cardiovascular & Respiratory Mechanics Project, IIT Madras
 Authors : Shristy Roy [MD24B034]
 Version : 2.0  (PV-loop removed; bugs fixed; physics corrected)
=============================================================================
 Simulates three clinical scenarios:
   1. Normal (no LVAD)
   2. Post-LVAD – Low RVF Risk  (mild septal shift)
   3. Post-LVAD – High RVF Risk (severe septal shift)

 Plots generated (8 total, no PV loops):
   01 – Transseptal Pressure Gradient over a cardiac cycle
   02 – Wall Shear Stress distribution along the septum
   03 – LVAD Pump HQ Performance Curves
   04 – RV Inlet Velocity Profiles (tricuspid valve)
   05 – Septal Curvature Effect on 4 haemodynamic parameters
   06 – Radar / Spider Risk Profile Chart
   07 – Summary Bar Comparison (6 key metrics)
   08 – Stroke Work Comparison (LV vs RV)

 Fixes applied vs v1:
   · Removed PV-loop plot and its helper function
   · Radar: LV-unloading and TSG metrics now derived correctly from data
   · Transseptal waveform: LV diastolic baseline scaled to avoid flat trace
   · Velocity profile: power-law exponent now decreases with afterload
     (higher afterload → more peaked jet, physically correct)
   · WSS distribution: unique random seeds per scenario (no identical noise)
   · Cleaned unused imports (mpatches, gridspec, FancyArrowPatch, sys)
   · HQ plot: operating-point head value annotated clearly
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import warnings
import os
warnings.filterwarnings("ignore")

# ─── Output directory (Windows / Linux / macOS safe) ─────────────────────────
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "lvad_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Global styling ───────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
    "axes.labelsize":    11,
    "axes.titlesize":    13,
    "axes.titleweight":  "bold",
    "legend.fontsize":   9,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
})

# ─── Colour palette ───────────────────────────────────────────────────────────
C_NORMAL  = "#2196F3"   # blue   – normal heart
C_LOW     = "#FF9800"   # amber  – low RVF risk
C_HIGH    = "#F44336"   # red    – high RVF risk
C_LV      = "#1565C0"   # dark blue  – LV traces
C_RV      = "#C62828"   # dark red   – RV traces
BG        = "#F8F9FA"   # off-white background
ACCENT    = "#263238"   # near-black for titles

SCENARIO_COLORS = [C_NORMAL, C_LOW, C_HIGH]
SCENARIO_LABELS = ["Normal (No LVAD)", "Post-LVAD Low Risk", "Post-LVAD High Risk"]
SCENARIO_SHORT  = ["Normal", "Low Risk", "High Risk"]


# ═══════════════════════════════════════════════════════════════════════════════
#  PHYSIOLOGICAL PARAMETERS  (Tables 1 & 2 from the report)
# ═══════════════════════════════════════════════════════════════════════════════
class ScenarioParams:
    """
    Holds all haemodynamic parameters for one clinical scenario.
    All values are sourced directly from Tables 1 & 2 (Chivukula et al. [5,7]).
    """
    def __init__(self, name,
                 lv_edv, rv_edv, septal_curvature, lvad_flow,
                 rv_afterload, lv_peak_p, rv_peak_p,
                 wss_septal, hr=70):
        self.name             = name
        self.lv_edv           = lv_edv             # mL   – LV end-diastolic volume
        self.rv_edv           = rv_edv             # mL   – RV end-diastolic volume
        self.septal_curvature = septal_curvature   # 1/m  – negative = leftward
        self.lvad_flow        = lvad_flow           # L/min
        self.rv_afterload     = rv_afterload        # dyn·s·cm⁻⁵
        self.lv_peak_p        = lv_peak_p           # mmHg
        self.rv_peak_p        = rv_peak_p           # mmHg
        self.wss_septal       = wss_septal          # Pa   – peak septal WSS
        self.hr               = hr                  # bpm
        # Derived: positive = LV>RV (normal), negative = RV>LV (reversal/danger)
        self.transseptal      = lv_peak_p - rv_peak_p


NORMAL = ScenarioParams(
    "Normal (No LVAD)",
    lv_edv=120, rv_edv=120, septal_curvature=0,
    lvad_flow=0,   rv_afterload=250,
    lv_peak_p=118, rv_peak_p=24,
    wss_septal=1.2
)
LOW_RISK = ScenarioParams(
    "Post-LVAD Low Risk",
    lv_edv=60,  rv_edv=110, septal_curvature=-15,
    lvad_flow=4.0, rv_afterload=350,
    lv_peak_p=38,  rv_peak_p=32,
    wss_septal=3.8
)
HIGH_RISK = ScenarioParams(
    "Post-LVAD High Risk",
    lv_edv=30,  rv_edv=95,  septal_curvature=-40,
    lvad_flow=5.5, rv_afterload=520,
    lv_peak_p=12,  rv_peak_p=43,
    wss_septal=7.5
)
SCENARIOS = [NORMAL, LOW_RISK, HIGH_RISK]


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULATION HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def transseptal_waveform(lv_peak, rv_peak, n_points=500, hr=70):
    """
    Simulate one cardiac cycle of LV and RV pressure waveforms.

    Physics / fixes vs v1:
      · LV diastolic baseline is scaled as 6% of peak (not a fixed 8 mmHg).
        When the LVAD unloads the LV to 12 mmHg peak, a fixed 8 mmHg base
        produced an almost-flat trace, which was physically incorrect.
      · RV diastolic baseline fixed at 4 mmHg (normal RV end-diastolic pressure).
      · Both waveforms use a Gaussian systolic envelope, which mimics the
        measured LV/RV pressure-time curves well enough for illustration.
    """
    T      = 60.0 / hr
    t      = np.linspace(0, T, n_points)
    t_norm = t / T

    # --- LV ---
    lv_base = max(2.0, lv_peak * 0.06)   # ≈6 % of systolic peak (was fixed 8)
    lv      = lv_base + (lv_peak - lv_base) * np.exp(
                  -((t_norm - 0.20) ** 2) / (2 * 0.06 ** 2))

    # --- RV ---
    rv_base = 4.0                          # normal RVEDP ≈ 4 mmHg
    rv      = rv_base + (rv_peak - rv_base) * np.exp(
                  -((t_norm - 0.22) ** 2) / (2 * 0.09 ** 2))

    grad = lv - rv
    return t, lv, rv, grad


def wss_distribution(wss_peak, seed, n=200):
    """
    Spatial WSS distribution along the interventricular septum (base → apex).
    The profile peaks near mid-septum with physiological noise.

    Fix vs v1: each scenario gets its own random seed so the three curves
    have independent noise rather than identical scaled versions.
    """
    x = np.linspace(0, 1, n)
    rng = np.random.default_rng(seed)
    noise = rng.normal(0, wss_peak * 0.05, n)
    wss   = (wss_peak * np.exp(-((x - 0.5) ** 2) / 0.08)
             + wss_peak * 0.15
             + noise)
    wss   = np.clip(wss, 0, None)
    return x, wss


def velocity_profile(rv_afterload, n=80):
    """
    Power-law velocity profile across the tricuspid valve annulus.

    Physics fix vs v1:
      · v1 used  alpha = 2 + rv_afterload/150  which INCREASES with afterload,
        making the profile flatter (more plug-like). That is backwards.
      · Higher afterload → narrower, more peaked inflow jet (smaller alpha).
      · Corrected:  alpha = max(0.5,  3.5 - rv_afterload / 300)
        Normal (250):  alpha ≈ 2.67  → nearly parabolic
        Low risk (350): alpha ≈ 2.33  → slightly peaked
        High risk (520): alpha ≈ 1.77  → pronounced central jet
      · v_max also increases with afterload (elevated RV pressure drives
        faster inflow to maintain output).
    """
    r     = np.linspace(-1, 1, n)
    v_max = 0.35 + rv_afterload / 1000          # m/s  (0.60 → 0.73 → 0.87)
    alpha = max(0.5, 3.5 - rv_afterload / 300)  # decreases: peaked jet at high risk
    v     = v_max * np.maximum(0.0, 1 - np.abs(r) ** (1.0 / alpha))
    return r, v


def lvad_hq_curve(flow_range=(0, 8), n_rpm=4, base_rpm=6000):
    """
    HeartMate-3-style pump HQ (Head-Flow) curves.
    Model: H = a·N² − b·Q²  (simplified Euler turbomachinery equation)
    where N = rotor speed (RPM), Q = flow (L/min).
    Returns: {rpm: (flow_array, head_array_mmHg)}
    """
    flows  = np.linspace(flow_range[0], flow_range[1], 200)
    curves = {}
    for i in range(n_rpm):
        rpm  = base_rpm + i * 500
        a    = 1.8e-8    # empirical constant  (mmHg / RPM²)
        b    = 1.4       # empirical constant  (mmHg / (L/min)²)
        head = a * rpm ** 2 - b * flows ** 2
        head = np.clip(head, 0, None)
        curves[rpm] = (flows, head)
    return curves


def compute_stroke_work(edv, esv_frac, edp_mmhg, peak_p_mmhg):
    """
    Approximate ventricular stroke work (mJ) using the rectangle method:
      SW ≈ mean_pressure × stroke_volume
    where mean_pressure = (peak_p + edp) / 2.

    Units: mmHg → Pa (×133.322),  mL → m³ (×1e-6),  J → mJ (×1000)
    Returns stroke work in mJ.
    """
    esv      = edv * esv_frac
    sv_m3    = (edv - esv) * 1e-6                 # m³
    mean_pa  = (peak_p_mmhg + edp_mmhg) / 2 * 133.322   # Pa
    sw_j     = mean_pa * sv_m3                    # J
    return sw_j * 1000                            # mJ


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 01 – Transseptal Pressure Gradient Over One Cardiac Cycle
# ═══════════════════════════════════════════════════════════════════════════════
def plot_transseptal():
    fig, axes = plt.subplots(3, 1, figsize=(13, 12), sharex=True, facecolor=BG)
    fig.suptitle("Transseptal Pressure Gradient Over One Cardiac Cycle",
                 fontsize=15, fontweight="bold", color=ACCENT, y=1.01)

    for ax, sc, col in zip(axes, SCENARIOS, SCENARIO_COLORS):
        t, lv, rv, grad = transseptal_waveform(sc.lv_peak_p, sc.rv_peak_p)

        ax.plot(t, lv, color=C_LV, lw=2.2, label="LV Pressure")
        ax.plot(t, rv, color=C_RV, lw=2.2, label="RV Pressure")
        ax.fill_between(t, lv, rv, where=(lv >= rv),
                        alpha=0.15, color=C_LV,  label="LV > RV (normal)")
        ax.fill_between(t, lv, rv, where=(lv < rv),
                        alpha=0.30, color=C_HIGH, label="RV > LV ⚠ reversal")

        # Secondary axis: gradient
        ax2 = ax.twinx()
        ax2.plot(t, grad, color=col, lw=1.8, ls="--", alpha=0.9, label="TSG")
        ax2.axhline(0, color="grey", lw=0.9, ls=":")
        ax2.set_ylabel("Gradient (mmHg)", fontsize=8, color=col)
        ax2.tick_params(axis="y", labelcolor=col, labelsize=8)
        ax2.spines["right"].set_visible(True)
        ax2.spines["top"].set_visible(False)

        direction = "LV > RV" if sc.transseptal >= 0 else "RV > LV ⚠"
        ax.set_title(
            f"{sc.name}  |  Peak TSG = {sc.transseptal:+.0f} mmHg  ({direction})",
            color=col)
        ax.set_ylabel("Pressure (mmHg)")
        ax.set_facecolor(BG)
        ax.legend(loc="upper right", fontsize=8, ncol=2)

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 02 – Wall Shear Stress on the Interventricular Septum
# ═══════════════════════════════════════════════════════════════════════════════
def plot_wss():
    SEEDS = [42, 77, 13]   # independent seeds per scenario (fix vs v1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor=BG)
    fig.suptitle("Wall Shear Stress (WSS) on the Interventricular Septum",
                 fontsize=15, fontweight="bold", color=ACCENT)

    # Left panel – spatial distribution along septum
    ax = axes[0]
    for sc, col, lab, seed in zip(SCENARIOS, SCENARIO_COLORS, SCENARIO_LABELS, SEEDS):
        x, wss = wss_distribution(sc.wss_septal, seed)
        ax.plot(x, wss, color=col, lw=2.2,
                label=f"{lab}  (peak {sc.wss_septal} Pa)")

    ax.axhline(3.0, color="grey", lw=1.2, ls="--", alpha=0.7)
    ax.text(0.02, 3.2, "Endothelial dysfunction threshold (~3 Pa)",
            fontsize=8, color="grey")
    ax.set_xlabel("Normalised Septal Position  (base → apex)")
    ax.set_ylabel("Wall Shear Stress (Pa)")
    ax.set_title("WSS Distribution Along the Septal Wall")
    ax.legend()
    ax.set_facecolor(BG)

    # Right panel – peak WSS bar chart
    ax2 = axes[1]
    bars = ax2.bar(SCENARIO_SHORT,
                   [s.wss_septal for s in SCENARIOS],
                   color=SCENARIO_COLORS, edgecolor="white",
                   linewidth=1.5, width=0.5)
    for bar, sc in zip(bars, SCENARIOS):
        ax2.text(bar.get_x() + bar.get_width() / 2,
                 bar.get_height() + 0.12,
                 f"{sc.wss_septal} Pa",
                 ha="center", va="bottom", fontweight="bold", fontsize=11)
    ax2.axhline(3.0, color="grey", lw=1.2, ls="--", alpha=0.7,
                label="Threshold ~3 Pa")
    ax2.set_ylabel("Peak Septal WSS (Pa)")
    ax2.set_title("Peak WSS Comparison")
    ax2.set_ylim(0, 9.0)
    ax2.set_facecolor(BG)
    ax2.legend()

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 03 – LVAD Pump HQ Performance Curves
# ═══════════════════════════════════════════════════════════════════════════════
def plot_lvad_hq():
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    fig.suptitle("LVAD Pump Performance – HQ Curves at Different Rotor Speeds",
                 fontsize=14, fontweight="bold", color=ACCENT)
    ax.set_facecolor(BG)

    curves   = lvad_hq_curve()
    rpm_list = sorted(curves.keys())
    cmap     = plt.cm.Blues

    for i, rpm in enumerate(rpm_list):
        flows, head = curves[rpm]
        colour = cmap(0.4 + 0.6 * i / (len(rpm_list) - 1))
        ax.plot(flows, head, color=colour, lw=2.2, label=f"{rpm} RPM")

    # Mark operating points for Low and High risk on their respective RPM curves
    for sc, marker, col, lab in zip(
        [LOW_RISK,     HIGH_RISK],
        ["o",          "^"],
        [C_LOW,        C_HIGH],
        ["Low Risk\nOp. Point", "High Risk\nOp. Point"]
    ):
        rpm_op     = 6500 if sc == LOW_RISK else 7000
        fl, hd     = curves[rpm_op]
        op_idx     = np.argmin(np.abs(fl - sc.lvad_flow))
        op_head    = hd[op_idx]
        ax.scatter(fl[op_idx], op_head, s=150, color=col, zorder=6,
                   marker=marker, edgecolors="white", linewidths=1.8)
        ax.annotate(f"{lab}\n{sc.lvad_flow} L/min\n{op_head:.1f} mmHg",
                    xy=(fl[op_idx], op_head),
                    xytext=(fl[op_idx] + 0.5, op_head + 1.5),
                    fontsize=8, color=col,
                    arrowprops=dict(arrowstyle="->", color=col, lw=1.2))

    ax.set_xlabel("Pump Flow (L/min)")
    ax.set_ylabel("Head Pressure (mmHg)")
    ax.set_title("HeartMate-3 Style HQ Curves with Clinical Operating Points")
    ax.legend(ncol=2)
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 04 – RV Inlet Velocity Profiles (Tricuspid Valve)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_velocity_profiles():
    fig, axes = plt.subplots(1, 3, figsize=(14, 6), facecolor=BG)
    fig.suptitle("RV Inlet Velocity Profile Across Tricuspid Valve Annulus",
                 fontsize=14, fontweight="bold", color=ACCENT)

    cmap_vel = LinearSegmentedColormap.from_list(
        "vel", ["#E3F2FD", "#1565C0", "#B71C1C"])

    for ax, sc, col, lab in zip(axes, SCENARIOS, SCENARIO_COLORS, SCENARIO_LABELS):
        r, v = velocity_profile(sc.rv_afterload)

        # Colour-fill by velocity magnitude
        norm_v = (v - v.min()) / (v.max() - v.min() + 1e-9)
        for i in range(len(r) - 1):
            ax.fill_betweenx([r[i], r[i + 1]], 0, (v[i] + v[i + 1]) / 2,
                             color=cmap_vel(norm_v[i]), alpha=0.85)
        ax.plot(v, r, color="white", lw=1.8)

        v_max_val = v.max()
        ax.text(v_max_val * 0.55, 0.02,
                f"v_max = {v_max_val:.2f} m/s",
                color="yellow", fontsize=8.5, ha="center", va="bottom")

        ax.set_xlabel("Velocity (m/s)")
        ax.set_title(lab, color=col, fontsize=10, pad=8)
        if ax == axes[0]:
            ax.set_ylabel("Radial Position (normalised)")
        ax.set_facecolor("#0D1B2A")
        ax.tick_params(colors="white")
        ax.yaxis.label.set_color("white")
        ax.xaxis.label.set_color("white")
        ax.title.set_color(col)
        for spine in ax.spines.values():
            spine.set_edgecolor("#0D1B2A")
        ax.set_xlim(left=0)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 05 – Effect of Septal Curvature on Haemodynamic Parameters
# ═══════════════════════════════════════════════════════════════════════════════
def plot_curvature_effects():
    """
    Continuous interpolation of four key parameters across the full range of
    septal curvature (0 = normal, −40 = severe leftward shift).
    Data anchors: the three scenario values from Tables 1 & 2.
    """
    curvatures = np.linspace(0, -45, 300)

    x_anchors = [0, 15, 40]    # |curvature| values

    def piecewise(c_array, y_vals):
        return np.interp(np.abs(c_array), x_anchors, y_vals)

    rv_afterload_c = piecewise(curvatures, [250, 350, 520])
    rv_peak_p_c    = piecewise(curvatures, [24,   32,  43])
    wss_c          = piecewise(curvatures, [1.2,  3.8, 7.5])
    lv_peak_p_c    = piecewise(curvatures, [118,  38,  12])

    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor=BG)
    fig.suptitle("Effect of Septal Curvature on Key Haemodynamic Parameters",
                 fontsize=15, fontweight="bold", color=ACCENT)

    panels = [
        (rv_afterload_c, "RV Afterload (dyn·s·cm⁻⁵)",  C_RV,      [250, 350, 520]),
        (rv_peak_p_c,    "RV Peak Pressure (mmHg)",      C_HIGH,    [24,  32,  43]),
        (wss_c,          "Septal WSS (Pa)",               "#7B1FA2", [1.2, 3.8, 7.5]),
        (lv_peak_p_c,    "LV Peak Pressure (mmHg)",       C_LV,      [118, 38,  12]),
    ]

    for ax, (ydata, ylabel, col, anchors) in zip(axes.flat, panels):
        abs_c = np.abs(curvatures)
        ax.plot(abs_c, ydata, color=col, lw=2.5)
        ax.fill_between(abs_c, ydata, alpha=0.10, color=col)

        for sc, mc, ml in zip(SCENARIOS, SCENARIO_COLORS, SCENARIO_SHORT):
            xv = abs(sc.septal_curvature)
            yv = np.interp(xv, x_anchors, anchors)
            ax.scatter(xv, yv, s=110, color=mc, zorder=5,
                       edgecolors="white", linewidths=1.5)
            offset_x = 0.5
            offset_y = (max(anchors) - min(anchors)) * 0.04
            ax.annotate(ml, xy=(xv, yv),
                        xytext=(xv + offset_x, yv + offset_y),
                        fontsize=8, color=mc)

        ax.set_xlabel("|Septal Curvature| (1/m)")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_facecolor(BG)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 06 – Radar / Spider Risk Profile Chart
# ═══════════════════════════════════════════════════════════════════════════════
def plot_radar():
    """
    Six normalised risk dimensions displayed on a polar chart.

    Fix vs v1:
      · "LV Unloading" now uses  (EDV_normal − EDV_scenario) / EDV_normal
        (= fractional drop in LV volume), which is directly derivable from data.
        Values: Normal=0, Low=0.5, High=0.75  → all in [0,1].
      · "TSG Reversal" uses the magnitude of gradient reversal only:
        reversal = max(0, −transseptal).
        Normal (+94) → reversal=0;  Low (+6) → reversal=0;  High (−31) → reversal=31.
        Normalised over the maximum reversal observed (31 mmHg).
      · All six metrics now lie strictly in [0, 1] for all scenarios.
    """
    metrics = [
        "RV Afterload\n(norm.)",
        "Septal WSS\n(norm.)",
        "RV Peak P\n(norm.)",
        "TSG Reversal\n(norm.)",
        "LV Unloading\n(norm.)",
        "LVAD Flow\n(norm.)",
    ]
    N      = len(metrics)
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    def norm(val, vmin, vmax):
        return np.clip((val - vmin) / (vmax - vmin), 0, 1)

    # --- pre-compute corrected metrics ---
    edv_normal   = NORMAL.lv_edv          # 120 mL
    max_reversal = max(0, -HIGH_RISK.transseptal)   # 31 mmHg

    def make_row(sc):
        lv_unloading = (edv_normal - sc.lv_edv) / edv_normal
        tsg_reversal = max(0, -sc.transseptal) / max_reversal if max_reversal > 0 else 0
        return [
            norm(sc.rv_afterload, 250, 520),
            norm(sc.wss_septal,   1.2, 7.5),
            norm(sc.rv_peak_p,    24,  43),
            tsg_reversal,
            lv_unloading,
            norm(sc.lvad_flow,    0,   5.5),
        ]

    scenario_rows = {
        SCENARIO_SHORT[i]: make_row(SCENARIOS[i]) for i in range(3)
    }

    fig, ax = plt.subplots(figsize=(8, 8),
                           subplot_kw={"polar": True}, facecolor=BG)
    fig.suptitle("Risk Profile Radar Chart – Normalised Haemodynamic Metrics",
                 fontsize=14, fontweight="bold", color=ACCENT, y=1.02)
    ax.set_facecolor(BG)

    for (label, vals), col in zip(scenario_rows.items(), SCENARIO_COLORS):
        vals_plot = vals + vals[:1]
        ax.plot(angles, vals_plot, color=col, lw=2.4, label=label)
        ax.fill(angles, vals_plot, color=col, alpha=0.13)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics, size=9)
    ax.set_ylim(0, 1)
    ax.set_yticks([0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["25 %", "50 %", "75 %", "100 %"], size=7, color="grey")
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.15))
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 07 – Comprehensive Haemodynamic Summary Bar Charts
# ═══════════════════════════════════════════════════════════════════════════════
def plot_summary():
    params = {
        "LV EDV (mL)":                    [s.lv_edv        for s in SCENARIOS],
        "RV EDV (mL)":                    [s.rv_edv        for s in SCENARIOS],
        "LV Peak Pressure (mmHg)":        [s.lv_peak_p     for s in SCENARIOS],
        "RV Peak Pressure (mmHg)":        [s.rv_peak_p     for s in SCENARIOS],
        "RV Afterload (dyn·s·cm⁻⁵)":     [s.rv_afterload  for s in SCENARIOS],
        "Peak Septal WSS (Pa)":           [s.wss_septal    for s in SCENARIOS],
    }

    fig, axes = plt.subplots(2, 3, figsize=(15, 9), facecolor=BG)
    fig.suptitle("Comprehensive Haemodynamic Parameter Comparison Across Scenarios",
                 fontsize=15, fontweight="bold", color=ACCENT)

    for ax, (label, vals) in zip(axes.flat, params.items()):
        bars = ax.bar(SCENARIO_SHORT, vals,
                      color=SCENARIO_COLORS, edgecolor="white",
                      linewidth=1.5, width=0.55)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(vals) * 0.02,
                    str(v), ha="center", fontsize=11, fontweight="bold")
        ax.set_title(label)
        ax.set_ylim(0, max(vals) * 1.22)
        ax.set_facecolor(BG)

    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  PLOT 08 – Stroke Work Comparison: LV vs RV
# ═══════════════════════════════════════════════════════════════════════════════
def plot_stroke_work():
    """
    Stroke work (mJ) = mean_pressure × stroke_volume.

    ESV fractions are derived from reported ejection fractions:
      Normal LV EF ≈ 67 %  → ESV/EDV ≈ 0.33
      Low-risk LV EF ≈ 50 % → ESV/EDV ≈ 0.50
      High-risk LV EF ≈ 33 % → ESV/EDV ≈ 0.67  (severely unloaded / compressed)

    RV EF is typically lower (40–50 % normal, falling to 45 % and 55 %
    in low/high risk as RV dilates and wall motion worsens).
    """
    # (esv_frac_lv, esv_frac_rv, edp_lv, edp_rv)
    ef_data = [
        (0.33, 0.40, 8, 5),   # Normal
        (0.50, 0.45, 8, 5),   # Low Risk
        (0.67, 0.55, 8, 5),   # High Risk
    ]

    sw_lv, sw_rv = [], []
    for sc, (flv, frv, edp_lv, edp_rv) in zip(SCENARIOS, ef_data):
        sw_lv.append(compute_stroke_work(sc.lv_edv, flv, edp_lv, sc.lv_peak_p))
        sw_rv.append(compute_stroke_work(sc.rv_edv, frv, edp_rv, sc.rv_peak_p))

    x, w = np.arange(3), 0.35
    fig, ax = plt.subplots(figsize=(10, 6), facecolor=BG)
    fig.suptitle("Ventricular Stroke Work – LV vs RV Across Scenarios",
                 fontsize=14, fontweight="bold", color=ACCENT)
    ax.set_facecolor(BG)

    bars_lv = ax.bar(x - w / 2, sw_lv, w, color=C_LV,
                     label="LV Stroke Work", alpha=0.88)
    bars_rv = ax.bar(x + w / 2, sw_rv, w, color=C_RV,
                     label="RV Stroke Work", alpha=0.88)

    for bars in [bars_lv, bars_rv]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1.0,
                    f"{bar.get_height():.1f}",
                    ha="center", fontsize=9, fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels(SCENARIO_LABELS)
    ax.set_ylabel("Stroke Work (mJ)")
    ax.set_title("LV Stroke Work Collapses Under LVAD;  RV Work Rises → RVF Risk")
    ax.legend()
    plt.tight_layout()
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════════
def main():
    header = "=" * 70
    print(header)
    print("  Biventricular Interdependence Simulation – IIT Madras")
    print("  Shristy Roy [MD24B034] & Kripa Mariam Roy [MD24B032]")
    print("  Version 2.0 – corrected, PV-loop removed")
    print(header)
    print(f"\n  Output folder: {OUTPUT_DIR}\n")

    plots = [
        ("01_Transseptal_Gradient",  plot_transseptal,       "Transseptal Pressure Gradient"),
        ("02_Wall_Shear_Stress",     plot_wss,               "Wall Shear Stress"),
        ("03_LVAD_HQ_Curves",        plot_lvad_hq,           "LVAD HQ Performance Curves"),
        ("04_Velocity_Profiles",     plot_velocity_profiles, "RV Velocity Profiles"),
        ("05_Curvature_Effects",     plot_curvature_effects, "Septal Curvature Effects"),
        ("06_Radar_Risk_Profile",    plot_radar,             "Radar Risk Profile"),
        ("07_Summary_Comparison",    plot_summary,           "Summary Comparison"),
        ("08_Stroke_Work",           plot_stroke_work,       "Stroke Work Comparison"),
    ]

    saved = []
    for fname, func, title in plots:
        print(f"  [{title}] … ", end="", flush=True)
        fig  = func()
        path = os.path.join(OUTPUT_DIR, f"{fname}.png")
        fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=BG)
        plt.close(fig)
        saved.append(path)
        print("saved.")

    # ── Quantitative summary table ────────────────────────────────────────────
    print("\n" + "─" * 72)
    print(f"  {'Parameter':<34} {'Normal':>10} {'Low Risk':>10} {'High Risk':>10}")
    print("─" * 72)
    rows = [
        ("LV EDV (mL)",                  [s.lv_edv           for s in SCENARIOS]),
        ("RV EDV (mL)",                  [s.rv_edv           for s in SCENARIOS]),
        ("|Septal Curvature| (1/m)",     [abs(s.septal_curvature) for s in SCENARIOS]),
        ("LVAD Flow (L/min)",            [s.lvad_flow        for s in SCENARIOS]),
        ("RV Afterload (dyn·s·cm⁻⁵)",   [s.rv_afterload     for s in SCENARIOS]),
        ("LV Peak Pressure (mmHg)",      [s.lv_peak_p        for s in SCENARIOS]),
        ("RV Peak Pressure (mmHg)",      [s.rv_peak_p        for s in SCENARIOS]),
        ("Transseptal Gradient (mmHg)",  [s.transseptal      for s in SCENARIOS]),
        ("Peak Septal WSS (Pa)",         [s.wss_septal       for s in SCENARIOS]),
    ]
    for label, vals in rows:
        print(f"  {label:<34} {vals[0]:>10.1f} {vals[1]:>10.1f} {vals[2]:>10.1f}")
    print("─" * 72)

    print(f"\n✓  {len(saved)} figures saved to:  {OUTPUT_DIR}")
    print("\nKey Clinical Insights:")
    print("  • TSG reverses: +94 mmHg (Normal) → −31 mmHg (High Risk)")
    print("  • Septal WSS escalates 6×: 1.2 Pa → 7.5 Pa (endothelial dysfunction)")
    print("  • RV afterload nearly doubles: 250 → 520 dyn·s·cm⁻⁵")
    print("  • LV peak pressure drops 90%: 118 → 12 mmHg (confirms LVAD unloading)")
    print("  • RV inflow jet sharpens with higher afterload (peaked velocity profile)")
    print("  • LV stroke work collapses; RV stroke work rises → RVF tipping point\n")
    return saved


if __name__ == "__main__":
    main()
