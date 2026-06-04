# Copyright 2026. Apache License 2.0.
# Regenerate figures/*.png from the measured numbers (see runs/*.json). Run: python make_figures.py
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)
SOFT, FORG = "#4C72B0", "#DD8452"
plt.rcParams.update({"font.size": 12, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.axisbelow": True, "figure.dpi": 150})

# ---- Figure 1: robustness crossover (the headline finding) ----
eps = [0.0, 0.05, 0.10, 0.20, 0.30]
sm = [2.4160, 2.9332, 3.4166, 4.3635, 5.2290]
sm_e = [0.010, 0.016, 0.024, 0.032, 0.045]
fo = [2.3013, 2.8942, 3.4469, 4.5065, 5.4151]
fo_e = [0.010, 0.011, 0.005, 0.010, 0.010]
fig, ax = plt.subplots(figsize=(8.5, 5.4))
ax.errorbar(eps, sm, yerr=sm_e, marker="o", capsize=4, color=SOFT, label="softmax attention")
ax.errorbar(eps, fo, yerr=fo_e, marker="s", capsize=4, color=FORG, label="forgetting attention")
ax.axvline(0.08, ls=":", color="#888", lw=1)
ax.annotate("crossover ≈ ε 0.08\nforget wins clean, loses under noise",
            xy=(0.08, 3.2), xytext=(0.12, 2.7), fontsize=10, color="#333",
            arrowprops=dict(arrowstyle="->", color="#666"))
ax.set_xlabel("context corruption rate  ε  (fraction of input tokens randomized)")
ax.set_ylabel("held-out bits / char  (lower is better)")
ax.set_title("Forgetting attention's win does NOT survive input noise\n"
             "it is the more accurate model on clean text, the LESS robust one under perturbation",
             fontsize=12.5)
ax.legend(loc="upper left", framealpha=0.95)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig1_robustness_crossover.png")); plt.close(fig)

# ---- Figure 2: the 2x2 factorial (FFN neuron x attention) ----
fig, ax = plt.subplots(figsize=(8, 5.2))
groups = ["softmax attn", "forgetting attn"]
smffn = [2.5245, 2.3357]; smffn_e = [0.018, 0.007]
ibnn = [2.5432, 2.3567]; ibnn_e = [0.013, 0.017]
x = range(len(groups)); w = 0.36
b1 = ax.bar([i - w/2 for i in x], smffn, w, yerr=smffn_e, capsize=4, label="SM FFN", color=SOFT)
b2 = ax.bar([i + w/2 for i in x], ibnn, w, yerr=ibnn_e, capsize=4, label="IBNN FFN", color="#C44E52")
for bars in (b1, b2):
    for b in bars:
        ax.annotate(f"{b.get_height():.3f}", (b.get_x()+b.get_width()/2, b.get_height()),
                    ha="center", va="bottom", fontsize=9, xytext=(0, 2), textcoords="offset points")
ax.set_xticks(list(x)); ax.set_xticklabels(groups)
ax.set_ylabel("held-out bits / char  (lower is better)"); ax.set_ylim(2.2, 2.62)
ax.set_title("2×2 factorial: the win is the forget gate, not the FFN neuron\n"
             "IBNN ≈ 0 under both attention types; no interaction", fontsize=12.5)
ax.legend(loc="upper right", framealpha=0.95)
fig.tight_layout(); fig.savefig(os.path.join(OUT, "fig2_factorial.png")); plt.close(fig)

print("wrote:")
for f in sorted(os.listdir(OUT)):
    if f.endswith(".png"):
        print("  figures/" + f)
