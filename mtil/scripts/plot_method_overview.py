"""Simplified method overview (thesis main text). Detailed version:
figures/training_loop_split_clip.drawio. Losses are grouped by purpose and coloured
to match figs/problem_cartoon (blues = tasks/forgetting, orange = zero-shot)."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D
from matplotlib.colors import to_rgb

OUT = "figs/method_overview"
plt.rcParams.update({"font.family": "serif", "mathtext.fontset": "dejavuserif",
                     "pdf.fonttype": 42, "font.size": 7})

INK, GREY = "#2b2a28", "#6b6a66"
LEARN, REMEMBER, ZERO, CLOSE = "#08306b", "#3d85c6", "#eb6834", "#6b6a66"

fig, ax = plt.subplots(figsize=(6.3, 2.7))
ax.set_xlim(0, 10.6); ax.set_ylim(-0.35, 4.55); ax.axis("off")


def box(cx, cy, w, h, text, ec=INK, fc="white", ls="-", fs=7, lw=0.8, **kw):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0,rounding_size=0.08",
                                ec=ec, fc=fc, ls=ls, lw=lw))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs, color=INK, **kw)


def arrow(p, q, dashed=False, color=INK):
    ax.annotate("", xy=q, xytext=p, arrowprops=dict(
        arrowstyle="-|>,head_length=0.35,head_width=0.18", lw=0.8,
        color=GREY if dashed else color, ls=(0, (3, 2)) if dashed else "-",
        shrinkA=0, shrinkB=0))


# inputs
IX, IW, IH = 1.1, 2.15, 0.72
inputs = {"task": (3.75, "Task $t$ data\n$(x_t, y_t)$"),
          "rep": (2.35, "Replay buffer\n$(x_{\\mathrm{rep}}, y_{\\mathrm{rep}})$, $B$ = 11k"),
          "ref": (0.85, "Reference data\nImageNet images,\nCC captions")}
for y, t in inputs.values():
    box(IX, y, IW, IH if t.count("\n") < 2 else 0.9, t, fc="#f4f3f0", ec="#b9b7b1")

# models: one box each, image | text halves
MX, MW = 4.45, 3.0
for cy, h, title, sub, ls, ec in (
        (3.05, 1.4, "Student CLIP $f_\\theta$ (trainable)", "\\theta", "-", INK),
        (0.85, 1.2, "Teacher CLIP $f_{\\theta^*}$ (frozen)", "\\theta^*", (0, (3, 2)), GREY)):
    box(MX, cy, MW, h, None, ec=ec, ls=ls, lw=1.0)
    ax.text(MX, cy + h / 2 - 0.22, title, ha="center", va="center", fontsize=7, color=INK)
    for dx, name, sup in ((-0.62, "Image enc.", "I"), (0.62, "Text enc.", "T")):
        box(MX + dx, cy - 0.17, 1.1, 0.6, f"{name}\n$f^{{{sup}}}_{{{sub}}}$",
            fc="#eaf1f9" if sup == "I" else "#fdf0e8", ec="#b9b7b1", fs=6.5, lw=0.6)

# losses, grouped by purpose
LX, LW, LH = 7.25, 1.05, 0.46
losses = [  # y, name, colour, group label
    (4.1, "$L_{\\mathrm{L2}}$", CLOSE, "stay close to $\\theta_{t-1}$"),
    (3.3, "$L_{\\mathrm{CE}}$", LEARN, "learn task $t$"),
    (2.5, "$L_{\\mathrm{rep}}$", REMEMBER, "remember old tasks"),
    (1.45, "$L_{\\mathrm{ZSCL}}$", ZERO, None),
    (0.75, "$L_{\\mathrm{RD}}$", ZERO, None),
]
for y, name, c, grp in losses:
    box(LX, y, LW, LH, name, ec=c, fc=(*to_rgb(c), 0.12), fs=8, lw=1.0)
    if grp:
        ax.text(LX + LW / 2 + 0.15, y, grp, va="center", fontsize=7, color=c)
ax.text(LX + LW / 2 + 0.15, 2.5 - 0.3,
        "only term that updates the\ntext encoder for old classes",
        va="top", fontsize=5.8, color=GREY, style="italic")
# bracket for the two zero-shot terms
bx = LX + LW / 2 + 0.12
ax.plot([bx, bx + 0.08, bx + 0.08, bx], [1.45, 1.45, 0.75, 0.75], color=ZERO, lw=0.8)
ax.text(bx + 0.23, 1.1, "keep zero-shot", va="center", fontsize=7, color=ZERO)

# arrows: inputs -> models
L_EDGE, R_EDGE = MX - MW / 2, MX + MW / 2
arrow((IX + IW / 2, 3.75), (L_EDGE, 3.4))
arrow((IX + IW / 2, 2.45), (L_EDGE, 2.95))
arrow((IX + IW / 2, 1.0), (L_EDGE, 2.65))
arrow((IX + IW / 2, 2.2), (L_EDGE, 1.05), dashed=True)
arrow((IX + IW / 2, 0.75), (L_EDGE, 0.75), dashed=True)

# arrows: models -> losses (student solid, teacher dashed)
for (y, *_), sy in zip(losses, (3.55, 3.3, 3.0, 2.7, 2.5)):
    tgt = y + (0.1 if y < 2 else 0)
    arrow((R_EDGE, sy), (LX - LW / 2, tgt))
for y in (1.45, 0.75):
    arrow((R_EDGE, 0.85), (LX - LW / 2, y - 0.1), dashed=True)

# total loss and legend
ax.text(10.55, -0.2, "$L = L_{\\mathrm{CE}} + w_r L_{\\mathrm{rep}} + L_{\\mathrm{ZSCL}}"
        " + \\lambda L_{\\mathrm{RD}} + \\alpha L_{\\mathrm{L2}}$",
        ha="right", va="center", fontsize=8, color=INK)
ax.legend(handles=[Line2D([], [], color=INK, lw=0.8, label="gradient flows"),
                   Line2D([], [], color=GREY, lw=0.8, ls=(0, (3, 2)),
                          label="no gradient (frozen)")],
          loc="lower left", bbox_to_anchor=(0.0, -0.06), fontsize=6, frameon=False,
          handlelength=2.2, ncol=2)

for ext in ("pdf", "png"):
    fig.savefig(f"{OUT}.{ext}", dpi=300, bbox_inches="tight")
