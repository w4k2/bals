"""
Analiza zależności od szumu
"""
import numpy as np
import config as config
import matplotlib.pyplot as plt
from scipy.signal import medfilt
from math import pi
from matplotlib import rcParams

# Kernel size
ks = 7

# ylim
ylim = (0.25, 1)
figsize = (7, 3.5)
ncol = 4

# Set plot params
rcParams["font.family"] = "monospace"
colors = [
    (0, 0, 0),
    (0.15, 0.6, 0.15),
    (0.9, 0, 0),
    (0.9, 0, 0),
    (0.9, 0, 0),
    (0.15, 0.15, 0.9),
    (0.15, 0.15, 0.9),
    (0.15, 0.15, 0.9),
]
ls = [":", "-", ":", "-.", "-", ":", "-.", "-"]
lw = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
titles = [
    "Sudden non-recurring drift",
    "Gradual non-recurring drift",
    "Incremental non-recurring drift",
    "Sudden recurring drift",
    "Gradual recurring drift",
    "Incremental recurring drift",
]

contexts = ["drift", "concept", "weights", "y_flip"]
n_chunks, chunk_size = config.properties_var()
"""
Read scores:
0. drift
1. concept
2. weights
3. y_flip
4. methods
5. chunks
6. metrics
"""
scores = np.load("scores-2.npy")


def plot_runs_on_context(c, context_scores):
    for m_idx, metric in enumerate(config.metrics()):
        metric_scores = context_scores[:, :, :, m_idx]
        # Iterate options inside context
        for r, runs in enumerate(metric_scores):
            context_instance = getattr(config, "%s_var" % contexts[c])()[r]
            # Prepare plot
            plt.clf()
            fig = plt.figure(figsize=figsize)
            ax = plt.axes()
            for m, method in enumerate(config.methods()):
                plt.plot(
                    medfilt(runs[m], kernel_size=ks),
                    label="%s\n%.3f" % (method, np.mean(runs[m])),
                    c=colors[m],
                    ls=ls[m],
                    lw=lw[m],
                )

            # Prettify plot
            ax.legend(
                loc=8, fancybox=False, shadow=True, ncol=ncol, fontsize=9, frameon=False
            )
            plt.grid(ls=":", c=(0.7, 0.7, 0.7))
            plt.xlim(0, n_chunks - 1)
            axx = plt.gca()
            axx.spines["right"].set_visible(False)
            axx.spines["top"].set_visible(False)

            plt.title(
                titles[r],
                # "%s - %s - %s"
                #% (contexts[c], context_instance, list(config.metrics().keys())[m_idx]),
                fontfamily="serif",
                y=1.04,
                fontsize=10,
            )
            plt.ylim(ylim[0], ylim[1])
            plt.xticks(fontfamily="serif")
            plt.yticks(fontfamily="serif")
            plt.ylabel("score", fontfamily="serif", fontsize=8)
            plt.xlabel("chunks", fontfamily="serif", fontsize=8)
            plt.tight_layout()

            filename = "%s-%i-%i" % (contexts[c], m_idx, r)
            plt.savefig("plots/%s-2.eps" % filename)
            plt.savefig("plots/%s-2.png" % filename)
            plt.savefig("foo.png")


def plot_radars_on_context(c, context_scores):
    mean_context_scores = np.mean(context_scores, axis=2)
    # Iterate options inside context
    for r, runs in enumerate(mean_context_scores):
        context_instance = getattr(config, "%s_var" % contexts[c])()[r]

        # Prepare
        metrics_labels = list(config.metrics().keys())

        angles = [
            n / float(len(metrics_labels)) * 2 * pi for n in range(len(metrics_labels))
        ]
        metrics_labels += [metrics_labels[0]]
        angles += [angles[0]]

        plt.clf()
        ax = plt.subplot(111, polar=True)

        # Prettify
        ax.set_theta_offset(pi / 2)
        ax.set_theta_direction(-1)
        ax.spines["polar"].set_visible(False)

        plt.xticks(angles, metrics_labels)

        for m, method in enumerate(config.methods()):
            c_run = list(runs[m])
            c_run += [c_run[0]]
            ax.plot(angles, c_run, label=method, c=colors[m], ls=ls[m], lw=lw[m])

        # Add legend
        plt.legend(
            loc="lower center",
            ncol=ncol,
            columnspacing=1,
            frameon=False,
            bbox_to_anchor=(0.5, -0.2),
            fontsize=8,
        )

        # Add a grid
        plt.grid(ls=":", c=(0.7, 0.7, 0.7))

        # Title
        plt.title(
            "%s - %s" % (contexts[c], context_instance),
            fontfamily="serif",
            y=1.08,
            size=11,
        )

        # Draw labels
        a = np.linspace(0, 1, 6)
        plt.yticks(a[1:], ["%.1f" % f for f in a[1:]], fontsize=6, rotation=90)
        plt.ylim(ylim[0], ylim[1])
        plt.gcf().set_size_inches(4, 4)
        plt.gcf().canvas.draw()
        angles = np.rad2deg(angles)

        ax.set_rlabel_position((angles[0] + angles[1]) / 2)

        har = [(a >= 90) * (a <= 270) for a in angles]

        for z, (label, angle) in enumerate(zip(ax.get_xticklabels(), angles)):
            x, y = label.get_position()
            lab = ax.text(
                x, y, label.get_text(), transform=label.get_transform(), fontsize=6
            )
            lab.set_rotation(angle)

            if har[z]:
                lab.set_rotation(180 - angle)
            else:
                lab.set_rotation(-angle)
            lab.set_verticalalignment("center")
            lab.set_horizontalalignment("center")
            lab.set_rotation_mode("anchor")

        for z, (label, angle) in enumerate(zip(ax.get_yticklabels(), a)):
            x, y = label.get_position()
            lab = ax.text(
                x,
                y,
                label.get_text(),
                transform=label.get_transform(),
                fontsize=4,
                c=(0.7, 0.7, 0.7),
            )
            lab.set_rotation(-(angles[0] + angles[1]) / 2)

            lab.set_verticalalignment("bottom")
            lab.set_horizontalalignment("center")
            lab.set_rotation_mode("anchor")

        ax.set_xticklabels([])
        ax.set_yticklabels([])

        plt.savefig("foo.png", bbox_inches="tight", dpi=250)

        filename = "%s-%i" % (contexts[c], r)
        plt.savefig("plots/%s-2" % filename, bbox_inches="tight", dpi=250)


# Search for analytic contexts
for c, context in enumerate(contexts):
    # Select only contexts with more than one option
    if scores.shape[c] > 1:
        # Select axes to flatten
        flat_axes = tuple(np.where(np.array(list(range(len(contexts)))) != c)[0])

        # Get context scores
        context_scores = np.mean(scores, axis=flat_axes)
        print("\tcontext %i - %s" % (c, context), context_scores.shape)

        # plot_radars_on_context(c, context_scores)
        plot_runs_on_context(c, context_scores)
