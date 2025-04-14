import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def plot_kde_fill(ax, data, color, linestyle, label, fill=True, alpha_fill=0.3, xmin=None, xmax=None, num_points=1000, bw_method=None, hist=False, bins=120):
    """
    Plots a KDE curve with optional fill under the curve and returns the x and y values.

    Parameters:
    - ax: The matplotlib axis to plot on.
    - data: Data points to calculate the KDE from.
    - color: Color for the line and fill.
    - linestyle: Line style for the curve.
    - label: Label for the legend.
    - fill: Whether to fill under the curve (default: True).
    - alpha_fill: Transparency of the fill (default: 0.3).
    - num_points: Number of points for the KDE curve (default: 1000).
    - bw_method: Bandwidth method for KDE (default: None).
    - hist: Whether to plot a histogram of the data (default: False).

    Returns:
    - x: The x values of the KDE curve.
    - y: The y values of the KDE curve.
    """
    data = np.array(data) if not isinstance(data, np.ndarray) else data
    kde = gaussian_kde(data, bw_method=bw_method)
    xmin = data.min() if xmin is None else xmin
    xmax = data.max() if xmax is None else xmax
    x = np.linspace(xmin, xmax, num_points)
    y = kde(x)
    if fill:
        ax.plot(x, y, color=color, linestyle=linestyle, label=None, alpha=1.0)
        ax.fill_between(
            x, 0, y,
            facecolor=color if fill else "none",
            edgecolor=color,
            linestyle=linestyle,
            label=label,
            alpha=alpha_fill)
    else:
        ax.plot(x, y, color=color, linestyle=linestyle, label=label, alpha=1.0)
    if hist:
        ax.hist(data, bins=bins, histtype='step', density=True, color=color, alpha=0.25, linewidth=0.3)
    return x, y


def plot_joint_distribution(xs, ys, x_min, x_max, y_min, y_max, x_label, y_label,
              image_prefix, text, show):
    """
    Plot join distribution and its marginal distributions
    """
    sns.set(style="white")
    num_samples = xs.size
    g = sns.jointplot(x=xs, y=ys, kind="kde", fill=True, bw_adjust=0.5,
                      height=5,        # Height of the joint plot (in inches)
                      ratio=4)         # Size ratio of marginal plots to joint)

    g.fig.set_size_inches(6, 6)
    g.ax_joint.scatter(xs, ys, s=5, color="black", alpha=0.3,
                       marker='o', linewidths=0)  # You can tweak size, color, alpha

    # Set axis limits
    g.ax_joint.set_xlim(x_min, x_max)
    g.ax_joint.set_ylim(y_min, y_max)

    # Force ticks to show
    g.ax_joint.tick_params(left=True, bottom=True, direction='in')
    g.ax_marg_x.tick_params(bottom=True)
    g.ax_marg_y.tick_params(left=True)

    g.fig.subplots_adjust(hspace=0.01, wspace=0.01)
    g.set_axis_labels(x_label, y_label, labelpad=8)
    g.ax_joint.text(0.15, 0.15, text + " enviroment number {}".format(num_samples), color='black', fontsize=14,
        transform=g.ax_joint.transAxes, verticalalignment='top')
    g.fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    g.fig.savefig(f"{image_prefix}.pdf")
    g.fig.savefig(f"{image_prefix}.png", dpi=300)
    g.fig.savefig(f"{image_prefix}.svg")
    if show: plt.show()
    plt.close()

def plot_joint_distributions(z_thresholds, npz_prefix, npz_x, npz_y, colors, x_min, x_max, y_min, y_max, x_label, y_label, image_prefix, text, show):
    sns.set(style="white")
    fig = plt.figure(figsize=(6, 6))
    grid = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.05, wspace=0.05)
    ax_joint = fig.add_subplot(grid[1, 0])
    ax_marg_x = fig.add_subplot(grid[0, 0], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(grid[1, 1], sharey=ax_joint)

    for k, (key, value) in enumerate(z_thresholds.items()):
        xs, ys = np.load(f"{npz_prefix}_{key}.npz")[npz_x], np.load(f"{npz_prefix}_{key}.npz")[npz_y]
        color = colors[key]

        if key == "All": # Joint KDE for 'All' only
            sns.kdeplot(x=xs, y=ys, fill=True, bw_adjust=0.5, ax=ax_joint,
                        cmap=sns.light_palette(color, as_cmap=True))
        else: # Scatter for Top and Bottom only
            ax_joint.scatter(xs, ys, s=5, marker = ',' if key =="Bottom"
                             else 'x', color=color, alpha=0.5,
                             label=f'{key} samples')

        sns.kdeplot(x=xs, ax=ax_marg_x, color=color, fill=False if key !=
                    "All" else True,
                    bw_adjust=0.5, alpha=0.3 if key ==
                    "All" else 1, label=f"{key} samples")
        sns.kdeplot(y=ys, ax=ax_marg_y, color=color, fill=False if key !=
                    "All" else True,
                    bw_adjust=0.5, alpha=0.3 if key ==
                    "All" else 1, label=f"{key} samples")

    ax_joint.set_xlim(x_min, x_max)
    ax_joint.set_ylim(y_min, y_max)
    ax_joint.set_xlabel(x_label)
    ax_joint.set_ylabel(y_label)
    ax_joint.tick_params(direction="in")
    #ax_joint.legend(loc='upper left', frameon=False)
    ax_marg_x.legend(loc='lower left', frameon=False)

    # Hide axis ticks for marginal plots
    ax_marg_x.axis("off")
    ax_marg_y.axis("off")

    # Create custom legend handles for marginal lines
    legend_lines = [ Line2D([0], [0], color=color, lw=2, label=label)
        for label, color in colors.items() ]
    ax_marg_x.legend(handles=legend_lines, loc='lower center', frameon=False,
                     ncol=1, bbox_to_anchor=(0.15, 0.00), fontsize=10)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15,
                        hspace=0.01, wspace=0.01)

    fig.savefig(f"{image_prefix}.pdf", bbox_inches='tight')
    fig.savefig(f"{image_prefix}.png", dpi=300, bbox_inches='tight')
    fig.savefig(f"{image_prefix}.svg", bbox_inches='tight')

    if show: plt.show()
    plt.close()

def plot_joint_distributions_in_row(z_thresholds, npz_prefix, npz_x, npz_y,
                                    x_min, x_max, y_min, y_max, x_label,
                                    y_label, image_prefix, text, show):
        nbin = 50
        figsize = (8, 2.5)
        cmap = 'Greens'
        xgrid = np.linspace(x_min, x_max, nbin)
        ygrid = np.linspace(y_min, y_max, nbin)
        X, Y = np.meshgrid(xgrid, ygrid)
        fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True, gridspec_kw={'wspace': 0.025}, constrained_layout=True)
        for k, (key, value) in enumerate(z_thresholds.items()):
            xs, ys = np.load(f"{npz_prefix}_{key}.npz")[npz_x], np.load(f"{npz_prefix}_{key}.npz")[npz_y]
            xy = np.vstack([xs, ys])
            kde = gaussian_kde(xy)
            positions = np.vstack([X.ravel(), Y.ravel()])
            Z = kde(positions).reshape(X.shape)

            contour = axes[k].pcolormesh(X, Y, Z, cmap=cmap, vmin=0, vmax=1200)
            axes[k].scatter(xs, ys, s=0.5, color='black', alpha=0.1)
            axes[k].tick_params(axis='both', direction='in', top=True, right=True)
            axes[k].text(0.2, 0.95, key, color='black', fontsize=10, transform=axes[k].transAxes, verticalalignment='top')
            axes[k].set_xlim(x_min, x_max)
            axes[k].set_ylim(y_min, y_max)
            axes[k].set_xlabel(x_label)
            if k == 0:
                axes[k].set_ylabel(y_label)  # Only the first subplot has a y-label

        # Create shared colorbar
        cbar = fig.colorbar(contour, ax=axes, orientation='vertical', fraction=0.05, pad=0.02)
        cbar.set_label(r'$\rho$')

        # Save and show
        plt.savefig(f'{image_prefix}.pdf')
        plt.savefig(f'{image_prefix}.png', dpi=600)
        plt.savefig(f'{image_prefix}.svg')
        if show: plt.show()
        plt.close()
