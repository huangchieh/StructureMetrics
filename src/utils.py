import numpy as np
from scipy.stats import gaussian_kde
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import re

def plot_kde_fill(ax, data, color, linestyle, label, fill=True, alpha_fill=0.3, xmin=None, xmax=None, num_points=100, bw_method=None, hist=False, bins=120, marker=None):
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
        ax.plot(x, y, color=color, linestyle=linestyle, marker=marker, markerfacecolor='none', markersize=4, markeredgewidth=1, label=label, alpha=1.0)
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

def plot_joint_distributions(z_thresholds, npz_prefix, npz_x, npz_y, colors, x_min, x_max, y_min, y_max, x_label, y_label, image_prefix, linestypes, show):
    sns.set(style="white")
    fig = plt.figure(figsize=(3.85, 3.85))
    grid = plt.GridSpec(2, 2, width_ratios=[4, 1], height_ratios=[1, 4], hspace=0.015, wspace=0.015)
    ax_joint = fig.add_subplot(grid[1, 0])
    ax_marg_x = fig.add_subplot(grid[0, 0], sharex=ax_joint)
    ax_marg_y = fig.add_subplot(grid[1, 1], sharey=ax_joint)

    for k, (key, value) in enumerate(z_thresholds.items()):
        xs, ys = np.load(f"{npz_prefix}_{key}.npz")[npz_x], np.load(f"{npz_prefix}_{key}.npz")[npz_y]
        color = colors[key]

        # if key == "All": # Joint KDE for 'All' only
        #     sns.kdeplot(x=xs, y=ys, fill=True, bw_adjust=0.5, ax=ax_joint,
        #                 cmap=sns.light_palette(color, as_cmap=True))
        # else: # Scatter for Top and Bottom only
        #     ax_joint.scatter(xs, ys, s=5, marker = ',' if key =="Bottom"
        #                      else 'x', color=color, alpha=0.5,
        #                      label=f'{key} samples')
        ax_joint.scatter(xs, ys, s=5, marker = ',' if key =="Bottom"
                            else 'x', color=color, alpha=0.5,
                            label=f'{key} samples')

        sns.kdeplot(x=xs, ax=ax_marg_x, color=color, fill=False,
                bw_adjust=0.5, alpha=1, label=f"{key} samples", linestyle=linestypes[key])
        sns.kdeplot(y=ys, ax=ax_marg_y, color=color, fill=False, 
                bw_adjust=0.5, alpha=1, label=f"{key} samples", linestyle=linestypes[key])

    ax_joint.set_xlim(x_min, x_max)
    ax_joint.set_ylim(y_min, y_max)
    ax_joint.set_xlabel(x_label)
    ax_joint.set_ylabel(y_label)
    ax_joint.tick_params(direction="in")
    #ax_marg_x.legend(loc='lower left', frameon=False)

    # Hide axis ticks for marginal plots
    ax_marg_x.axis("off")
    ax_marg_y.axis("off")

    # Create custom legend handles for marginal lines
    legend_lines = [ Line2D([0], [0], color=color, lw=2, label=label)
        for label, color in colors.items() ]
    ax_marg_x.legend(handles=legend_lines, loc='lower center', frameon=False,
                     ncol=1, bbox_to_anchor=(0.15, 0.00), fontsize=10)
    fig.subplots_adjust(left=0.25, right=0.99, top=0.92, bottom=0.15,
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
        figsize = (6, 2)
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


def get_scan_window_from_xyz(file_path):
    """
    Extract the full 3D scan window [xmin, ymin, zmin], [xmax, ymax, zmax]
    from the comment line of an XYZ file.

    Returns:
        xyz_min (np.ndarray): [xmin, ymin, zmin]
        xyz_max (np.ndarray): [xmax, ymax, zmax]
    """
    with open(file_path, 'r') as f:
        lines = f.readlines()

    if len(lines) < 2:
        raise ValueError("XYZ file too short to contain scan window.")

    comment = lines[1].strip()

    match = re.search(r'Scan window: \[\[([^\]]+)\], \[([^\]]+)\]', comment)
    if not match:
        raise ValueError("Scan window not found in comment line.")

    xyz_min = np.array(list(map(float, match.group(1).split())))
    xyz_max = np.array(list(map(float, match.group(2).split())))

    return xyz_min, xyz_max

def draw_unit_cells(ax, origin, cell_vectors, nx, ny, color='black', lw=1.0, ls=':'):
    """
    Draw a grid of unit cells (without overdraw) in the xy plane.

    Parameters:
        ax: matplotlib axis
        origin: 3D vector (usually at bottom-left of grid)
        cell_vectors: 3x3 lattice vectors
        nx, ny: number of repetitions in a and b directions
        color: line color
        lw: line width
        ls: line style
    """

    a_vec = cell_vectors[0]
    b_vec = cell_vectors[1]

    # Draw vertical grid lines (along b direction)
    for i in range(nx + 1):
        start = origin + i * a_vec
        end = start + ny * b_vec
        xs, ys = zip(start[:2], end[:2])
        ax.plot(xs, ys, color=color, lw=lw, linestyle=ls)

    # Draw horizontal grid lines (along a direction)
    for j in range(ny + 1):
        start = origin + j * b_vec
        end = start + nx * a_vec
        xs, ys = zip(start[:2], end[:2])
        ax.plot(xs, ys, color=color, lw=lw, linestyle=ls)

def draw_3d_axis_indicator(ax, anchor=(0.95, 0.05), length=40, style='xy'):
    """
    Draw a small 3D axis compass with visually equal-length arrows (in pixels), 
    independent of figure size or axis aspect ratio.

    Parameters:
        ax: matplotlib axis
        anchor: (x, y) in axes fraction (0–1)
        length: arrow length in pixels
        style: 'xy' or 'x-out'
    """
    import matplotlib.transforms as mtransforms

    fig = ax.figure
    renderer = fig.canvas.get_renderer()

    # Convert anchor to display coords (pixels)
    x0_disp, y0_disp = ax.transAxes.transform(anchor)

    def draw_arrow(dx, dy, label, color, ha, va):
        # Compute end point in display space
        x1_disp = x0_disp + dx
        y1_disp = y0_disp + dy

        # Convert both start and end back to axes coords
        start = ax.transAxes.inverted().transform((x0_disp, y0_disp))
        end = ax.transAxes.inverted().transform((x1_disp, y1_disp))

        # Draw arrow and label
        ax.annotate('', xy=end, xytext=start,
                    arrowprops=dict(arrowstyle='->', lw=1.5, color=color),
                    xycoords='axes fraction')
        ax.text(end[0], end[1], label, fontsize=12, color=color,
                ha=ha, va=va, transform=ax.transAxes)

    if style == 'xy':
        draw_arrow(length, 0, 'x', 'red', ha='left', va='center')
        draw_arrow(0, length, 'y', 'green', ha='center', va='bottom')
        ax.text(anchor[0], anchor[1], 'z', fontsize=12, color='blue',
                ha='center', va='center', transform=ax.transAxes)

    elif style == 'x-out':
        draw_arrow(length, 0, 'y', 'green', ha='left', va='center')
        draw_arrow(0, length, 'z', 'blue', ha='center', va='bottom')
        ax.text(anchor[0], anchor[1], 'x', fontsize=12, color='red',
                ha='center', va='center', transform=ax.transAxes)

    else:
        raise ValueError(f"Unsupported style '{style}'. Use 'xy' or 'x-out'.")