import numpy as np
from scipy.stats import gaussian_kde

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
