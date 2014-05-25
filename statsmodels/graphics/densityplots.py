'''
Density plots
'''

import numpy as np
import matplotlib as plt
from statsmodels.graphics import utils
from statsmodels.distributions.empirical_distribution import ECDF

__all__ = ['plot_density', 'plot_rugplot']


def add_rugplot(dens, ax, **kwargs):
    """
    Add rugplot from density object.

    Parameters
    ----------
    dens : mensity object
    ax : matplotlib Axes instance
        The Axes to which to add the plot
    kwargs: addition arguments to pass to plot
    
    Returns
    -------
    fig : matplotlib Figure instance
        The figure that holds the instance.
    """
    y = dens.endog
    ylim = ax.get_ylim()
    yrange = ylim[1] - ylim[0]
    ax.plot(y, np.zeros(y.shape), 'b+', ms=48 * yrange, **kwargs)
    return ax.figure


def add_ecdf(dens, ax, **kwargs):
    """
    Add ecdf from density objects endog.

    Parameters
    ----------
    dens : mensity object
    ax : matplotlib Axes instance
        The Axes to which to add the plot
    kwargs: addition arguments to pass to plot
    
    Returns
    -------
    fig : matplotlib Figure instance
        The figure that holds the instance.
    """
    x = dens.endog
    ecdf = ECDF(x)
    ax.plot(ecdf.x, ecdf.y, **kwargs)
    return ax.figure


def plot_density(dens, ptype='pdf', ax=None, **kwargs):
    """Plot density on precomputed support grid.

    Parameters
    ----------
    dens : density object
    ax : Matplotlib AxesSubplot instance, optional
        If given, this subplot is used to plot in instead of a new figure being
        created.
    kwargs
        The keyword arguments are passed to the plot command for the fitted
        values points.

    Returns
    -------
    fig : Matplotlib figure instance
        If `ax` is None, the created figure.  Otherwise the figure to which
        `ax` is connected.

    """

    fig, ax = utils.create_mpl_ax(ax)

    if ptype == 'pdf':
        y = dens.density
    elif ptype == 'cdf':
        y = dens.cdf_values
    else:
        return fig

    x = dens.support
    ax.plot(x, y, **kwargs)

    return fig