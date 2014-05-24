'''
Density plots
'''

import numpy as np
from statsmodels.graphics import utils

__all__ = ['plot_density']


def plot_density(dens, ax=None, **kwargs):
    """Plot density on precomputed support grid.

    Parameters
    ----------
    dens : density container instance
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

    y = dens.density
    x = dens.support

    ax.plot(x, y, **kwargs)

    return fig