
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
from typing import Any
from .utils import indexable

def NewFigure(
        rows  :int = 1,
        cols  :int = 1,
        scale :float = 1.0,
        aspect_ratio :float = 1.0
    ):
    fig, axs = plt.subplots(
            nrows = rows, ncols = cols,
            # sharey='row',
            figsize=(4.0*scale*cols/aspect_ratio, 4.0*scale*rows)
    )
    return fig, axs

def AxisLabels(
        xlabel :str = 'linear',
        ylabel :str = 'linear',
        panel  :Any = None,
        fontsize : float = 15
    ):
    if panel is None:
        plt.xlabel(xlabel, fontsize = fontsize)
        plt.ylabel(ylabel, fontsize = fontsize)
        return
    panel.set_xlabel(xlabel, fontsize = fontsize)
    panel.set_ylabel(ylabel, fontsize = fontsize)
    
def AxisScales(
        xscale :str = 'linear',
        yscale :str = 'linear',
        panel  :Any = None
    ):
    if panel is None:
        plt.xscale(xscale)
        plt.yscale(yscale)
        return
    panel.set_xscale(xscale)
    panel.set_yscale(yscale)

def AxisLogs(panel :Any = None):
    AxisScales('log', 'log', panel)

def Tight():
    plt.tight_layout()

def CustomLegend(*elements :dict, panel :Any = None, **kwargs):
    """Provides custom line style objects to define the legend elements"""
    lines = [ Line2D([0], [0], **e) for e in elements]
    if panel is None: panel = plt
    if 'framealpha' not in kwargs: kwargs['framealpha'] = 1.0
    if 'shadow' not in kwargs: kwargs['shadow'] = False
    panel.legend(handles = lines, **kwargs)

def XTicks(ticks: list[float] = [], panel :Any = None, fontsize :float = np.nan):
    """Sets the X axis tick labels"""
    if panel is None: panel = plt.gca()
    if ticks:
        panel.set_xticks(ticks)
        panel.get_xaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    if ~np.isnan(fontsize):
        panel.tick_params(axis='x', which='both', labelsize=fontsize)

def YTicks(ticks: list[float] = [], panel :Any = None, fontsize :float = np.nan):
    """Sets the Y axis tick labels"""
    if panel is None: panel = plt.gca()
    if ticks:
        panel.set_yticks(ticks)
        panel.get_yaxis().set_major_formatter(mpl.ticker.ScalarFormatter())
    if ~np.isnan(fontsize):
        panel.tick_params(axis='y', which='both', labelsize=fontsize)

def StylePanels(panels :list|Any = None, fontsize :int = 15):
    """Applies a custom figure style"""
    if panels is None:
        panels = plt.gca()
    if not indexable(panels):
        panels = [panels]
    for p in panels:
        p.tick_params(
            axis='both', which='both',
            direction = 'in',
            labelsize = fontsize, 
            width=1.0, length=7.0,
            top=True, bottom=True, right = True
        )
