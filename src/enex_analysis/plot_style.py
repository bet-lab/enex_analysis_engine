"""Plot styling helpers for enex_analysis.

Expose fs, pad, LW and a function apply_plot_style() that applies rcParams and returns
a small helper dict. Import this module in plotting scripts to keep font sizes
consistent across the project.

Usage:
    from enex_analysis.plot_style import fs, pad, LW, apply_plot_style
    fig = plt.figure(figsize=(6,4))
    apply_plot_style()
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import dm

# Base rc font size
plt.rcParams['font.size'] = 9

fs = {
    'label': dm.fs(0),
    'tick': dm.fs(-1.5),
    'legend': dm.fs(-2.0),
    'subtitle': dm.fs(-0.5),
    'cbar_tick': dm.fs(-2.0),
    'cbar_label': dm.fs(0),
    'cbar_title': dm.fs(-1),
    'setpoint': dm.fs(-1),
    'text': dm.fs(-3.0),
}

pad = {
    'label': 6,
    'tick': 5,
}


def apply_plot_style(rc_update: dict | None = None) -> dict:
    """Apply the project plot style to matplotlib rcParams.

    Parameters
    ----------
    rc_update : dict, optional
        Extra rcParams to update after the base style is applied.

    Returns
    -------
    dict
        The final rcParams that were applied (subset: font sizes and line widths).
    """
    base = {
        'font.size': 9,
        'axes.titlesize': fs['subtitle'],
        'axes.labelsize': fs['label'],
        'xtick.labelsize': fs['tick'],
        'ytick.labelsize': fs['tick'],
        'legend.fontsize': fs['legend'],
        'figure.titlesize': fs['subtitle'],
    }
    if rc_update:
        base.update(rc_update)

    plt.rcParams.update(base)
    return base


# ---------------------------------------------------------------------------
# Auto-apply option
# If the environment variable ENEX_PLOT_AUTOAPPLY is set to '1' (default), the
# full style will be applied at import time so users only need to import the
# module to enable the project's plotting style. Set to '0' to disable.
import os

_AUTO_APPLY = os.getenv('ENEX_PLOT_AUTOAPPLY', '1') == '1'

if _AUTO_APPLY:
    # Apply style on import
    apply_plot_style()