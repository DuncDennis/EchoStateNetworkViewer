"""
Take df with only one Parameter column.
Create violin plot of valid times.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import os
from PIL import Image

# PATH DEFAULTS:
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PLOT_BASE_FOLDER_PATH = os.path.join(DIR_PATH, "..", "pub_plots_results", "m_vs_m_scatter_plot")


# PARAMETER TRANSFORMATIONS:
# here as HTML (as plotly is not so good yet).
PARAM_TRANSFORM = {
    "P node_bias_scale": r"\text{Node bias scale } \sigma_\text{b}$",
    "P t_train": r"\text{Train size } N_\text{T}$",
    # "P r_dim": r"\text{Reservoir dimension } r_\text{dim}$",
    "P r_dim": "<i>r</i><sub>dim</sub>",
    "P reg_param": "<i>β</i>",
    "P w_in_scale": "<i>σ</i>",
    "P n_avg_deg": "d",
    "P n_rad": "<i>ρ</i><sub>0</sub>",
    "P dt": r"\text{Time step of system } \Delta t$",
    "P x_train_noise_scale": r"\text{Train noise scale } \sigma_\text{T}$",
    "P rr_type": r"\text{ridge regression type}$",
    "P r_to_rgen_opt": r"\text{Readout function } \Psi$",
}

# METRIC TRANSFORMATIONS:
METRIC_TRANSFORM = {
    # "M TRAIN PCMAX": r"\text{max pc } i_\text{max}$",
    # "M TRAIN PCMAX": r"\text{PC cutoff }\; i_\text{max}$",
    "M TRAIN PCMAX": r" i_\text{max}$",
    "M VALIDATE VT": r" t_\text{v} \lambda_\text{max}$"
}

# Log x-axis for some parameters:
LOG_COLOR_PARAMS = [
    "P reg_param",
    "P n_rad",
    "P n_avg_deg"
]

# If log: exponent format
EXPONENT_FORMAT = "power"

# width and height of figure:
WIDTH = 600
HEIGHT = int(0.7*WIDTH)

# Template and fonts:
TEMPLATE = "simple_white"
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 25

# Latex text size:
LATEX_TEXT_SIZE = "large"  # normalsize, large, Large, huge

# Traces:
MARKERLINEWIDTH = 1
MARKERLINECOLOR = 'DarkSlateGrey'
MARKERSIZE = 12
LINECOLOR = "Black"
ERROR_THICKNESS = 1


# Grid settings:
X_GRID_SETTINGS = {
    "showgrid": True,
    "gridwidth": 1,
    "gridcolor": 'rgba(0,0,0,0.2)',
}

Y_GRID_SETTINGS = {
    "showgrid": True,
    "gridwidth": 1,
    "gridcolor": 'rgba(0,0,0,0.2)',
}


# Margin dict:
MARGIN_DICT = dict(l=20, r=20, t=20, b=20)
# MARGIN_DICT = dict()

# Y-axis range and ticks:
Y_AXIS_DICT = dict(
    range=[0, 9],
    # tick0 = 0,
    # dtick = 2,
)

X_AXIS_DICT = dict(
    range=[0, 520],
    # tick0 = 0,
    # dtick = 5,
)


def m_vs_m_scatter_plot(df: pd.DataFrame,
                        x_metric: str,
                        y_metric: str,
                        color_param: str,
                        name: str,
                        save_bool: bool = False,
                        color_dtick: int | None = None ):
    """SCATTER PLOT METRIC VS METRIC, PARAM AS COLOR.

    Args:
        df: Dataframe with the x_metric, y_metric, and color_param column.
        name: Output name of the file.
        save_bool: If false only save temporary image. If true save to <name>.
    """


    if color_param in LOG_COLOR_PARAMS:
        df[color_param] = np.log10(df[color_param])
        tick_prefix = "10<sup>"
        tick_suffix = "</sup>"
    else:
        tick_prefix = ""
        tick_suffix = ""
    # Basic figure:
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(x=df[x_metric + "|" + "avg"],
                   y=df[y_metric + "|" + "avg"],
                   error_y=dict(
                       array=df[y_metric + "|" + "error_high"],
                       arrayminus=df[y_metric + "|" + "error_low"],
                       thickness=ERROR_THICKNESS
                   ),
                   error_x=dict(
                       array=df[x_metric + "|" + "error_high"],
                       arrayminus=df[x_metric + "|" + "error_low"],
                       thickness=ERROR_THICKNESS
                   ),
                   mode="markers",
                   marker=dict(color=df[color_param],

                               colorbar=dict(
                                   len=0.95,
                                   # borderwidth=5,
                                   # outlinewidth=2,
                                   tickprefix=tick_prefix,
                                   ticksuffix=tick_suffix,
                                   dtick=color_dtick,
                                   ticks="outside", # "outside"
                                   title=PARAM_TRANSFORM[color_param],
                                   orientation="v",
                                   # side="right"
                               ),
                               colorscale="portland",
                               size=MARKERSIZE,
                               line=dict(width=MARKERLINEWIDTH,
                                         color=MARKERLINECOLOR))
                   ),
    )

    xaxis_title = rf"$\{LATEX_TEXT_SIZE}" + METRIC_TRANSFORM[x_metric]
    yaxis_title = rf"$\{LATEX_TEXT_SIZE}" + METRIC_TRANSFORM[y_metric]

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    # layout:
    fig.update_layout(
        width=WIDTH,
        height=HEIGHT,
        template=TEMPLATE,
        font=dict(
            size=FONT_SIZE,
            family=FONT_FAMILY
        ),
        margin=MARGIN_DICT,
        showlegend=False
    )

    fig.update_yaxes(**Y_GRID_SETTINGS)
    fig.update_xaxes(**X_GRID_SETTINGS)

    fig.update_yaxes(**Y_AXIS_DICT)
    fig.update_xaxes(**X_AXIS_DICT)

    if save_bool:
        file_name = name + ".png"
        total_file_path = os.path.join(PLOT_BASE_FOLDER_PATH, file_name)
    else:
        total_file_path = os.path.join(DIR_PATH, "tmp", "temp_preview_plot.png")

    fig.write_image(total_file_path, scale=3)
    image = Image.open(total_file_path)

    return image, total_file_path




