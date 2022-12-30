"""
1. Take the aggregated sweep plot data from results_viewer_v2 which have the following columns:
- one or two parameter columns of the type "P <param_name>"
- "avg", "error_high" and "error_low" for the measurements for that parameter point.

2. Plot them nicely. For each class of plots there will be one function.
"""

import pandas as pd
import plotly.graph_objects as go
import os
from PIL import Image

# PATH DEFAULTS:
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PLOT_BASE_FOLDER_PATH = os.path.join(DIR_PATH, "..", "pub_plots_results", "oned_valid_time_sweep")


# PARAMETER TRANSFORMATIONS:
PARAM_TRANSFORM = {
    "P node_bias_scale": r"\text{Node bias scale } \sigma_\text{b}$",
    "P t_train": r"\text{Train size } N_\text{T}$",
    "P r_dim": r"\text{Reservoir dimension } r_\text{dim}$",
    "P reg_param": r"\text{Regularization parameter } \beta$",
    "P w_in_scale": r"\text{Input strength } \sigma$",
    "P n_avg_deg": r"\text{Average degree } d$",
    "P n_rad": r"\text{Spectral radius } \rho_0$",
    "P dt": r"\text{Time step of system } \Delta t$",
    "P x_train_noise_scale": r"\text{Train noise scale } \sigma_\text{T}$",
}


# DEFAULT PARAMETERS: 
PARAM_DEFAULTS = {
    "P node_bias_scale": 0.4,
    "P t_train": 1000,
    "P r_dim": 500,
    "P reg_param": 1e-7,
    "P w_in_scale": 1.0,
    "P n_avg_deg": 5.0,
    "P n_rad": 0.4,
    "P dt": 0.1,
    "P x_train_noise_scale": 0.0,
}

# Log x-axis for some parameters:
LOG_X_PARAMS = [
    "P reg_param",
]

# If log: exponent format
EXPONENT_FORMAT = "power"  # e, power

# width and height of figure:
HEIGHT = 350
WIDTH = int(2.1 * HEIGHT)

# Template and fonts:
TEMPLATE = "simple_white"
FONT_FAMILY = "Times New Roman"
FONT_SIZE = 25

# Latex text size:
LATEX_TEXT_SIZE = "large"  # normalsize, large, Large, huge

# Traces:
LINEWIDTH = 3
LINECOLOR = "Black"

# Errorbar:
ERRORWIDTH = 8
ERRORLINEWIDTH = 2

# Grid settings:
GRID_SETTINGS = {
    "showgrid": True,
    "gridwidth": 1,
    "gridcolor": "gray"
}

# Margin dict:
MARGIN_DICT = dict(l=5, r=5, t=5, b=5)

# Default line:
DEFAULT_LINE_DICT = dict(
    line_width=5,
    line_dash="dash",
    line_color="green",
    opacity=0.6
    )

# Y-axis range and ticks:
Y_AXIS_DICT = dict(
    range=[-0.5, 8.5],
    tick0 = 0,
    dtick = 2,
)


def onedim_vt(df: pd.DataFrame,
              name: str,
              save_bool: bool = False):
    """Take the df with only one parameter column and plot the valid time vs parameter value.
    add a vertical line for the default parameter value.

    Args:
        df: Dataframe with only one P column.
        name: Output name of the file.
        save_bool: If false only save temporary image. If true save to <name>.
    """
    # get parameter columns (should be only one):
    parameter_cols = [x for x in df.columns if x.startswith("P ")]

    # check if it is only one:
    if len(parameter_cols) != 1:
        raise ValueError("onedim_vt only works when there is exactly one param column. ")

    # get name of parameter column
    p_name = parameter_cols[0]

    # Basic figure:
    fig = go.Figure()

    # Add trace:
    fig.add_trace(
        go.Scatter(
            x=df[p_name],
            y=df["avg"],
            error_y={"array": df["error_high"],
                     "arrayminus": df["error_low"],
                     "width": ERRORWIDTH,
                     "thickness": ERRORLINEWIDTH},
            line=dict(
                width=LINEWIDTH,
                color=LINECOLOR
            )
        )
    )

    # Add vertical default line:
    if p_name in PARAM_DEFAULTS.keys():
        default_x = PARAM_DEFAULTS[p_name]
        fig.add_vline(
            x=default_x,
            **DEFAULT_LINE_DICT
        )

    # logarithmic x-axis
    if p_name in LOG_X_PARAMS:
        fig.update_xaxes(type="log",
                         exponentformat=EXPONENT_FORMAT)

    if p_name in PARAM_TRANSFORM.keys():
        xaxis_title = PARAM_TRANSFORM[p_name]
        xaxis_title = rf"$\{LATEX_TEXT_SIZE}" + xaxis_title
    else:
        xaxis_title = str(p_name)

    yaxis_title = r" t_\text{v} \lambda_\text{max}$"
    yaxis_title = rf"$\{LATEX_TEXT_SIZE}" + yaxis_title

    fig.update_layout(
        xaxis_title=xaxis_title,
        yaxis_title=yaxis_title,
    )

    # layout:
    fig.update_layout(
        template=TEMPLATE,
        font=dict(
            size=FONT_SIZE,
            family=FONT_FAMILY
        ),
        margin=MARGIN_DICT
    )

    fig.update_yaxes(**GRID_SETTINGS)
    fig.update_yaxes(**Y_AXIS_DICT)

    if save_bool:
        file_name = name + ".png"
        total_file_path = os.path.join(PLOT_BASE_FOLDER_PATH, file_name)
    else:
        total_file_path = os.path.join(DIR_PATH, "tmp", "temp_preview_plot.png")

    fig.write_image(total_file_path, scale=3)
    image = Image.open(total_file_path)

    return image, total_file_path




