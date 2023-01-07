"""
1. Take the aggregated sweep plot data from results_viewer_v2 which have the following columns:
- one or two parameter columns of the type "P <param_name>"
- "avg", "error_high" and "error_low" for the measurements for that parameter point.

2. Plot them nicely. For each class of plots there will be one function.
"""
from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go
import os
from PIL import Image

# PATH DEFAULTS:
DIR_PATH = os.path.dirname(os.path.realpath(__file__))
PLOT_BASE_FOLDER_PATH = os.path.join(DIR_PATH, "..", "pub_plots_results", "twod_valid_time_sweep")


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
    "P model_error_eps": r"\text{model error }\epsilon$",
    "P predictor_type": r"\text{hybrid type}$",
    "P system": r"\text{system}$"
}



# Log x-axis for some parameters:
LOG_X_PARAMS = [
    "P reg_param",
    "P model_error_eps"
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


# Y-axis range and ticks:
Y_AXIS_DICT = dict(
    # range=[-0.5, 8.5],
    tick0 = 0,
    dtick = 5,
)

# LEGEND:
SHOWLEGEND = True
LEGENDDICT = dict(
    orientation="h",
    yanchor="bottom",
    y=1.01,  # 0.99
    xanchor="left",
    # x=0.01,
    font=dict(size=20),
    bordercolor="grey",
    borderwidth=2,
)

# ORDER AND NAMES for HYBRID:
HYBRIDORDER = {"only reservoir": 1,
               "input hybrid": 2,
               "output hybrid": 3,
               "full hybrid": 4,
               "only model": 5}

HYBRIDNAMES = {"no_hybrid": "only reservoir",
               "input_hybrid": "input hybrid",
               "output_hybrid": "output hybrid",
               "full_hybrid": "full hybrid",
               "model_predictor": "only model"}

# NAMES FOR PCA BOOL:
PCANAMES = {True: "PC-transform",
            False: "no PC-transform"}

def twodim_vt(df: pd.DataFrame,
              x_param: str,
              name: str,
              save_bool: bool = False,
              plot_args: dict | None = None):
    """Take the df with two parameter columns and plot the valid time vs parameter value and color.

    Add a vertical line for the default parameter value.

    Args:
        df: Dataframe with exactly 2 P column.
        name: Output name of the file.
        save_bool: If false only save temporary image. If true save to <name>.
        plot_args: Additional plot args.
    """
    # get parameter columns (should be only one):
    parameter_cols = [x for x in df.columns if x.startswith("P ")]

    # check if it is only one:
    if len(parameter_cols) != 2:
        raise ValueError("twodim_vt only works when there are 2 param columns. ")

    other_param = parameter_cols.copy()
    other_param.remove(x_param)
    other_param = other_param[0]

    # if hybrid types rename the values of P predictor_type:
    if "P predictor_type" in parameter_cols: # can be x_param or other_param.
        df["P predictor_type"] = df["P predictor_type"].apply(lambda x: HYBRIDNAMES[x])

    if "P perform_pca_bool" in parameter_cols:
        df["P perform_pca_bool"] = df["P perform_pca_bool"].apply(lambda x: PCANAMES[x])

    # Basic figure:
    fig = go.Figure()

    # Add trace for each unique value of other parameter:
    unique_for_p = pd.Series(df[other_param].value_counts().index)

    # Fix order if hybrid:
    if other_param == "P predictor_type":
        try:
            unique_for_p.sort_values(key=lambda series: series.apply(lambda x: HYBRIDORDER[x]),
                                     inplace=True)
        except:
            print("hybrid sorting error")

    for other_p_val in unique_for_p:
        sub_df = df[df[other_param] == other_p_val]

        # name:
        trace_name = str(other_p_val)

        # Add trace:
        fig.add_trace(
            go.Scatter(
                x=sub_df[x_param],
                y=sub_df["avg"],
                error_y={"array": sub_df["error_high"],
                         "arrayminus": sub_df["error_low"],
                         "width": ERRORWIDTH,
                         "thickness": ERRORLINEWIDTH},
                line=dict(
                    width=LINEWIDTH,
                    # color=LINECOLOR
                ),
                name=trace_name
            )
        )


    # logarithmic x-axis
    if x_param in LOG_X_PARAMS:
        fig.update_xaxes(type="log",
                         exponentformat=EXPONENT_FORMAT,
                         dtick=1)

    xaxis_title = PARAM_TRANSFORM[x_param]
    xaxis_title = rf"$\{LATEX_TEXT_SIZE}" + xaxis_title

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

    fig.update_layout(
        legend=LEGENDDICT,
        showlegend=SHOWLEGEND
    )

    # extras:
    fig.update_layout(**plot_args)

    if save_bool:
        file_name = name + ".png"
        total_file_path = os.path.join(PLOT_BASE_FOLDER_PATH, file_name)
    else:
        total_file_path = os.path.join(DIR_PATH, "tmp", "temp_preview_plot.png")

    fig.write_image(total_file_path, scale=3)
    image = Image.open(total_file_path)

    return image, total_file_path




