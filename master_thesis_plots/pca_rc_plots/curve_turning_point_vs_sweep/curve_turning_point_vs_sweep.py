"""Create an explained variance with error band plot and sweep some parameters.
Idea: Plot the turning point of the fermi-style curve vs. a sweep parameter (probably r_dim).
"""
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import itertools
import plotly.express as px
# from scipy.signal import savgol_filter

col_pal = px.colors.qualitative.Plotly
col_pal_iterator = itertools.cycle(col_pal)


import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities

def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return "rgba" + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))

# Create data:
sys_obj = sims.Lorenz63(dt=0.1)
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 1000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 1000,
                    "n_train_sects": 1,
                    "n_validate_sects": 1,
                    "normalize_and_center": False,
                    }
n_train = ts_creation_args["n_train_sects"]
train_sync_steps = ts_creation_args["t_train_sync"]
train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                         **ts_creation_args)

# Build RC args:
build_args = {
    "x_dim": 3,
    "r_dim": 500,
    "n_rad": 0.4,
    "n_avg_deg": 5.0,
    "n_type_opt": "erdos_renyi",
    "r_to_rgen_opt": "linear_r",
    "act_fct_opt": "tanh",
    "node_bias_opt": "random_bias",
    "node_bias_scale": 0.4,
    "w_in_opt": "random_sparse",
    "w_in_scale": 1.0,
    "x_train_noise_scale": 0.0,
    # "reg_param": 1e-7,
    "reg_param": 1e-20,
    "ridge_regression_opt": "bias",
    "scale_input_bool": True,
}

# Ensemble size:
n_ens = 1

# seeds:
seed = 301
rng = np.random.default_rng(seed)
seeds = rng.integers(0, 10000000, size=n_ens)

legend_orientation = "v"
legend_pos_y = 1.07
legend_pos_x = 0.95
entrywidth=0

# sweep:


# sweep_key = "n_rad"
# sweep_name = r"\rho_0"
# sweep_values = [0.0, 0.1, 0.4, 0.8, 1.2]
# f_name = "spectralradius"
# legend_orientation = "h"
# legend_pos_y = 1.07
# legend_pos_x = 1.01
# entrywidth = 87

sweep_key = "r_dim"
sweep_name = r"r_\text{dim}"
# sweep_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700]
# sweep_values = [50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000]
sweep_values =  [20, 50, 100, 200, 300, 400, 500, 600, 800, 1000, 1500, 2000]
# sweep_values = [100, 200, 300, 400, 500]
# sweep_values = [300, 400, 500]
# sweep_values = [500]
f_name = "rdim"

# sweep_key = "x_train_noise_scale"
# sweep_name = r"\text{Input noise scale}"
# sweep_values = [0.0, 0.001, 0.01, 0.1, 1.0]
# f_name = "noise"

# sweep_key = "act_fct_opt"
# sweep_name = r"\text{Activation fct.}"
# sweep_values = ["tanh", "sigmoid", "relu", "linear"]
# f_name = "actfct"

# sweep_key = "node_bias_scale"
# sweep_name = r"\sigma_\text{B}"
# sweep_values = [0.0, 0.4, 0.8, 1.2, 1.6]
# f_name = "nodebias"

# sweep_key = "w_in_scale"
# sweep_name = r"\sigma"
# # sweep_values = [0.2, 0.6, 1.0, 1.4, 1.8]
# sweep_values = [0.1, 1.0, 2.0, 3.0, 4.0]
# f_name = "winstrength"

# sweep_key = "n_avg_deg"
# sweep_name = r"d"
# sweep_values = [2, 5, 50, 150, 300]
# f_name = "avgdeg"

reg_param = build_args["reg_param"]

sweep_name = r"\large " + sweep_name

for i_sweep, sweep_value in enumerate(sweep_values):
    print(f"{f_name}: {sweep_value}")
    build_args[sweep_key] = sweep_value

    # Do experiment:
    for i_ens in range(n_ens):
        print(i_ens)
        # Build rc:
        esn_obj = esn.ESN()
        with utilities.temp_seed(seeds[i_ens]):
            esn_obj.build(**build_args)
        for i_train in range(n_train):
            # Train RC:
            train_data = train_data_list[i_train]
            # train_data = train_data[:, 0:1]
            _, _, more_out = esn_obj.train(train_data, sync_steps=train_sync_steps, more_out_bool=True)
            res_states = more_out["r"]
            pca = PCA()
            res_pca_states = pca.fit_transform(res_states)
            train_steps = res_pca_states.shape[0]
            # expl_var_ratio = pca.explained_variance_ratio_
            # expl_var_ratio = pca.explained_variance_
            expl_var = np.var(res_pca_states, axis=0)
            n_components = expl_var.size

            y = expl_var / (expl_var + reg_param / train_steps)

            idx = (np.abs(y - 0.5)).argmin()
            pc_index_mid = idx + 1

            if i_train == 0 and i_ens == 0 and i_sweep == 0:
                # explained variances:
                # mid_point_results = np.zeros((n_ens, n_train))
                mid_point_results = np.zeros((n_ens, n_train, len(sweep_values)))
                # expl_var_ratio_results = np.zeros((n_ens, n_train, n_components))

            # explained variances:
            mid_point_results[i_ens, i_train, i_sweep] = pc_index_mid


# Data:
average = np.median(mid_point_results, axis=(0,1))
low = np.min(mid_point_results, axis=(0,1))
high = np.max(mid_point_results, axis=(0,1))

# Plot:
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=sweep_values,
               y=average,
               error_y={"array": high - average,
                        "arrayminus": average - low}
               )
)

# plot params:
yaxis_title = r"$\Large\text{Cutoff}$"
xaxis_title =  "$" + sweep_name + "$"
title = None
linewidth = 3
height = 350
width = int(2.1 * height)

log_x=False
log_y=False
# x_axis = dict(tickmode="linear",
#               tick0=0,
#               dtick=500)
x_axis=None

# y_axis = dict(tickmode="linear",
#               tick0=0,
#               dtick=500)
y_axis=None

font_size = 23
legend_font_size = 23
font_family = "Times New Roman"

fig.update_layout(
    title=title,
    width=width,
    height=height,
    xaxis=x_axis,
    yaxis=y_axis,
    yaxis_title=yaxis_title,
    xaxis_title=xaxis_title,

    font=dict(
        size=font_size,
        family=font_family
    ),
    # legend=dict(
    #     orientation=legend_orientation,
    #     yanchor="top",
    #     y=legend_pos_y,
    #     xanchor="right",
    #     x=legend_pos_x,
    #     font=dict(size=legend_font_size),
    #     entrywidth=entrywidth,
    #     bordercolor="grey",
    #     borderwidth=2,
    #     # tracegroupgap=4,
    #     # entrywidth=2
    #     # title=fr"${sweep_name}$",
    #     # title=r"$test$",
    #     # tracegroupgap = 10
    # )

    # legend=dict(
    #     orientation="h",
    #     yanchor="bottom",
    #     y=1.01,  # 0.99
    #     xanchor="left",
    #     # x=0.01,
    #     font=dict(size=legend_font_size))

    )

fig.update_traces(line=dict(color="Black", width=linewidth))

if log_x:
    fig.update_layout(
        xaxis={
            'exponentformat': 'E'}
    )
    fig.update_xaxes(type="log")

if log_y:
    fig.update_layout(
        yaxis={
            'exponentformat': 'E'}
    )
    fig.update_yaxes(type="log")

fig.update_layout(template="simple_white",
                  showlegend=False,
                  )

# fig.update_xaxes(
#     showgrid=True,
#     gridwidth=1,
#     gridcolor="gray",
# )
fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor="gray",
)

fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)

# SAVE
# fig.write_image("intro_expl_var_w_error.pdf", scale=3)
fig.write_image(f"curve_turning_point_vs_sweep__{f_name}.png", scale=3)

