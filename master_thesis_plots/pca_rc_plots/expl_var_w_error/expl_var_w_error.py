"""Create Explained variance of pca states plot with error band.

NOT USED.
"""
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities

# Create data:
sys_obj = sims.Lorenz63()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 5000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 400,
                    "n_train_sects": 1,
                    "n_validate_sects": 1,
                    "normalize_and_center": True,
                    }
n_train = ts_creation_args["n_train_sects"]
train_sync_steps = ts_creation_args["t_train_sync"]
train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                         **ts_creation_args)

# Build RC args:
build_args = {
    "x_dim": 3,
    "r_dim": 100,
    "n_rad": 0.4,
    "n_avg_deg": 3.0,
    "n_type_opt": "erdos_renyi",
    "r_to_rgen_opt": "linear_r",
    "act_fct_opt": "tanh",
    "node_bias_opt": "random_bias",
    "node_bias_scale": 0.1,
    "w_in_opt": "random_sparse",
    "w_in_scale": 1.0,
    "x_train_noise_scale": 0.0,
    "reg_param": 1e-7,
    "ridge_regression_opt": "no_bias",
    "scale_input_bool": False,
}

# Ensemble size:
n_ens = 5

# seeds:
seed = 1
rng = np.random.default_rng(seed)
seeds = rng.integers(0, 10000000, size=n_ens)


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
        _, _, more_out = esn_obj.train(train_data, sync_steps=train_sync_steps, more_out_bool=True)
        res_states = more_out["r"]
        pca = PCA()
        res_pca_states = pca.fit_transform(res_states)
        expl_var_ratio = pca.explained_variance_ratio_
        if i_train == 0 and i_ens == 0:
            # explained variances:
            n_components = expl_var_ratio.size
            expl_var_ratio_results = np.zeros((n_ens, n_train, n_components))

            # whole reservoir states:
            # train_steps, r_dim = res_states.shape
            # results = np.zeros((n_ens, n_train, r_dim, train_steps, r_dim))

        # explained variances:
        expl_var_ratio_results[i_ens, i_train, :] = expl_var_ratio

        # whole resrvoir states:
        # results[i_ens, i_train, :, :] = res_states

# Plot:
x = np.arange(0, n_components)
y = np.median(expl_var_ratio_results, axis=(0, 1))
y_low = np.min(expl_var_ratio_results, axis=(0, 1))
y_high = np.max(expl_var_ratio_results, axis=(0, 1))
print(x.shape, y.shape, y_low.shape, y_high.shape)

x = x.tolist()
y = y.tolist()
y_low = y_low.tolist()
y_high = y_high.tolist()


# plot params:
yaxis_title = r"$\text{Explained Variance Ratio } \lambda_i$"
xaxis_title =  r'$\text{Principal Component } \boldsymbol{p}_i$'
title = None
height = 500
width = int(1.4 * height)

log_x=False
log_y=True


x_axis = dict(tickmode="linear",
              tick0=0,
              dtick=500)
x_axis=None

y_axis = dict(tickmode="linear",
              tick0=0,
              dtick=500)
y_axis=None

xrange = [-5, 105]
yrange = None  # [0, 15]
font_size = 15
legend_font_size = 11
font_family = "Times New Roman"

fig = go.Figure()
fig.add_trace(
    go.Scatter(x=x, y=y, showlegend=False,
               line=dict(color='rgba(0,100,80,1.0)'))
)

fig.add_trace(
    go.Scatter(
        x=x+x[::-1],  # x, then x reversed
        y=y_high + y_low[::-1],  # upper, then lower reversed
        fill='toself',
        fillcolor='rgba(0,100,80,0.2)',
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    )
)

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

    legend=dict(
        orientation="h",
        yanchor="bottom",
        y=1.01,  # 0.99
        xanchor="left",
        # x=0.01,
        font=dict(size=legend_font_size))

    )

fig.update_yaxes(range=yrange)
fig.update_xaxes(range=xrange)

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

fig.update_layout(template="plotly_white",
                  showlegend=False,
                  )

fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)

# SAVE
# fig.write_image("intro_expl_var_w_error.pdf", scale=3)
fig.write_image("expl_var_w_error.png", scale=3)

