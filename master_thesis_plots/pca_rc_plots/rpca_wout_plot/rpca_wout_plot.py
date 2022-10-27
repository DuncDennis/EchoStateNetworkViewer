"""Create Explained variance of pca states plot with error band.

NOT USED.
"""
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities

import itertools
import plotly.express as px
col_pal = px.colors.qualitative.Plotly
col_pal_iterator = itertools.cycle(col_pal)

def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return "rgba" + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))

# Create data:
sys_obj = sims.Lorenz63()
# sys_obj = sims.Logistic()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 5000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 1000,
                    "n_train_sects": 5,
                    "n_validate_sects": 1,
                    "normalize_and_center": True,
                    }
n_train = ts_creation_args["n_train_sects"]
train_sync_steps = ts_creation_args["t_train_sync"]
pred_sync_steps = ts_creation_args["t_validate_sync"]
train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                         **ts_creation_args)

# Build RC args:
build_args = {
    "x_dim": 3,
    "r_dim": 500,
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
    # "ridge_regression_opt": "no_bias",
    "ridge_regression_opt": "bias",
    "scale_input_bool": False,
    "perform_pca_bool": True
}

# Ensemble size:
n_ens = 10

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
        res_states = more_out["rgen"]
        pca = PCA()
        res_pca_states = pca.fit_transform(res_states)
        expl_var_ratio = pca.explained_variance_ratio_
        # expl_var_ratio = pca.explained_variance
        w_out = esn_obj.get_w_out()

        # Prediction:
        validate_data = validate_data_list_of_lists[i_train][0]
        _, _, more_out_pred = esn_obj.predict(validate_data,
                                              sync_steps=pred_sync_steps,
                                              more_out_bool=True)

        rgen_pred = more_out_pred["rgen"]
        rgen_pca_pred = pca.transform(rgen_pred)


        if i_train == 0 and i_ens == 0:
            # explained variances:
            n_components = expl_var_ratio.size
            expl_var_ratio_results = np.zeros((n_ens, n_train, n_components))
            rpca_train_std_results = np.zeros((n_ens, n_train, n_components))
            rpca_pred_std_results = np.zeros((n_ens, n_train, n_components))
            w_out_pca_results = np.zeros((n_ens,
                                          n_train,
                                          w_out.shape[0],
                                          w_out.shape[1]))

        # explained variances:
        expl_var_ratio_results[i_ens, i_train, :] = expl_var_ratio

        if "perform_pca_bool" in build_args:
            if build_args["perform_pca_bool"]:
                w_out_pca = w_out
        else:
            w_out_pca = w_out @ pca.components_.T
        w_out_pca_results[i_ens, i_train, :, :] = w_out_pca

        # Rpca std for train and predict:
        rpca_train_std_results[i_ens, i_train, :] = np.std(res_pca_states, axis=0)
        rpca_pred_std_results[i_ens, i_train, :] = np.std(rgen_pca_pred, axis=0)

        # whole resrvoir states:
        # results[i_ens, i_train, :, :] = res_states


abs_w_out = np.abs(w_out_pca_results)
mean_abs_w_out = np.median(abs_w_out, axis=(0, 1))
error_low_w_out = np.quantile(abs_w_out, q=0.25, axis=(0, 1))
error_high_w_out = np.quantile(abs_w_out, q=0.75, axis=(0, 1))

mean_rpca_train_std = np.median(rpca_train_std_results, axis=(0, 1))
mean_rpca_pred_std = np.median(rpca_pred_std_results, axis=(0, 1))
low_rpca_train_std = np.quantile(rpca_train_std_results, q=0.25, axis=(0, 1))
high_rpca_train_std = np.quantile(rpca_train_std_results, q=0.75, axis=(0, 1))
low_rpca_pred_std = np.quantile(rpca_pred_std_results, q=0.25, axis=(0, 1))
high_rpca_pred_std = np.quantile(rpca_pred_std_results, q=0.75, axis=(0, 1))

# fig = go.Figure()
# for i in range(esn_obj.y_dim):
#     fig.add_trace(
#         go.Scatter(y=mean_abs_w_out[i, :], mode= 'none',
#                    stackgroup='one')
#     )
#
# # SAVE
# fig.write_image("rpca_wout_plot.png", scale=3)

# Plot:

# plot params:
yaxis_title = r"$|w_\text{out, pca}[j, i]|$"
xaxis_title =  r'$\text{Principal Component } \boldsymbol{p}_i$'
title = None
height = 400
width = int(1.4 * height)

log_x=False
log_y=False

x_axis = dict(tickmode="linear",
              tick0=0,
              dtick=100)

# y_axis = dict(tickmode="linear",
#               tick0=0,
#               dtick=500)
y_axis=None

# xrange = [-5, 105]
xrange = None
yrange = None  # [0, 15]
font_size = 15
legend_font_size = 11
font_family = "Times New Roman"

fig = go.Figure()
# fig = make_subplots(specs=[[{"secondary_y": True}]])
for i in range(esn_obj.y_dim):
    fig.add_trace(
        go.Scatter(y=mean_abs_w_out[i, :],
                   mode='none',
                   stackgroup='one',
                   name=fr"$j = {i+1}$",
                   )
    )

#train rpca std
data_list = ((mean_rpca_train_std, low_rpca_train_std, high_rpca_train_std, "train"),
             (mean_rpca_pred_std, low_rpca_pred_std, high_rpca_pred_std, "predict"))
for data in data_list:
    mean, low, high, name = data
    color = next(col_pal_iterator)
    color_fade = hex_to_rgba(color, 0.8)
    fig.add_trace(
        go.Scatter(y=mean,
                   showlegend=True,
                   line=dict(color=color),
                   name=name,
                   yaxis="y2"),
        # secondary_y=True
    )
    x = np.arange(n_components).tolist()
    high = high.tolist()
    low = low.tolist()
    fig.add_trace(
        go.Scatter(x=x + x[::-1], y=high + low[::-1],
                   line=dict(color='rgba(255,255,255,0)'),
                   fill="toself",
                   fillcolor=color_fade,
                   showlegend=False,
                   yaxis="y2"),
        # secondary_y=True
    )

y = mean_rpca_train_std / (mean_rpca_train_std + build_args["reg_param"])
name = r"$d^2 / (d^2 + \lambda)$"
fig.add_trace(
    go.Scatter(x=x,
               y=y,
               name=name,
               line=dict(color="black", dash="dot"),
               showlegend=True,
               yaxis="y3"),
    # secondary_y=True,
)

fig.update_layout(
    yaxis2=dict(
        title=r"$\text{Std of rpca (train and predict)}$",
        type="log",
        exponentformat='E',
        overlaying="y",
        side="right"
    ),
    yaxis3=dict(visible=False,
                overlaying="y")
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
        # orientation="h",
        yanchor="bottom",
        # y=1.01,  # 0.99
        y=0.65,  # 0.99
        xanchor="left",
        x=0.69,
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
                  showlegend=True,
                  )

fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)



# SAVE
# fig.write_image("intro_expl_var_w_error.pdf", scale=3)
fig.write_image("rpca_wout_plot.png", scale=3)

#### SUBPLOTS WITH ERROR:

height = 500
width = int(1.4 * height)
font_size = 15
legend_font_size = 15
font_family = "Times New Roman"
linewidth = 0.6

yaxis_title = r"$r_\text{pca, i}$"
xaxis_title =  r"$|w_\text{out, pca}[j, i]|$"

nr_out = esn_obj.y_dim
fig = make_subplots(rows=nr_out, cols=1,
                    # shared_yaxes=True,
                    shared_xaxes=True,
                    # vertical_spacing=True,
                    # print_grid=True,
                    x_title=xaxis_title,
                    y_title=yaxis_title,
                    # row_heights=[height] * nr_taus,
                    # column_widths=[width],
                    # subplot_titles=[fr"$i = {1+x}$" for x in range(nr_out)],
                    # horizontal_spacing=1,
                    # row_titles=[fr"$i = {x}$" for x in dimensions],
                    )

x = np.arange(w_out.shape[1]).tolist()
for i in range(nr_out):
    color = next(col_pal_iterator)
    fig.add_trace(
        go.Scatter(x=x, y=mean_abs_w_out[i, :],
        line=dict(width=linewidth,
                  color=color),
        name=f"{i}", showlegend=True),
        row=i+1, col=1
    )
    low = error_low_w_out[i, :].tolist()
    high = error_high_w_out[i, :].tolist()
    fig.add_trace(
        go.Scatter(x=x + x[::-1], y=high + low[::-1],
                   line=dict(color='rgba(255,255,255,0)'),
                   fill="toself",
                   fillcolor=hex_to_rgba(color, 0.5),
                   showlegend=False),
        row=i + 1, col=1
    )
# fig.update_layout(
#     title=title,
#     width=width,
#     height=height,
#     xaxis=x_axis,
#     yaxis=y_axis,
#     yaxis_title=yaxis_title,
#     xaxis_title=xaxis_title,
#
#     font=dict(
#         size=font_size,
#         family=font_family
#     ),
#
#     legend=dict(
#         # orientation="h",
#         yanchor="top",
#         y=1.01,
#         xanchor="right",
#         x=0.95,
#         font=dict(size=legend_font_size)
#     )
#     )
#
# # fig.update_yaxes(range=yrange)
# # fig.update_xaxes(range=xrange)
#
# if log_x:
#     fig.update_layout(
#         xaxis={
#             'exponentformat': 'E'}
#     )
#     fig.update_xaxes(type="log")
#
# if log_y:
#     fig.update_layout(
#         yaxis={
#             'exponentformat': 'E'}
#     )
#     fig.update_yaxes(type="log")
#
# fig.update_layout(template="plotly_white",
#                   showlegend=True,
#                   )
#
# fig.update_layout(
#     margin=dict(l=20, r=20, t=20, b=20),
# )

# SAVE
# fig.write_image("intro_expl_var_w_error.pdf", scale=3)
fig.write_image(f"rpca_wout_subplots_w_error.png", scale=3)

