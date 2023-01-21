"""Plot some driven time series for rc with and without network. """
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities
import src.esn_src.measures as meas

def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return "rgba" + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))

# Create data:
sys_obj = sims.Lorenz63(dt=0.1)
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 2000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 555,
                    "n_train_sects": 15,
                    "n_validate_sects": 1,
                    "normalize_and_center": False,
                    }

n_train = ts_creation_args["n_train_sects"]
train_sync_steps = ts_creation_args["t_train_sync"]
train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                         **ts_creation_args)

x_dim = sys_obj.sys_dim

# Build RC args:
build_args = {
    "x_dim": x_dim,
    "r_dim": 500,
    "n_rad": 0.4,
    "n_avg_deg": 5.0,
    "n_type_opt": "erdos_renyi",
    "r_to_rgen_opt": "linear_r",
    "act_fct_opt": "tanh",
    "node_bias_opt": "random_bias",
    "node_bias_scale": 0.0,
    "w_in_opt": "random_sparse",
    "w_in_scale": 1,
    "x_train_noise_scale": 0.0,
    "reg_param": 1e-7,
    "ridge_regression_opt": "bias",
    "scale_input_bool": True,
}

x_dim = build_args["x_dim"]
r_dim = build_args["r_dim"]

# Ensemble size:
n_ens = 15

# seeds:
seed = 300
rng = np.random.default_rng(seed)
seeds = rng.integers(0, 10000000, size=n_ens)

# Do experiment:
for i_ens in range(n_ens):
    print(i_ens)

    # Build rc with network:
    esn_obj = esn.ESN()
    with utilities.temp_seed(seeds[i_ens]):
        esn_obj.build(**build_args)

    # Build rc without network:
    esn_obj_nonet = esn.ESN()
    with utilities.temp_seed(seeds[i_ens]):
        build_args_nonet = build_args.copy()
        build_args_nonet["n_rad"] = 0.0
        esn_obj_nonet.build(**build_args_nonet)

    for i_train in range(n_train):
        train_data = train_data_list[i_train]

        # Train rc with network and get rpca_states and components:
        _, _, more_out = esn_obj.train(train_data,
                                       sync_steps=train_sync_steps,
                                       more_out_bool=True)
        res_states = more_out["r"]
        pca = PCA()
        res_pca_states = pca.fit_transform(res_states)
        comps = pca.components_


        # Train rc without network:
        _, _, more_out_nonet = esn_obj_nonet.train(train_data,
                                                   sync_steps=train_sync_steps,
                                                   more_out_bool=True)
        res_states_nonet = more_out_nonet["r"]
        pca_nonet = PCA()
        res_pca_states_nonet = pca_nonet.fit_transform(res_states_nonet)
        comps_nonet = pca_nonet.components_

        if i_train == 0 and i_ens == 0:
            train_steps = res_pca_states.shape[0]

            # data for states:
            results_pca_states = np.zeros((n_ens, n_train, train_steps, r_dim))
            results_pca_states_nonet = np.zeros((n_ens, n_train, train_steps, r_dim))

            # data for comps:
            results_pca_comps = np.zeros((n_ens, n_train, r_dim, r_dim))
            results_pca_comps_nonet = np.zeros((n_ens, n_train, r_dim, r_dim))

        # get data states:
        results_pca_states[i_ens, i_train, :, :] = res_pca_states
        results_pca_states_nonet[i_ens, i_train, :, :] = res_pca_states_nonet

        # get data comps:
        results_pca_comps[i_ens, i_train, :, :] = comps
        results_pca_comps_nonet[i_ens, i_train, :, :] = comps_nonet


# CORRELATIONS:
# data containers:
results_ts_corr = np.zeros((n_ens, n_train, r_dim))
results_comp_corr = np.zeros((n_ens, n_train, r_dim))
# get corr:
for i_ens in range(n_ens):
    for i_train in range(n_train):
        for i_pc in range(r_dim):
            # # correlate time series:
            # ts_corr = np.correlate(results_pca_states[i_ens, i_train, :, i_pc],
            #                       results_pca_states_nonet[i_ens, i_train, :, i_pc])
            # # correlate components:
            # comp_corr = np.correlate(results_pca_comps[i_ens, i_train, i_pc, :],
            #                         results_pca_comps_nonet[i_ens, i_train, i_pc, :])

            # correlate time series:
            ts_corr = np.corrcoef(results_pca_states[i_ens, i_train, :, i_pc],
                                  results_pca_states_nonet[i_ens, i_train, :, i_pc])[0, 1]
            # correlate components:
            comp_corr = np.corrcoef(results_pca_comps[i_ens, i_train, i_pc, :],
                                    results_pca_comps_nonet[i_ens, i_train, i_pc, :])[0, 1]


            results_ts_corr[i_ens, i_train, i_pc] = ts_corr
            results_comp_corr[i_ens, i_train, i_pc] = comp_corr

# Absolute correlations:
abs_ts_corr = np.abs(results_ts_corr)
abs_comp_corr = np.abs(results_comp_corr)

# Get median, lower and higher quartile:
abs_ts_corr_median = np.median(abs_ts_corr, axis=(0, 1))
abs_ts_corr_low = np.quantile(abs_ts_corr, q=0.25, axis=(0, 1))
abs_ts_corr_high = np.quantile(abs_ts_corr, q=0.75, axis=(0, 1))

abs_comp_corr_median = np.median(abs_comp_corr, axis=(0, 1))
abs_comp_corr_low = np.quantile(abs_comp_corr, q=0.25, axis=(0, 1))
abs_comp_corr_high = np.quantile(abs_comp_corr, q=0.75, axis=(0, 1))


# PLOT COLORS:
pca_state_hex_col = "#943126" # blue
pca_comp_hex_col = "#196F3D" # green


# PLOT CORRELATIONS:
max_pc = None
xaxis_title = r"$\large \text{principal component } i$"
# height = 350
height = 400
# width = int(2.0 * height)
# width = int(1.3 * height)
# width = int(0.8 * height)
width = int(0.9 * height)
xrange = [0.5, 10.5]
yrange = None  # [0, 15]
# font_size = 18
font_size = 20
font_family = "Times New Roman"
# max_pc = None


# PC time series correlation:
# yaxis_title = r"$\large{Corr. Coef. } r^{\rho = 0}_{\text{pc}, i}, r_{\text{pc}, i}$"
# yaxis_title = r"$\large C(r_{\text{pc}, i}, r^0_{\text{pc}, i})$"
yaxis_title = r"$\large |C(r_{\text{pc}, i}, \hat{r}_{\text{pc}, i})|$"

x = np.arange(1, r_dim + 1).tolist()[:max_pc]
y = abs_ts_corr_median.tolist()[:max_pc]
y_low = abs_ts_corr_low.tolist()[:max_pc]
y_high = abs_ts_corr_high.tolist()[:max_pc]

hex_color = pca_state_hex_col
rgb_line = hex_to_rgba(hex_color, 1.0)
rgb_fill = hex_to_rgba(hex_color, 0.45)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.array(x),
        y=np.array(y),
        mode="lines+markers",
        line=dict(color=rgb_line,
                  width=3,
                  ),
    )
)
fig.add_trace(
    go.Scatter(
        x=x + x[::-1],  # x, then x reversed
        y=y_high + y_low[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor=rgb_fill,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    )
)
fig.add_vline(x=x_dim,
              line_width=3,
              line_dash="dash",
              line_color="black",
              annotation_text= r"$\Large x_\text{dim}$",
              annotation_font_size= 30,
              annotation_position="top right",
              annotation_xshift = 10,
              )

fig.update_layout(
    width=width,
    height=height,
    font=dict(
        size=font_size,
        family=font_family
    ),
    template="simple_white",
    xaxis=dict(
        dtick=1,
        tick0 = 1,
        range=xrange,
        title=xaxis_title,
    ),
    yaxis=dict(
        range=yrange,
        title=yaxis_title,
    ),
    margin=dict(l=20, r=20, t=20, b=20),
    showlegend=False,
)
file_name = f"corr_pc_time_series_ens.png"
fig.write_image(file_name, scale=3)


# PC components correlation:
yaxis_title = r"$\large | C(\boldsymbol{p}_i, \hat{\boldsymbol{p}}_i) |$"

x = np.arange(1, r_dim + 1).tolist()[:max_pc]
y = abs_comp_corr_median.tolist()[:max_pc]
y_low = abs_comp_corr_low.tolist()[:max_pc]
y_high = abs_comp_corr_high.tolist()[:max_pc]
hex_color = pca_comp_hex_col
rgb_line = hex_to_rgba(hex_color, 1.0)
rgb_fill = hex_to_rgba(hex_color, 0.45)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=np.array(x),
        y=np.array(y),
        mode="lines+markers",
        line=dict(color=rgb_line,
                  width=3
                  ),
    )
)
fig.add_trace(
    go.Scatter(
        x=x + x[::-1],  # x, then x reversed
        y=y_high + y_low[::-1], # upper, then lower reversed
        fill='toself',
        fillcolor=rgb_fill,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    )
)
fig.add_vline(x=x_dim,
              line_width=3,
              line_dash="dash",
              line_color="black",
              annotation_text= r"$\Large x_\text{dim}$",
              annotation_font_size= 30,
              annotation_position="top right",
              annotation_xshift = 10,
              )

fig.update_layout(
    width=width,
    height=height,
    font=dict(
        size=font_size,
        family=font_family
    ),
    template="simple_white",
    xaxis=dict(
        dtick=1,
        tick0 = 1,
        range=xrange,
        title=xaxis_title,
    ),
    yaxis=dict(
        range=yrange,
        title=yaxis_title,
    ),
    margin=dict(l=20, r=20, t=20, b=20),
    showlegend=False,
)

# fig.show()
file_name = f"corr_pc_components_ens.png"
fig.write_image(file_name, scale=3)


# SCATTER PLOTS:

# height = 500
# width = int(1.1*height)
height = 350
width = int(1.7*height)
font_size = 15
font_family = "Times New Roman"
# horizontal_spacing = 0.2
horizontal_spacing = 0.18
vertical_spacing = 0.2


# PLOT PC TIMESERIES AS 4 Scatterplots:
marker_color = hex_to_rgba(pca_state_hex_col, alpha=0.3)
marker_line = hex_to_rgba(pca_state_hex_col, alpha=0.5)
marker_size = 3
marker_line_width = 0.3

fig = make_subplots(
    rows=2, cols=2,
    horizontal_spacing=horizontal_spacing,
    vertical_spacing=vertical_spacing,
    # subplot_titles = (r"$\large i = 1$",
    #                   r"$\large i = 2$",
    #                   r"$\large i = 3$",
    #                   r"$\large i = 4$")
)
i_pc = 0
for row in (1, 2):
    for col in (1, 2):
        fig.add_trace(
            go.Scatter(
                x=results_pca_states[0, 0, :, i_pc],
                y=results_pca_states_nonet[0, 0, :, i_pc],
                mode="markers",
                marker=dict(size=marker_size,
                            color=marker_color,
                            line=dict(width=marker_line_width,
                                      color=marker_line)
                            )
            ),
            row=row, col=col
        )

        # if i_pc == 2
        fig.update_xaxes(
            title= r"$r_{\text{pc}, " + f"{i_pc+1}" + "}$",
            title_standoff=0.0,
            # zeroline=True,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(60, 60, 60, 0.2)",
            row=row, col=col
        )
        fig.update_yaxes(
            title= r"$\hat{r}_{\text{pc}, " + f"{i_pc+1}" + "}$",
            title_standoff=0.0,
            # zeroline=True,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(60, 60, 60, 0.2)",
            row=row, col=col
        )
        i_pc+=1

fig.update_layout(
    width=width,
    height=height,
    font=dict(
        size=font_size,
        family=font_family
    ),
    showlegend=False,
    template="simple_white",
    margin=dict(l=20, r=20, t=20, b=20),
)
file_name = f"rpca_states_scatter.png"
fig.write_image(file_name, scale=3)



# PLOT PC COMPONENTS AS 4 Scatterplots:
marker_color = hex_to_rgba(pca_comp_hex_col, alpha=0.3)
marker_line = hex_to_rgba(pca_comp_hex_col, alpha=0.5)
marker_size = 3
marker_line_width = 0.3

fig = make_subplots(
    rows=2, cols=2,
    horizontal_spacing=horizontal_spacing,
    vertical_spacing=vertical_spacing,
    # subplot_titles = (r"$\large i = 1$",
    #                   r"$\large i = 2$",
    #                   r"$\large i = 3$",
    #                   r"$\large i = 4$")
)
i_pc = 0
for row in (1, 2):
    for col in (1, 2):
        fig.add_trace(
            go.Scatter(
                x=results_pca_comps[0, 0, i_pc, :],
                y=results_pca_comps_nonet[0, 0, i_pc, :],
                mode="markers",
                marker=dict(size=marker_size,
                            color=marker_color,
                            line=dict(width=marker_line_width,
                                      color=marker_line)
                            )
            ),
            row=row, col=col
        )

        # if i_pc == 2
        fig.update_xaxes(
            title= r"$\boldsymbol{p}_" + f"{i_pc+1}" + "$",
            title_standoff=0.0,
            # zeroline=True,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(60, 60, 60, 0.2)",
            row=row, col=col
        )
        fig.update_yaxes(
            title= r"$\hat{\boldsymbol{p}}_" + f"{i_pc+1}" + "$",
            title_standoff=0.0,
            # zeroline=True,
            showgrid=True,
            gridwidth=1,
            gridcolor="rgba(60, 60, 60, 0.2)",
            row=row, col=col
        )
        i_pc+=1

fig.update_layout(
    width=width,
    height=height,
    font=dict(
        size=font_size,
        family=font_family
    ),
    showlegend=False,
    template="simple_white",
    margin=dict(l=20, r=20, t=20, b=20),
)
file_name = f"rpca_comps_scatter.png"
fig.write_image(file_name, scale=3)


# # TEST PLOTS:
# # choose PC index:
# i_pc = 4
#
# # Plot pc states as time series:
# alpha = 0.5
# colors = ["#EA3546", "#248CA9"]
# t_max = 100
# fig = go.Figure()
# x = np.arange(train_steps)
# # with network:
# fig.add_trace(
#     go.Scatter(
#         x=x[:t_max],
#         y=results_pca_states[0, 0, :t_max, i_pc],
#         line=dict(color=hex_to_rgba(colors[0], alpha=alpha)),
#         name="with network")
# )
# # without network:
# fig.add_trace(
#     go.Scatter(
#         x=x[:t_max],
#         y=results_pca_states_nonet[0, 0, :t_max, i_pc],
#         line=dict(color=hex_to_rgba(colors[1], alpha=alpha)),
#         name="no network")
# )
# file_name = f"rpca_time_series_test.png"
# fig.write_image(file_name, scale=3)
#
#
# # Plot pc states as scatter:
# fig = go.Figure()
# fig.add_trace(
#     go.Scatter(
#         x=results_pca_states[0, 0, :, i_pc],
#         y=results_pca_states_nonet[0, 0, :, i_pc],
#         mode="markers",
#         marker=dict(size=5,
#                     color="rgba(60, 30, 100, 0.5)",
#                     line=dict(width=1,
#                               color="black")
#                     )
#     )
# )
# fig.update_layout(
#     yaxis=dict(
#         title="no network"
#     ),
#     xaxis=dict(
#         title="network"
#     )
# )
# file_name = f"rpca_states_scatter_test.png"
# fig.write_image(file_name, scale=3)
#
#
# # Plot pc comps as scatter:
# fig = go.Figure()
# fig.add_trace(
#     go.Scatter(
#         x=results_pca_comps[0, 0, i_pc, :],
#         y=results_pca_comps_nonet[0, 0, i_pc, :],
#         mode="markers",
#         marker=dict(size=5,
#                     color="rgba(60, 60, 60, 0.5)",
#                     line=dict(width=1,
#                               color="black")
#                     )
#     )
# )
# fig.update_layout(
#     yaxis=dict(
#         title="no network"
#     ),
#     xaxis=dict(
#         title="network"
#     )
# )
# file_name = f"rpca_comps_test.png"
# fig.write_image(file_name, scale=3)
