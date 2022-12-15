"""Create the correlation plot between pca and input. """
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import plotly.express as px

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities
import src.esn_src.measures as meas
col_pal = px.colors.qualitative.Plotly
col_pal_iterator = itertools.cycle(col_pal)

def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return "rgba" + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))

# Create data:
sys_obj = sims.Lorenz63(dt=0.1)
# sys_obj = sims.Lorenz96(sys_dim=10)
# sys_obj = sims.LinearSystem()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 1000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 1000,
                    "n_train_sects": 10,
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
    "node_bias_scale": 0.4,
    "w_in_opt": "random_sparse",
    "w_in_scale": 1.0,
    "x_train_noise_scale": 0.0,
    "reg_param": 1e-7,
    "ridge_regression_opt": "bias",
    "scale_input_bool": True,
}

x_dim = build_args["x_dim"]

# Ensemble size:
n_ens = 2

# seeds:
seed = 250
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
        inp = train_data[train_sync_steps:-1, :]
        _, _, more_out = esn_obj.train(train_data, sync_steps=train_sync_steps, more_out_bool=True)
        res_states = more_out["r"]
        pca = PCA()
        pca.fit(res_states)
        components = pca.components_ # n_components, n_features

        if i_train == 0 and i_ens == 0:
            # explained variances:
            n_components = components.shape[0]
            correlation_results = np.zeros((n_ens, n_train, n_components, x_dim))
        w_in = esn_obj.w_in

        for i_x in range(x_dim):
            for i_comp in range(n_components):
                # correlation_results[i_ens, i_train, i_comp, i_x] = np.dot(esn_obj.w_in[:, i_x], components[i_comp, :])
                # corr = np.corrcoef(w_in[:, i_x], components[i_comp, :])
                corr = np.correlate(w_in[:, i_x], components[i_comp, :])
                correlation_results[i_ens, i_train, i_comp, i_x] = corr

# Calculate mean correlation summed over x_dim:
abs_summed_corr = np.sum(np.abs(correlation_results), axis=-1)

# mean absolute summed correlation:
mean_abs_summed_corr = np.median(abs_summed_corr, axis=(0, 1))

# low absolute summed correlation:
low_abs_summed_corr = np.min(abs_summed_corr, axis=(0, 1))
# low_abs_summed_corr = np.quantile(abs_summed_corr, q=0.25, axis=(0, 1))

# high absolute summed correlation:
high_abs_summed_corr = np.max(abs_summed_corr, axis=(0, 1))
# high_abs_summed_corr = np.quantile(abs_summed_corr, q=0.75, axis=(0, 1))


# only abs corr:
abs_corr = np.abs(correlation_results)

# mean abs corr:
mean_abs_corr = np.median(abs_corr, axis=(0, 1))

# low absolute corr:
low_abs_corr = np.min(abs_corr, axis=(0, 1))

# high absolute corr:
high_abs_corr = np.max(abs_corr, axis=(0, 1))


# PLOT:
height = 300
width = int(1.3 * height)
font_size = 25
legend_font_size = 25
font_family = "Times New Roman"

# xaxis_title = r"$\text{Principal component } \boldsymbol{p}_i$"
xaxis_title = r"$\Large\text{Principal component } i$"
# yaxis_title =  r"$\sum_i |\text{corr}_{ij}|$"
# yaxis_title =  r"$\text{Summed correlation between } W_\text{in} \text{ and } \boldsymbol{p}_i$"
yaxis_title =  r"$\Large A_i$"

# first plot:
max_comp = 25

# Get data:
x = np.arange(1, max_comp + 1)

x_list = list(x)
y = list(mean_abs_summed_corr)[:max_comp]
low_list = list(low_abs_summed_corr)[:max_comp]
high_list = list(high_abs_summed_corr)[:max_comp]

linewidth=3
# colors:
hex_color = next(col_pal_iterator)
rgb_line = hex_to_rgba(hex_color, 1.0)
rgb_fill = hex_to_rgba(hex_color, 0.45)

# plot median:
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=x,
        y=mean_abs_summed_corr,
        line=dict(color=rgb_line,
                  width=linewidth),
        showlegend=False)
)

fig.add_trace(
    go.Scatter(
        x=x_list + x_list[::-1],  # x, then x reversed
        y=high_list + low_list[::-1], # upper, then lower reversed
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
              # annotation_text= r"$x_\text{dim} $" + fr"${x_dim}$",
              annotation_text= r"$\Large x_\text{dim}$",
              annotation_font_size=30,
              annotation_position="top right"
              )

fig.update_layout(template="simple_white",
                  showlegend=False,
                  font=dict(
                      size=font_size,
                      family=font_family
                  ),
                  xaxis_title=xaxis_title,
                  yaxis_title=yaxis_title
                  )

fig.update_layout(
    margin=dict(l=70, r=20, t=20, b=50),
)

# SAVE
file_name = f"corr_of_pca_with_win.png"
fig.write_image(file_name, scale=3)


###### For each Dimensions now: ######

# PLOT:
height = 300
width = int(1.3 * height)
font_size = 25
legend_font_size = 25
font_family = "Times New Roman"

# xaxis_title = r"$\text{Principal component } \boldsymbol{p}_i$"
xaxis_title = r"$\Large\text{Principal component } i$"
# yaxis_title =  r"$\sum_i |\text{corr}_{ij}|$"
# yaxis_title =  r"$\text{Summed correlation between } W_\text{in} \text{ and } \boldsymbol{p}_i$"
# yaxis_title =  r"$\Large A_i$"
yaxis_title =  r"?"

linewidth=3


# first plot:
max_comp = 25

# Get data:
x = np.arange(1, max_comp + 1)

x_list = list(x)

fig = go.Figure()
for i_x in range(x_dim):
    # colors:
    hex_color = next(col_pal_iterator)
    rgb_line = hex_to_rgba(hex_color, 1.0)
    rgb_fill = hex_to_rgba(hex_color, 0.45)

    y = list(mean_abs_corr[:, i_x])[:max_comp]
    low_list = list(low_abs_corr[:, i_x])[:max_comp]
    high_list = list(high_abs_corr[:, i_x])[:max_comp]

    # plot median:
    fig.add_trace(
        go.Scatter(
            x=x_list,
            y=y,
            line=dict(color=rgb_line,
                      width=linewidth),
            name=f"{i_x + 1}",
            showlegend=True)
    )

    fig.add_trace(
        go.Scatter(
            x=x_list + x_list[::-1],  # x, then x reversed
            y=high_list + low_list[::-1], # upper, then lower reversed
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
                  # annotation_text= r"$x_\text{dim} $" + fr"${x_dim}$",
                  annotation_text= r"$\Large x_\text{dim}$",
                  annotation_font_size=30,
                  annotation_position="top right"
                  )

fig.update_layout(template="simple_white",
                  showlegend=True,
                  font=dict(
                      size=font_size,
                      family=font_family
                  ),
                  xaxis_title=xaxis_title,
                  yaxis_title=yaxis_title
                  )

fig.update_layout(
    margin=dict(l=70, r=20, t=20, b=50),
)

# SAVE
file_name = f"corr_of_pca_with_win_dimwise.png"
fig.write_image(file_name, scale=3)
