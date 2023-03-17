"""Create an explained variance with error band plot and sweep some parameters. """
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
import itertools
import copy
import plotly.io as pio
import plotly.express as px
# Plotly col_pal:
# col_pal = px.colors.qualitative.Plotly
# simple_white color palett:
col_pal = pio.templates["simple_white"].layout.colorway
col_pal_iterator = itertools.cycle(col_pal)


import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities

# import plotly.io as pio
# plotly_template = pio.templates["simple_white"]
# print(plotly_template.layout)
# print(plotly_template.layout.colorway)

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
                    "t_validate": 2000,
                    "n_train_sects": 15,
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
    # "n_rad": 0.4,
    "n_avg_deg": 5.0,
    "n_type_opt": "erdos_renyi",
    "r_to_rgen_opt": "linear_r",
    # "act_fct_opt": "tanh",
    "node_bias_opt": "random_bias",
    "node_bias_scale": 0.4,
    "w_in_opt": "random_sparse",
    "w_in_scale": 1.0,
    "x_train_noise_scale": 0.0,
    "reg_param": 1e-7,
    "ridge_regression_opt": "bias",
    "scale_input_bool": True,
}

# Ensemble size:
n_ens = 15

# seeds:
seed = 300
rng = np.random.default_rng(seed)
seeds = rng.integers(0, 10000000, size=n_ens)

# sweep:

sweep_values = ["linear + no network",
                "linear + network",
                "tanh + no network",
                "tanh + network"]

results_sweep = []

for i_sweep, sweep_value in enumerate(sweep_values):
    print(i_sweep, sweep_value)
    if sweep_value == "linear + no network":
        build_args["act_fct_opt"] = "linear"
        build_args["n_rad"] = 0.0
    elif sweep_value == "linear + network":
        build_args["act_fct_opt"] = "linear"
        build_args["n_rad"] = 0.4
    elif sweep_value == "tanh + no network":
        build_args["act_fct_opt"] = "tanh"
        build_args["n_rad"] = 0.0
    elif sweep_value == "tanh + network":
        build_args["act_fct_opt"] = "tanh"
        build_args["n_rad"] = 0.4
    else:
        raise ValueError("Not recognized")

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
            expl_var = np.var(res_pca_states, axis=0)
            # expl_var_ratio = pca.explained_variance_ratio_
            # expl_var_ratio = pca.explained_variance_
            if i_train == 0 and i_ens == 0:
                # explained variances:
                n_components = res_pca_states.shape[1]
                # n_components = expl_var_ratio.size
                expl_var_results = np.zeros((n_ens, n_train, n_components))

                # expl_var_ratio_results = np.zeros((n_ens, n_train, n_components))

            # explained variances:
            expl_var_results[i_ens, i_train, :] = expl_var

    results_sweep.append(expl_var_results)


# Plot creation:
# sweep_key = "act_and_net"
# sweep_name = r"\text{Act. fct. and network}"
sweep_name = r""

# f_name = "act_and_net"



# Basic Figure:
fig = go.Figure()
# yaxis_title = r"$\text{Explained Variance Ratio } \lambda_i$"
# yaxis_title = r"$\text{Explained Variance } \lambda_i$"
# yaxis_title = r"$\text{Expl. variance }\lambda_i$"
# yaxis_title = r"$\Large\lambda_i$"
yaxis_title = r"$\Large\phi_i$"
# xaxis_title =  r'$\text{Principal Component } \boldsymbol{p}_i$'
# xaxis_title =  r'$\Large \text{Principal component } \boldsymbol{p}_i$'
xaxis_title =  r'$\Large \text{principal component } i$'
width = 650
height = int(0.50*width)

log_x=False
log_y=True



font_size = 25
legend_font_size = 25
font_family = "Times New Roman"
linewidth=3



# For each of the four options:
for i_sweep, sweep_value in enumerate(sweep_values):
    # Get results:
    expl_var_ratio_results = results_sweep[i_sweep]
    # Get number of components (x_axis
    n_components = expl_var_ratio_results[0, 0, :].size
    # Get x data:
    x = np.arange(1, n_components + 1)
    # Get y data:
    y = np.median(expl_var_ratio_results, axis=(0, 1))
    y_low = np.min(expl_var_ratio_results, axis=(0, 1))
    y_high = np.max(expl_var_ratio_results, axis=(0, 1))

    # Print shape:
    print(x.shape, y.shape, y_low.shape, y_high.shape)

    # Transform data to lists (to be able to plot cont. error bands.)
    x = x.tolist()
    y = y.tolist()
    y_low = y_low.tolist()
    y_high = y_high.tolist()

    # Get color of line and fill:
    hex_color = next(col_pal_iterator)
    rgb_line = hex_to_rgba(hex_color, 1.0)
    rgb_fill = hex_to_rgba(hex_color, 0.45)

    # Define trace name:
    # trace_name = fr"${sweep_name} = {sweep_value}$"
    trace_name = r"$\text{" + sweep_value + r"}$"
    # trace_name = fr"${sweep_value}$"

    # Add median line:
    fig.add_trace(
        go.Scatter(x=x,
                   y=y,
                   showlegend=True,
                   name=trace_name,
                   line=dict(color=rgb_line,
                             width=linewidth))
    )

    # Add error band fill:
    fig.add_trace(
        go.Scatter(
            x=x + x[::-1],  # x, then x reversed
            y=y_high + y_low[::-1],  # upper, then lower reversed
            fill='toself',
            fillcolor=rgb_fill,
            line=dict(color='rgba(255,255,255,0)'), # no line color
            hoverinfo="skip",
            showlegend=False
        )
    )

# Horizontal line for cutooff:
ev_cutoff = build_args["reg_param"] / ts_creation_args["t_train"]
print("ev cutoff", ev_cutoff)
# add hline for factor plot:
fig.add_trace(
    go.Scatter(x=[1, n_components + 1],
               y=[ev_cutoff, ev_cutoff],
               line=dict(color="black", dash="dash", width=3),
               marker=dict(size=0),
               showlegend=False,
               mode="lines",
               ),
)

fig.add_annotation(x=1, y=np.log10(ev_cutoff),
                   text=r"$\large \phi_\text{co}$",
                   showarrow=False,
                   yshift=-15,
                   xshift=100,)



# Basic figure layout:

fig.update_layout(
    width=width,
    height=height,
    yaxis_title=yaxis_title,
    xaxis_title=xaxis_title,

    font=dict(
        size=font_size,
        family=font_family
    ),
    template="simple_white",
    showlegend=True
    )



if log_x:
    fig.update_layout(
        xaxis={
            'exponentformat': 'power'}
    )
    fig.update_xaxes(type="log")

if log_y:
    fig.update_layout(
        yaxis={
            'exponentformat': 'power'}
    )
    fig.update_yaxes(type="log")

fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)

# total plot:

total_fig = copy.deepcopy(fig)

x_max = np.max([x.shape[-1] for x in results_sweep])
# xrange = None
# yrange = None  # [0, 15]

total_fig.update_xaxes(tick0=0,
                       range=[0, x_max])
total_fig.update_yaxes(dtick=10)

# Legend:
legend_orientation = "v"
legend_pos_y = 1.07
legend_pos_x = 0.95
entrywidth=0

total_fig.update_layout(
    legend=dict(
        orientation=legend_orientation,
        yanchor="top",
        y=legend_pos_y,
        xanchor="right",
        x=legend_pos_x,
        font=dict(size=legend_font_size),
        entrywidth=entrywidth,
        bordercolor="grey",
        borderwidth=2,
        # tracegroupgap=4,
        # entrywidth=2
        # title=fr"${sweep_name}$",
        # title=r"$test$",
        # tracegroupgap = 10
    )
)

# Grid:
total_fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(0,0,0,0.2)',
)
total_fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(0,0,0,0.2)',
)

total_fig.write_image(f"expl_var_w_error_act_and_net_nozoom.png", scale=3)

# Zoomed plot:

zoom_fig = copy.deepcopy(fig)

xrange = [0.7, 6]
yrange = [-2, 2.3]  # [0, 15]

zoom_fig.update_xaxes(
    range=xrange)

zoom_fig.update_yaxes(
    range=yrange,
    dtick=1,
)

# Legend:
legend_orientation = "v"
legend_pos_y = 1.07
legend_pos_x = 0.95
entrywidth=0

zoom_fig.update_layout(
    legend=dict(
        orientation=legend_orientation,
        yanchor="top",
        y=legend_pos_y,
        xanchor="right",
        x=legend_pos_x,
        font=dict(size=legend_font_size),
        entrywidth=entrywidth,
        bordercolor="grey",
        borderwidth=2,
        # tracegroupgap=4,
        # entrywidth=2
        # title=fr"${sweep_name}$",
        # title=r"$test$",
        # tracegroupgap = 10
    )
)

# Grid:
# fig.update_xaxes(
#     showgrid=True,
#     gridwidth=1,
#     gridcolor="gray",
# )
# zoom_fig.update_yaxes(
#     showgrid=True,
#     gridwidth=1,
#     gridcolor="gray",
# )

zoom_fig.update_yaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(0,0,0,0.2)',
)
zoom_fig.update_xaxes(
    showgrid=True,
    gridwidth=1,
    gridcolor='rgba(0,0,0,0.2)',
)

zoom_fig.write_image(f"expl_var_w_error_act_and_net_zoom.png", scale=3)




# x_axis = dict(tickmode="linear",
#               tick0=0,
#               dtick=500)
# x_axis=None

# y_axis = dict(tickmode="linear",
#               tick0=0,
#               dtick=500)

# y_axis=None


# fig.update_yaxes(range=yrange)
# fig.update_xaxes(range=xrange)


# fig.update_xaxes(
#     showgrid=True,
#     gridwidth=1,
#     gridcolor="gray",
# )
# fig.update_yaxes(
#     showgrid=True,
#     gridwidth=1,
#     gridcolor="gray",
# )

# SAVE
# fig.write_image("intro_expl_var_w_error.pdf", scale=3)
# fig.write_image(f"expl_var_w_error_act_and_net_zoom__{f_name}.png", scale=3)

