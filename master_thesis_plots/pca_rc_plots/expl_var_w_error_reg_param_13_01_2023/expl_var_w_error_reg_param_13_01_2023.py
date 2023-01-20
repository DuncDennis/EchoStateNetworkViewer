"""Create an explained variance with error band plot and sweep some parameters. """
import numpy as np
from sklearn.decomposition import PCA
import plotly
import plotly.graph_objects as go
from plotly.express.colors import sample_colorscale
import itertools
import plotly.express as px
col_pal = px.colors.qualitative.Plotly
col_pal_iterator = itertools.cycle(col_pal)


import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities

# plotly.colors.PLOTLY_SCALES["Portland"]


def get_color_value(val: float,
                    full_val_range: list[float, float],
                    log10_bool: bool = False):

    if log10_bool:
        val = np.log10(val)
        full_val_range = [np.log10(full_val_range[0]),
                          np.log10(full_val_range[1])]

    print(val, full_val_range)

    ratio = (val - full_val_range[0])/(full_val_range[1] - full_val_range[0])
    return sample_colorscale('Portland', [ratio])[0]


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

# Ensemble size:
n_ens = 15

# seeds:
seed = 300
rng = np.random.default_rng(seed)
seeds = rng.integers(0, 10000000, size=n_ens)


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
            # expl_var_ratio_results = np.zeros((n_ens, n_train, n_components))
            expl_var_results = np.zeros((n_ens, n_train, n_components))

        # explained variances:
        # expl_var_ratio_results[i_ens, i_train, :] = expl_var_ratio
        expl_var_results[i_ens, i_train, :] = expl_var

# Reg param values:
reg_param_values = [1e-1, 1e-4, 1e-7, 1e-10,
                    1e-13]
reg_trace_name = r"$\large \beta = "

# Plot:
fig = go.Figure()

legend_orientation = "v"
legend_pos_y = 1.07
legend_pos_x = 0.95
entrywidth = 0

yaxis_title = r"$\Large\phi_i$"
xaxis_title =  r'$\Large \text{principal component } i$'
title = None
height = 400
width = int(1.6 * height)

x_max = expl_var_results.shape[-1]
# xrange = [-5, x_max + 5]
xrange = [0, x_max + 5]
# xrange = None
# yrange = None  # [0, 15]
yrange = [-32, 3]  # [0, 15]
# yrange=None
font_size = 23
legend_font_size = 23
font_family = "Times New Roman"
linewidth=3


# plot ev
n_components = expl_var_results[0, 0, :].size
x = np.arange(0, n_components)
y = np.median(expl_var_results, axis=(0, 1))
y_low = np.min(expl_var_results, axis=(0, 1))
y_high = np.max(expl_var_results, axis=(0, 1))
print(x.shape, y.shape, y_low.shape, y_high.shape)
x = x.tolist()
y = y.tolist()
y_low = y_low.tolist()
y_high = y_high.tolist()
hex_color="#000000"
# hex_color = next(col_pal_iterator)
rgb_line = hex_to_rgba(hex_color, 1.0)
rgb_fill = hex_to_rgba(hex_color, 0.35)
fig.add_trace(
    go.Scatter(x=x, y=y, showlegend=False,
               line=dict(color=rgb_line,
                         width=linewidth))
)
fig.add_trace(
    go.Scatter(
        x=x + x[::-1],  # x, then x reversed
        y=y_high + y_low[::-1],  # upper, then lower reversed
        fill='toself',
        fillcolor=rgb_fill,
        line=dict(color='rgba(255,255,255,0)'),
        hoverinfo="skip",
        showlegend=False
    )
)

# Plot reg_param lines:
for i_reg, reg in enumerate(reg_param_values):
    hex_color = next(col_pal_iterator)
    # rgb_line = hex_to_rgba(hex_color, 1.0)

    rgb_line = get_color_value(val=reg,
                               full_val_range=[1e-13, 1e-1],
                               log10_bool=True)

    ev_cutoff = reg/ts_creation_args["t_train"]
    fig.add_trace(
        go.Scatter(x=xrange,
                   y=[ev_cutoff, ev_cutoff],
                   line=dict(color=rgb_line,
                             dash="dash",
                             width=3),
                   marker=dict(size=0),
                   name=reg_trace_name + r"10^{" + f"{int(np.log10(reg))}" + "}$",
                   showlegend=True,
                   mode="lines",
                   ),
    )
    if i_reg == 0:
        # annotation for last ev_cutoff
        fig.add_annotation(x=1, y=np.log10(ev_cutoff),
                           text=r"$\large \phi_\text{co}(\beta)$",
                           showarrow=False,
                           yshift=+25,
                           xshift=+100,)

fig.update_layout(
    template="simple_white",
    showlegend=True,
    title=title,
    width=width,
    height=height,

    xaxis=dict(
        range=xrange,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.2)',
    ),

    yaxis=dict(
        range=yrange,
        exponentformat= 'power',
        type="log",
        dtick=10,
        showgrid=True,
        gridwidth=1,
        gridcolor='rgba(0,0,0,0.2)',
    ),

    yaxis_title=yaxis_title,
    xaxis_title=xaxis_title,

    font=dict(
        size=font_size,
        family=font_family
    ),
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
    )

    )


fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)

# SAVE
fig.write_image(f"expl_var_w_error_reg_param.png", scale=3)
