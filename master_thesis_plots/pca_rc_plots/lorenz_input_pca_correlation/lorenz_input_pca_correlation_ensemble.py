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
# sys_obj = sims.LinearSystem()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 1000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 1000,
                    "n_train_sects": 2,
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

x_dim = build_args["x_dim"]

# Ensemble size:
n_ens = 2

# seeds:
seed = 300
rng = np.random.default_rng(seed)
seeds = rng.integers(0, 10000000, size=n_ens)

# time delays:
# tau_list = [0, 1, 2, 3, 4, 5, 8, 10, 15]
tau_list = [0, 1, 2, 3, 5, 8, 10, 20, 30, 50, 100]
n_delays = len(tau_list)

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
        res_pca_states = pca.fit_transform(res_states)
        if i_train == 0 and i_ens == 0:
            # explained variances:
            n_components = res_pca_states.shape[-1]
            correlation_results = np.zeros((n_ens, n_train, n_components, x_dim, n_delays))

        for i_tau, tau in enumerate(tau_list):
            correlation = meas.cross_correlate(inp, res_pca_states, tau)
            correlation_results[i_ens, i_train, :, :, i_tau] = correlation

# PLOT:
max_index = 30
height = 500
width = int(1.3 * height)
font_size = 15
legend_font_size = 20
font_family = "Times New Roman"

yaxis_title = r"$\text{absolute of } C[\boldsymbol{x}(i - \tau), \boldsymbol{r}_\text{pca}(i)]$"
xaxis_title =  r'$\text{Principal Component } \boldsymbol{p}_i$'

nr_taus = len(tau_list)

subplot_titles = [fr"$\tau = {tau}$" for tau in tau_list]
fig = make_subplots(rows=nr_taus, cols=1, shared_yaxes=True,
                    shared_xaxes=True, vertical_spacing=None,
                    print_grid=True,
                    x_title=xaxis_title,
                    y_title=yaxis_title,
                    # row_heights=[height] * nr_taus,
                    # column_widths=[width],
                    subplot_titles=subplot_titles,
                    # row_titles=[str(x) for x in tau_list],
                    )
# shape: (n_ens, n_train, n_components, i_tau)
summed_abs_corr = np.sum(np.abs(correlation_results), axis=3)

# shape (n_components, i_tau):
mean_summed_abs_corr = np.median(summed_abs_corr, axis=(0, 1))
# lower_summed_abs_corr = np.quantile(summed_abs_corr, q=0.25, axis=(0, 1))
# higher_summed_abs_corr = np.quantile(summed_abs_corr, q=0.75, axis=(0, 1))
lower_summed_abs_corr = np.min(summed_abs_corr, axis=(0, 1))
higher_summed_abs_corr = np.max(summed_abs_corr, axis=(0, 1))


# x axis to plot:
x = np.arange(1, max_index+1)


# linewidth:
linewidth = 3

for i_tau, tau in enumerate(tau_list):

    # colors:
    hex_color = next(col_pal_iterator)
    rgb_line = hex_to_rgba(hex_color, 1.0)
    rgb_fill = hex_to_rgba(hex_color, 0.45)

    mean = mean_summed_abs_corr[:max_index, i_tau]
    error_low = lower_summed_abs_corr[:max_index, i_tau]
    error_high = higher_summed_abs_corr[:max_index, i_tau]
    plot = mean

    fig.add_trace(
        go.Bar(x=x,
               y=plot,
               showlegend=False,
               error_y=dict(symmetric=False,
                            thickness=1,
                            array=error_high - mean,
                            arrayminus=mean - error_low)
               ),
        # go.Scatter(x=x,
        #            y=plot,
        #            # name=fr"$\tau = {tau}$",
        #            showlegend=False,
        #            mode="lines",
        #            line=dict(color=rgb_line,
        #                      width=linewidth)
        #            ),
        row=i_tau+1, col=1
    )
    # fig.add_trace(
    #     go.Scatter(
    #         x=x.tolist() + x.tolist()[::-1],  # x, then x reversed
    #         y=error_high.tolist() + error_low.tolist()[::-1],  # upper, then lower reversed
    #         fill='toself',
    #         fillcolor=rgb_fill,
    #         line=dict(color='rgba(255,255,255,0)'),
    #         hoverinfo="skip",
    #         showlegend=False
    #     ),
    #     row=i_tau+1, col=1
    # )

# Y axis range:
# fig.update_yaxes(range=[0, 2.5],
#                  tick0=0.0,
#                  dtick=1.0)

fig.update_layout(template="simple_white",
                  showlegend=False,
                  font=dict(
                      size=font_size,
                      family=font_family
                  ),
                  )
fig.update_layout(
    margin=dict(l=70, r=20, t=20, b=50),
)

fig.update_layout(
    width=width,
    height=height,
    legend=dict(
        # orientation="h",
        yanchor="top",
        y=1.01,
        xanchor="right",
        x=0.95,
        font=dict(size=legend_font_size)
    )
)

# SAVE
file_name = f"lorenz_input_pca_correlation_ensemble.png"
# file_name = f"intro_pca_traj_{name}.pdf"
fig.write_image(file_name, scale=3)

