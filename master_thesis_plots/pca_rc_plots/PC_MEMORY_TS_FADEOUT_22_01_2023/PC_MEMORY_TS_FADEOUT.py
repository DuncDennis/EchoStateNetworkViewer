"""Drive reservoir with signal that abrubtly stops. See echo in reservoir states. """
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import itertools
import plotly.express as px
col_pal = px.colors.qualitative.Plotly
col_pal_iterator = itertools.cycle(col_pal)

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
# sys_obj = sims.ComplexButterfly()
# sys_obj = sims.LinearSystem()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 2000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 110,
                    "n_train_sects": 5,
                    "n_validate_sects": 1,
                    "normalize_and_center": False,
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
    # "n_rad": 0.0,
    # "n_rad": 0.1,
    # "n_rad": 0.9,
    "n_avg_deg": 5.0,
    "n_type_opt": "erdos_renyi",
    "r_to_rgen_opt": "linear_r",
    "act_fct_opt": "tanh",
    "node_bias_opt": "random_bias",
    "node_bias_scale": 0.4,
    # "node_bias_scale": 0.0,
    "w_in_opt": "random_sparse",
    "w_in_scale": 1.0,
    "x_train_noise_scale": 0.0,
    "reg_param": 1e-7,
    "ridge_regression_opt": "bias",
    "scale_input_bool": True,
}

# N ens:
n_ens = 5

# seeds:
seed = 300
rng = np.random.default_rng(seed)
seeds = rng.integers(0, 10000000, size=n_ens)

t_cutoff = 100

# Do experiment:
for i_ens in range(n_ens):
    print(i_ens)

    # Build rc with network:
    esn_obj = esn.ESN()
    with utilities.temp_seed(seeds[i_ens]):
        esn_obj.build(**build_args)

    for i_train in range(n_train):
        train_data = train_data_list[i_train]

        # Train rc with network and get rpca_states and components:
        _, _, more_out = esn_obj.train(train_data,
                                       sync_steps=train_sync_steps,
                                       more_out_bool=True)
        res_states = more_out["r"]
        pca = PCA()
        res_pca_states = pca.fit_transform(res_states)

        # reset and sync reservoir:
        # use validate data here:
        # validate_data = validate_data_list_of_lists[i_train][0]
        validate_data = validate_data_list_of_lists[-1][0]
        sync = validate_data[:pred_sync_steps]
        true_data = validate_data[pred_sync_steps:]
        # reset:
        esn_obj.reset_reservoir()
        # sync the reservoir:
        esn_obj.drive(sync, more_out_bool=False)
        # frive with modfied signal:
        to_repeat = true_data.shape[0] - t_cutoff

        # value_to_repeat = true_data[t_cutoff] # 3 dim
        # value_to_repeat = np.array([0, 0, 0]) # 3 dim
        value_to_repeat = esn_obj.standard_scaler.mean_ # 3 dim

        true_data[t_cutoff:, :] = np.repeat(value_to_repeat[np.newaxis, :], to_repeat, axis=0)

        res_drive_states, more_out = esn_obj.drive(true_data, more_out_bool=True)
        # PC transform driven states:
        pc_driven_states = pca.transform(res_drive_states)
        if i_train == 0 and i_ens == 0:
            results_pc_driven_states = np.zeros((n_ens,
                                                 n_train,
                                                 pc_driven_states.shape[0],
                                                 pc_driven_states.shape[1]))

        results_pc_driven_states[i_ens, i_train, :, :] = pc_driven_states
# Do experiment:

# dimensions = [0, 1, 5, 10, 25, 50, 100, 200, 300, 500]
# dimensions = np.arange(10).tolist() + [49, 199, 299]
# dimensions = [0, 1, 2, 3, 4, 5, 6, 19, 50, 100, 200, 300]
# dimensions = [0, 4, 49, 99, 199, 299, 399]
# dimensions = (np.array([1, 5, 10, 50, 150, 500])-1).tolist()
dimensions = (np.array([1, 5, 10, 50, 200, 500])-1).tolist()
nr_dims = len(dimensions)

# PLOT:
height = 450
# width = int(1.4 * height)
# width = int(0.8 * height)
# width = int(1 * height)
width = int(0.8 * height)
# width = int(0.5 * height)
font_size = 15
legend_font_size = 15
font_family = "Times New Roman"
# linewidth = 0.6
linewidth = 0.5

# yaxis_title = r"$r_\text{pca, i}$"
# yaxis_title = r"$\large r_{\text{pc}, i}(t + \Delta t) - \large r_{\text{pc}, i}(t)$"
yaxis_title = r"$\large |r_{\text{pc}, i}(t + \Delta t) - \large r_{\text{pc}, i}(t)|$"
xaxis_title =  r'$\large \text{time steps}$'
# xaxis_title =  r'test'

fig = make_subplots(rows=nr_dims, cols=1, shared_yaxes=True,
                    shared_xaxes=True, vertical_spacing=None,
                    print_grid=True,
                    # x_title=xaxis_title,
                    # y_title=yaxis_title,
                    # row_heights=[height] * nr_taus,
                    # column_widths=[width],
                    # subplot_titles=["input signal"] + [fr"$i = {x}$" for x in dimensions],
                    # subplot_titles=[fr"$i = {x + 1}$" for x in dimensions],
                    # horizontal_spacing=1,
                    # row_titles=[fr"$i = {x}$" for x in dimensions],
                    )
# add input signal

# fig.add_trace(
#     go.Scatter(y=inp[:, 0], name="input",
#                line=dict(
#                    width=linewidth
#                ),
#                mode="lines", showlegend=False,
#                ),
#     row=1, col=1
# )


hex_color = next(col_pal_iterator)

x_min, x_max = t_cutoff - 2, t_cutoff + 4

# get y range for each i_ens, i_train and i_d:
selection = np.diff(results_pc_driven_states, axis=2)[:, :, x_min: x_max + 1, :]
min_y = np.min(selection, axis=(0, 1, 2))
max_y = np.max(selection, axis=(0, 1, 2))

# abs diff:
selection = np.abs(np.diff(results_pc_driven_states, axis=2))[:, :, x_min: x_max + 1, :]
# min_y = np.min(selection, axis=(0, 1, 2))
min_y = np.zeros(len(dimensions))
max_y = np.max(selection, axis=(0, 1, 2))

color = hex_to_rgba(hex_color, 0.7)
for i_d, d in enumerate(dimensions):

    for i_ens in range(n_ens):
        for i_train in range(n_train):
            pc_driven_states = results_pc_driven_states[i_ens, i_train, :, :]
            # hex_color = "#525252"
            # hex_color = next(col_pal_iterator)
            # fig.add_hline(y=0,
            #               row=i_d + 1, col=1,
            #               line_width=1.0,
            #               line_dash="dash",
            #               line_color="black",
            #               opacity=1.0
            #               )
            print(color)

            # y = np.diff(pc_driven_states[:, i_d])
            y = np.abs(np.diff(pc_driven_states[:, i_d]))
            # y = pc_driven_states[:, i_d]

            # for rpca in results_rpca:
            fig.add_trace(
                go.Scatter(y=y,
                           line=dict(
                               width=linewidth, color=color
                           ),
                           mode="lines", showlegend=False,
                           ),
                row=i_d + 1, col=1
            )
    fig.update_xaxes(
        dtick=1,
        range=[x_min, x_max],
        row=i_d + 1, col=1,
    )

    # y_selection = y[t_cutoff-2: t_cutoff + 4]
    # min_y = np.min(y_selection)
    # max_y = np.max(y_selection)

    fig.update_yaxes(
        nticks=1,
        tick0=0.0,
        showticklabels=True,
        range = [min_y[i_d], max_y[i_d]],
        ticks="",
        row=i_d + 1, col=1,
    )
    fig.add_annotation(text=f"$\large i = {d + 1}$",
                       showarrow=False,
                       xshift=111,
                       yshift=15,
                       row=i_d + 1, col=1,
                       # row=1, col=1,
                       )

fig.update_xaxes(
    title=xaxis_title,
    title_standoff=10,
    row=i_d + 1, col=1
)

fig.update_yaxes(
    title=yaxis_title,
    title_standoff=10,
    row=3, col=1)

fig.add_vline(x=t_cutoff,
              row="all",
              line_width=1.5,
              line_dash="dot",
              line_color="black",
              opacity=1
              # line_color=hex_to_rgba(hex_color, 1.0),
              # line_opacity=0.9,
              )

fig.update_layout(
    # template="plotly_white",
    template="simple_white",
                  # showlegend=False,
                  showlegend=True,
                  font=dict(
                      size=font_size,
                      family=font_family
                  ),
                  )
fig.update_layout(
    # margin=dict(l=70, r=20, t=20, b=50),
    margin=dict(l=45, r=20, t=20, b=50),
    showlegend=False,
)

# fig.update_yaxes(showticklabels=False)

# for i in range(nr_dims+1):
#     # fig.layout.annotations[i].update(x=0.1)
#     fig.layout.annotations[i].update(x=0.09)
    # fig.layout.annotations[i].update(x=0.15)
    # fig.layout.annotations[i].update(x=0.11)

fig.update_layout(
    width=width,
    height=height,
    # legend=dict(
    #     orientation="h",
    #     yanchor="top",
    #     y=1.1,
    #     xanchor="right",
    #     x=0.9,
    #     font=dict(size=legend_font_size)
    # ),
)

fig.add_annotation(showarrow=False,
                   text=r"$\large t_\text{off}$",
                   yshift = 33,
                   xshift = -50)

# SAVE
file_name = f"time_series_rpca_signal_fadeout_ens.png"
# file_name = f"intro_pca_traj_{name}.pdf"
fig.write_image(file_name, scale=3)

