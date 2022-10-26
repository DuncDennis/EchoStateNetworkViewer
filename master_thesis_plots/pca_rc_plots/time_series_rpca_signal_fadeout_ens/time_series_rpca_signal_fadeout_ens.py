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
sys_obj = sims.Lorenz63()
# sys_obj = sims.ComplexButterfly()
# sys_obj = sims.LinearSystem()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 200,
                    "t_train": 100,
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
    "x_dim": 1,
    "r_dim": 500,
    "n_rad": 0.2,
    "n_avg_deg": 5.0,
    "n_type_opt": "erdos_renyi",
    "r_to_rgen_opt": "linear_r",
    "act_fct_opt": "tanh",
    "node_bias_opt": "random_bias",
    "node_bias_scale": 0.0,
    "w_in_opt": "random_sparse",
    "w_in_scale": 1.0,
    "x_train_noise_scale": 0.0,
    "reg_param": 1e-7,
    "ridge_regression_opt": "no_bias",
    "scale_input_bool": False,
}

# N ens:
n_ens = 50

# seeds:
seed = 1
rng = np.random.default_rng(seed)
seeds = rng.integers(0, 10000000, size=n_ens)
# Do experiment:

# Train (i.e. Drive):
# Train RC:
train_data = train_data_list[0]

# only one dim:
train_data = train_data[:, 0:1]

# Abruptstop of train data:
i_stop = 50
train_data[train_sync_steps + i_stop:, :] = 0
inp = train_data[train_sync_steps:-1, :]

results_rpca = []

for i in range(n_ens):
    # Build rc with network:
    esn_obj = esn.ESN()
    with utilities.temp_seed(seeds[i]):
        esn_obj.build(**build_args)

    # With network:
    fit, true, more_out = esn_obj.train(train_data, sync_steps=train_sync_steps, more_out_bool=True)
    res_states = more_out["r"]
    pca = PCA()
    res_pca_states_with = pca.fit_transform(res_states)
    results_rpca.append(res_pca_states_with)

dimensions = [0, 1, 5, 10, 50, 200, 300, 500]
nr_dims = len(dimensions)

# PLOT:
height = 500
width = int(1.4 * height)
font_size = 15
legend_font_size = 15
font_family = "Times New Roman"
linewidth = 1

yaxis_title = r"$r_\text{pca, i}$"
xaxis_title =  r'$\text{Time steps}$'

fig = make_subplots(rows=nr_dims + 1, cols=1, shared_yaxes=True,
                    shared_xaxes=True, vertical_spacing=None,
                    print_grid=True,
                    x_title=xaxis_title,
                    y_title=yaxis_title,
                    # row_heights=[height] * nr_taus,
                    # column_widths=[width],
                    subplot_titles=["input signal"] + [fr"$i = {x}$" for x in dimensions],
                    # horizontal_spacing=1,
                    # row_titles=[fr"$i = {x}$" for x in dimensions],
                    )
# add input signal

fig.add_trace(
    go.Scatter(y=inp[:, 0], name="input",
               line=dict(
                   width=linewidth
               ),
               mode="lines", showlegend=False,
               ),
    row=1, col=1
)

for i_d, d in enumerate(dimensions):
    color = hex_to_rgba(next(col_pal_iterator), 0.2)
    print(color)
    if i_d == 0:
        name_net = ""
        showlegend=True

    else:
        name_net = None
        showlegend = False

    for rpca in results_rpca:
        fig.add_trace(
            go.Scatter(y=rpca[:, i_d], name=name_net,
                       line=dict(
                           width=linewidth, color=color
                       ),
                       mode="lines", showlegend=showlegend,
                       ),
            row=i_d + 2, col=1
        )
# fig.update_yaxes(range=[0, 1])

fig.update_layout(template="plotly_white",
                  # showlegend=False,
                  showlegend=True,
                  font=dict(
                      size=font_size,
                      family=font_family
                  ),
                  )
fig.update_layout(
    margin=dict(l=70, r=20, t=20, b=50),
    showlegend=False,
)

fig.update_yaxes(showticklabels=False)

for i in range(nr_dims+1):
    fig.layout.annotations[i].update(x=0.1)

fig.update_layout(
    width=width,
    height=height,
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.1,
        xanchor="right",
        x=0.9,
        font=dict(size=legend_font_size)
    )
)

fig.add_vline(x=i_stop, row="all", line_width=1, line_dash="dash", line_color="black")


# SAVE
file_name = f"time_series_rpca_signal_fadeout_ens.png"
# file_name = f"intro_pca_traj_{name}.pdf"
fig.write_image(file_name, scale=3)

