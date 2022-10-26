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
# sys_obj = sims.Logistic()
# sys_obj = sims.ComplexButterfly()
# sys_obj = sims.LinearSystem()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 200,
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
pred_sync_steps = ts_creation_args["t_validate_sync"]
train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                         **ts_creation_args)

# Build RC args:
build_args = {
    "x_dim": 1,
    "r_dim": 500,
    "n_rad": 0.5,
    "n_avg_deg": 5.0,
    "n_type_opt": "erdos_renyi",
    "r_to_rgen_opt": "linear_r",
    "act_fct_opt": "tanh",
    "node_bias_opt": "random_bias",
    "node_bias_scale": 0.2,
    "w_in_opt": "random_sparse",
    "w_in_scale": 1.0,
    "x_train_noise_scale": 0.0,
    "reg_param": 1e-7,
    "ridge_regression_opt": "no_bias",
    "scale_input_bool": False,
}

# N ens:
n_ens = 1

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

# Prediction data:
validate_data = validate_data_list_of_lists[0][0]
# only one dim:
validate_data = validate_data[:, 0:1]

for i in range(n_ens):
    # Build rc with network:
    esn_obj = esn.ESN()
    with utilities.temp_seed(seeds[i]):
        esn_obj.build(**build_args)

    # Train:
    fit, true, more_out = esn_obj.train(train_data,
                                        sync_steps=train_sync_steps,
                                        more_out_bool=True)
    res_states = more_out["r"]
    pca = PCA()
    res_pca_states_with = pca.fit_transform(res_states)

    # Predict:
    pred, true_pred, more_out_pred = esn_obj.predict(validate_data,
                                                     sync_steps=pred_sync_steps,
                                                     more_out_bool=True)

    pca_pred = pca.transform(more_out_pred["r"])

    # As comparison run reservoir with prediction input:
    # sync:  Warning: drive one step less, because during predict the previous r_state is saved.
    esn_obj.drive(validate_data[:pred_sync_steps - 1, :])
    # drive:
    r_pred_true = esn_obj.drive(validate_data[pred_sync_steps - 1:, :])

    pca_pred_true = pca.transform(r_pred_true)

dimensions = [0, 1, 10, 50, 300, 500]
# dimensions = []
nr_dims = len(dimensions)

# PLOT:
height = 500
width = int(1.4 * height)
font_size = 15
legend_font_size = 15
font_family = "Times New Roman"
linewidth = 0.6

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
pred_color = hex_to_rgba(next(col_pal_iterator), 1.0)
true_color = hex_to_rgba(next(col_pal_iterator), 1.0)

color_list = [pred_color, true_color]

for data, data_name, color in zip([pred, true_pred], ["pred", "true"], color_list):
    fig.add_trace(
        go.Scatter(y=data[:, 0], name=data_name,
                   line=dict(
                       width=linewidth, color=color
                   ),
                   mode="lines", showlegend=True,
                   ),
        row=1, col=1
    )

for i_d, d in enumerate(dimensions):
    #
    # # print(color)
    # if i_d == 0:
    #     name_net = ""
    #     showlegend=True
    #
    # else:
    #     name_net = None
    #     showlegend = False

    # for rpca in results_rpca:
    for data, data_name, color in zip([pca_pred, pca_pred_true],
                                      ["rpcapred", "rpcatrue"],
                                      color_list):
        fig.add_trace(
            go.Scatter(y=data[:, i_d], name=data_name,
                       line=dict(
                           width=linewidth,
                           color=color
                       ),
                       mode="lines", showlegend=False,
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

# fig.add_vline(x=i_stop, row="all", line_width=1, line_dash="dash", line_color="black")


# SAVE
file_name = f"rpca_train_predict_time_series.png"
# file_name = f"intro_pca_traj_{name}.pdf"
fig.write_image(file_name, scale=3)

