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

# Create data:
sys_obj = sims.Lorenz63()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 1500,
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
    "n_rad": 0.9,
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

# seeds:
seed = 1

# Do experiment:

# Build rc with network:
esn_obj = esn.ESN()
with utilities.temp_seed(seed):
    esn_obj.build(**build_args)

# Build rc without network:
esn_obj_no_net = esn.ESN()
build_args["n_rad"] = 0
with utilities.temp_seed(seed):
    esn_obj_no_net.build(**build_args)

# Train (i.e. Drive):
# Train RC:
train_data = train_data_list[0]

# With network:
fit, true, more_out = esn_obj.train(train_data, sync_steps=train_sync_steps, more_out_bool=True)
res_states = more_out["r"]
pca = PCA()
res_pca_states_with = pca.fit_transform(res_states)

# Without network:
fit, true, more_out = esn_obj_no_net.train(train_data, sync_steps=train_sync_steps, more_out_bool=True)
res_states = more_out["r"]
pca = PCA()
res_pca_states_without = pca.fit_transform(res_states)

dimensions = [0, 1, 2, 5, 10]
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

fig = make_subplots(rows=nr_dims, cols=1, shared_yaxes=True,
                    shared_xaxes=True, vertical_spacing=None,
                    print_grid=True,
                    x_title=xaxis_title,
                    y_title=yaxis_title,
                    # row_heights=[height] * nr_taus,
                    # column_widths=[width],
                    subplot_titles=[fr"$i = {x}$" for x in dimensions],
                    # horizontal_spacing=1,
                    # row_titles=[fr"$i = {x}$" for x in dimensions],
                    )

for i_d, d in enumerate(dimensions):
    color_net = "red"
    color_nonet = "blue"
    if i_d == 0:
        name_net = "network"
        name_nonet = "no network"
        showlegend=True

    else:
        name_net, name_nonet = None, None
        showlegend = False


    fig.add_trace(
        go.Scatter(y=res_pca_states_with[:, i_d], name=name_net,
                   line=dict(
                       width=linewidth, color=color_net
                   ),
                   mode="lines", showlegend=showlegend,
                   ),
        row=i_d + 1, col=1
    )
    fig.add_trace(
        go.Scatter(y=res_pca_states_without[:, i_d], name=name_nonet,
                   line=dict(
                       width=linewidth, color=color_nonet
                   ), showlegend=showlegend,
                   mode="lines",
                   ),
        row=i_d + 1, col=1
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

for i in range(nr_dims):
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



# SAVE
file_name = f"time_series_rpca_network_influence.png"
# file_name = f"intro_pca_traj_{name}.pdf"
fig.write_image(file_name, scale=3)

