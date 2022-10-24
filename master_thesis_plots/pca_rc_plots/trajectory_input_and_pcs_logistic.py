"""Create the PCA intro plot. """
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.esn_src.esn_new_develop as esn
import src.esn_src.data_preprocessing as datapre
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities

# Create data:
sys_obj = sims.Logistic()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 3000,
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

# Build rc:
esn_obj = esn.ESN()
with utilities.temp_seed(seed):
    esn_obj.build(**build_args)

# Train (i.e. Drive):
# Train RC:
train_data = train_data_list[0]

fit, true, more_out = esn_obj.train(train_data, sync_steps=train_sync_steps, more_out_bool=True)
res_states = more_out["r"]
pca = PCA()
res_pca_states = pca.fit_transform(res_states)

pca_input = PCA()
true_pca = pca_input.fit_transform(true)


color = "black"
linewidth = 0.5
markersize = 2
height = 500
# width = int(1.4 * height)
width = 500
mode="markers" # "lines"


camera = None

# Input:
# name = "input"
# x = true[:, 0].tolist()
# y = true[:, 1].tolist()
# z = true[:, 2].tolist()
# camera = dict(eye=dict(x=1.25, y=-1.25, z=1.25))

# Input with embedding:
# true_embed = datapre.embedding(true, embedding_dimension=2)
# name = "input_embed"
# x = true_embed[:, 0].tolist()
# y = true_embed[:, 1].tolist()
# z = true_embed[:, 2].tolist()
camera = dict(eye=dict(x=1.25, y=-1.25, z=1.25))

# PCA reservoir 1-2-3:
name = "pca_res_first"
x = res_pca_states[:, 0].tolist()
y = res_pca_states[:, 1].tolist()
z = res_pca_states[:, 2].tolist()
# camera = dict(eye=dict(x=0.8, y=0.9, z=1.25),
#               up=dict(x=0, y=1, z=0))

# PCA reservoir 3-4-5:
# name = "pca_res_later"
# x = res_pca_states[:, 3].tolist()
# y = res_pca_states[:, 4].tolist()
# z = res_pca_states[:, 5].tolist()
# camera = dict(eye=dict(x=0.8, y=0.9, z=1.25),
#               up=dict(x=0, y=1, z=0))

fig = go.Figure()
fig.add_trace(
    go.Scatter3d(x=x, y=y, z=z,
                 line=dict(
                     color=color,
                     width=linewidth
                 ),
                 marker=dict(size=markersize),
                 mode=mode,
                 meta=dict())
)
fig.update_layout(template="plotly_white",
                  showlegend=False,
                  )

fig.update_scenes(
    xaxis_title="",
    yaxis_title="",
    zaxis_title="",

    xaxis_showticklabels=False,
    yaxis_showticklabels=False,
    zaxis_showticklabels=False
)

fig.update_layout(scene_camera=camera,
                  width=width,
                  height=height,
                  )
fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)

# SAVE
# fig.write_image("intro_expl_var_w_error.pdf", scale=3)
file_name = f"logistic_pca_traj_{name}.png"
# file_name = f"intro_pca_traj_{name}.pdf"
fig.write_image(file_name, scale=3)

