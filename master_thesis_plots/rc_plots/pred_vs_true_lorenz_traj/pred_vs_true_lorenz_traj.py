"""RC plot: train vs predict trajectory lorenz """
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities

predicted_color = '#EF553B'  # red
true_color = '#636EFA'  # blue

# Create data:
dt = 0.1
sys_obj = sims.Lorenz63(dt=dt)

ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 1000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 1000,
                    "n_train_sects": 1,
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

# seeds:
seed = 2

# Do experiment:

# Build rc:
esn_obj = esn.ESN()
with utilities.temp_seed(seed):
    esn_obj.build(**build_args)

# Train RC:
train_data = train_data_list[0]
fit, true, more_out = esn_obj.train(train_data, sync_steps=train_sync_steps, more_out_bool=True)

pred_data = validate_data_list_of_lists[0][0]
print("pred_data shape: ", pred_data.shape)
pred, true_pred, more_out_pred = esn_obj.predict(pred_data,
                                                 sync_steps=pred_sync_steps,
                                                 more_out_bool=True)


linewidth = 3
height = 500
# width = int(1.4 * height)
width = 500



# LORENZ:
# cx, cy, cz = 1.25, -1.25, 1.25
# cx, cy, cz = 1.25, -1.25, 1.0
cx, cy, cz = 1.25, -1.25, 0.5
# f = 1.2
f = 1.5
camera = dict(eye=dict(x=f * cx,
                       y=f * cy,
                       z=f * cz))

fig = go.Figure()

# TRUE:
name = "True"
x = true_pred[:, 0].tolist()
y = true_pred[:, 1].tolist()
z = true_pred[:, 2].tolist()
fig.add_trace(
    go.Scatter3d(x=x, y=y, z=z,
                 line=dict(
                     color=true_color,
                     width=linewidth
                 ),
                 name=name,
                 mode="lines",
                 meta=dict())
)

# TRUE:
name = "Predicted"
x = pred[:, 0].tolist()
y = pred[:, 1].tolist()
z = pred[:, 2].tolist()
fig.add_trace(
    go.Scatter3d(x=x, y=y, z=z,
                 line=dict(
                     width=linewidth,
                     color=predicted_color,
                     # dash="dot"
                 ),
                 name=name,
                 mode="lines",
                 meta=dict())
)

fig.update_layout(template="simple_white",
                  # showlegend=False,
                  font=dict(
                      size=18,
                      family="Times New Roman"
                  ),
                  legend=dict(
                      orientation="h",
                      yanchor="top",
                      y=0.8,
                      xanchor="right",
                      x=0.6,
                      font=dict(size=20)
                  ),
                  # xaxis_title=r"$test$"
                  )


fig.update_scenes(
    # xaxis_title=r"$x(t)$",
    # yaxis_title=r"$y(t)$",
    # zaxis_title=r"$z(t)$",

    xaxis_showticklabels=False,
    yaxis_showticklabels=False,
    zaxis_showticklabels=False
)



fig.update_layout(scene_camera=camera,
                  width=width,
                  height=height,
                  )
fig.update_layout(
    # margin=dict(l=5, r=5, t=5, b=5),
    margin=dict(l=0, r=0, t=0, b=0),
)

# SAVE
# fig.write_image("intro_expl_var_w_error.pdf", scale=3)
file_name = f"pred_vs_true_lorenz_traj.png"
# file_name = f"intro_pca_traj_{name}.pdf"
fig.write_image(file_name, scale=3)

