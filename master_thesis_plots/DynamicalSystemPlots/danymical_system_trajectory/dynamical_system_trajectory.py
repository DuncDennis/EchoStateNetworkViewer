"""Create the PCA intro plot. """
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities

N = 10000

time_series = sims.Lorenz63(dt=0.05).simulate(time_steps=N)
systemname = "Lorenz"

color = "black"
linewidth = 0.5
height = 500
# width = int(1.4 * height)
width = 500

# Input:
name = "input"
x = time_series[:, 0].tolist()
y = time_series[:, 1].tolist()
z = time_series[:, 2].tolist()

# LORENZ:
cx, cy, cz = 1.25, -1.25, 1.25
f = 1.2


camera = dict(eye=dict(x=f * cx,
                       y=f * cy,
                       z=f * cz))

# PCA transformed input:
# name = "pca_input"
# x = true_pca[:, 0].tolist()
# y = true_pca[:, 1].tolist()
# z = true_pca[:, 2].tolist()

# PCA reservoir 1-2-3:
# name = "pca_res_first"
# x = res_pca_states[:, 0].tolist()
# y = res_pca_states[:, 1].tolist()
# z = res_pca_states[:, 2].tolist()
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
                 mode="lines",
                 meta=dict())
)
fig.update_layout(
    # template="plotly_white",
    template="simple_white",
    showlegend=False,
)

fig.update_scenes(
    # xaxis_title="",
    # yaxis_title="",
    # zaxis_title="",

    # xaxis_showticklabels=False,
    # yaxis_showticklabels=False,
    # zaxis_showticklabels=False
)

fig.update_layout(scene_camera=camera,
                  width=width,
                  height=height,
                  )
fig.update_layout(
    # margin=dict(l=20, r=20, t=20, b=20),
    # margin=dict(l=0, r=20, t=20, b=35),
    # margin=dict(l=5, r=5, t=5, b=5),
    margin=dict(l=0, r=0, t=0, b=0),
)

# SAVE
# fig.write_image("intro_expl_var_w_error.pdf", scale=3)
# file_name = f"dynamical_system_trajectory.png"
file_name = f"dynamical_system_trajectory_{systemname}.pdf"
# file_name = f"intro_pca_traj_{name}.pdf"
# fig.write_image(file_name, scale=3)
fig.write_image(file_name, scale=3, format="pdf")

