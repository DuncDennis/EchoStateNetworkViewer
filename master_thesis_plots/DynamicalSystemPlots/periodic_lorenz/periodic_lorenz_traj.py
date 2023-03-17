"""RC plot: train vs predict trajectory lorenz """
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities


def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return "rgba" + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))


normal_color = '#F37735'  # red
periodic_color = '#3DA4AB'  # blue
# alpha = 0.5
normal_color = hex_to_rgba(normal_color, 0.9)
periodic_color = hex_to_rgba(periodic_color, 1.0)


# Create data:
# dt = 0.1
# mle = 0.9059


# Lorenz chaotic:
sigma = 10.0
rho = 28.0
beta = 8 / 3
dt = 0.1
sys_obj = sims.Lorenz63(dt=dt,
                        sigma=sigma,
                        rho=rho,
                        beta=beta)

# Lorenz periodic:
sigma = 10.0
rho = 28.0
beta = 0.2
dt = 0.1
sys_obj_periodic = sims.Lorenz63(dt=dt,
                                 sigma=sigma,
                                 rho=rho,
                                 beta=beta)

# initial condition:
skip_steps = 5000
# initial_condition_new = sys_obj.simulate(time_steps=skip_steps)[-1, :]


steps = 2000

# Baseline traj:
chaotic_traj = sys_obj.simulate(steps + skip_steps)[skip_steps:, :]

# Perturbed traj:
perioidic_traj = sys_obj_periodic.simulate(steps + skip_steps)[skip_steps:, :]



linewidth = 2
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
name = "chaotic input"
x = chaotic_traj[:-500, 0].tolist()
y = chaotic_traj[:-500, 1].tolist()
z = chaotic_traj[:-500, 2].tolist()
fig.add_trace(
    go.Scatter3d(x=x, y=y, z=z,
                 line=dict(
                     color=normal_color,
                     width=1
                 ),
                 name=name,
                 mode="lines",
                 meta=dict())
)

# name = "Predicted"
name = "periodic input"
x = perioidic_traj[:, 0].tolist()
y = perioidic_traj[:, 1].tolist()
z = perioidic_traj[:, 2].tolist()
fig.add_trace(
    go.Scatter3d(x=x, y=y, z=z,
                 line=dict(
                     width=linewidth,
                     color=periodic_color,
                     # dash="dot"
                 ),
                 name=name,
                 mode="lines",
                 meta=dict())
)



fig.update_layout(template="simple_white",
                  showlegend=True,
                  font=dict(
                      size=18,
                      family="Times New Roman"
                  ),
                  legend=dict(
                      orientation="v",
                      yanchor="top",
                      y=0.9,
                      xanchor="right",
                      x=0.8,
                      font=dict(size=20),
                      bordercolor="grey",
                      borderwidth=2,
                      itemsizing="constant"
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

# fig.update_layout(
#     scene=dict(
#         xaxis=dict(range=[-200, 200]),
#         yaxis=dict(range=[-200, 200]),
#         zaxis=dict(range=[-200, 50]),
#     )
# )

fig.update_layout(
    # margin=dict(l=5, r=5, t=5, b=5),
    margin=dict(l=0, r=0, t=0, b=0),
)


print(fig.layout.scene)

# SAVE
# fig.write_image("intro_expl_var_w_error.pdf", scale=3)
file_name = f"periodic_lorenz_traj.png"
# file_name = f"intro_pca_traj_{name}.pdf"
fig.write_image(file_name, scale=3)

