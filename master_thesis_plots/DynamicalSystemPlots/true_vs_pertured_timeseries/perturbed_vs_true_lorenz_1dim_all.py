"""RC plot: train vs predict trajectory lorenz """
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities
import src.esn_src.measures as meas

predicted_color = '#EF553B'  # red
true_color = '#636EFA'  # blue

# def hex_to_rgba(h, alpha):
#     '''
#     converts color value in hex format to rgba format with alpha transparency
#     '''
#     return tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha])
#
# predicted_color = hex_to_rgba(predicted_color, 0.5)  # opacity
# true_color = hex_to_rgba(true_color, 0.5)  # opacity

# Create data:
# dt = 0.1
mle = 0.9059


# Lorenz:
sigma = 10.0
rho = 28.0
beta = 8 / 3
dt = 0.1
sys_obj = sims.Lorenz63(dt=dt,
                        sigma=sigma,
                        rho=rho,
                        beta=beta)

# initial condition:
skip_steps = 1000
initial_condition_new = sys_obj.simulate(time_steps=skip_steps)[-1, :]

# perturbed initial condition:
pert_scale = 1e-4
# pert_init_cond = initial_condition_new + pert_scale
pert_init_cond = initial_condition_new + np.array([pert_scale, 0, 0])

steps = 2000

# Baseline traj:
base_traj = sys_obj.simulate(steps, starting_point=initial_condition_new)

# Perturbed traj:
pert_traj = sys_obj.simulate(steps, starting_point=pert_init_cond)


true_pred = base_traj

# eps kbm prediction:
pred = pert_traj



error_series_ts = meas.error_over_time(y_pred=pred,
                                       y_true=true_pred,
                                       normalization="root_of_avg_of_spacedist_squared")
vt = meas.valid_time_index(error_series_ts, error_threshold=0.4)
vt = mle * dt * vt
vt = np.round(vt, 1)

latex_text_size = "large"
# latex_text_size = "Large"
# latex_text_size = "huge"
# latex_text_size = "normalsize"
linewidth = 3
height = 350
width = int(2 * height)
# width = 500

dim = 0
t_max = 250
x = np.arange(pred.shape[0])[:t_max] * dt * mle
# xaxis_title =  r'$\text{time } t \lambda_\text{max}$'
# xaxis_title =  rf"$\{latex_text_size} " + r'\lambda_\text{max} \cdot t$'
xaxis_title =  rf"$\{latex_text_size} " + r't \cdot \lambda_\text{max}$'
fig = make_subplots(rows=3,
                    cols=1,
                    shared_xaxes=True,
                    vertical_spacing=None,
                    print_grid=True,
                    x_title=xaxis_title,
                    # y_title=yaxis_title,
                    # row_heights=[height] * nr_taus,
                    # column_widths=[width],
                    # subplot_titles=[str(x) for x in tau_list],
                    # row_titles=[str(x) for x in tau_list],
                    )



index_to_dimstr = {0: rf"$\{latex_text_size} " + r"x(t)$",
                   1: rf"$\{latex_text_size} " + r"y(t)$",
                   2: rf"$\{latex_text_size} " + r"z(t)$",}

for i_x in range(3):
    dimstr = index_to_dimstr[i_x]
    # fig.update_yaxes(title_text=dimstr, index=i_x)
    if i_x == 0:
        showlegend=True
    else:
        showlegend=False
    # TRUE:
    name = "basis"
    # true_color = "black"
    y = true_pred[:t_max, i_x]
    fig.add_trace(
        go.Scatter(x=x, y=y,
                   line=dict(
                       color=true_color,
                       width=linewidth,
                   ),
                   showlegend=showlegend,
                   name=name,
                   mode="lines"),
        row=i_x + 1, col=1
    )

    # TRUE:
    # name = "Predicted (ϵ- model)"
    # name = "Perturbed-traj"
    name = r"perturbed"
    # predicted_color = "red"
    # predicted_color = '#EF553B'
    y = pred[:t_max, i_x]
    fig.add_trace(
        go.Scatter(x=x, y=y,
                   line=dict(
                       color=predicted_color,
                       width=linewidth,
                       # dash="dot"
                   ),
                   showlegend=showlegend,
                   name=name,
                   mode="lines"),
        row=i_x + 1, col=1
    )

    fig.update_yaxes(
        title_text=dimstr,
        row=i_x + 1,
        col=1,
        title_standoff=5,
    )

    fig.update_yaxes(
        # showticklabels=False,
        showticklabels=True,
        row=i_x + 1,
        col=1,
    )

    fig.update_xaxes(
        range=[0, np.max(x)],
        row=i_x + 1,
        col=1,
    )

fig.update_yaxes(
    tick0 = -15,
    dtick = 15,
    row=1,
    col=1,
)

fig.update_yaxes(
    tick0 = -20,
    dtick = 20,
    row=2,
    col=1,
)

fig.update_yaxes(
    tick0 = 10,
    dtick = 30,
    row=3,
    col=1,
)

# # Valid time:
#     if i_x == 0:
#
#         # pre_text = r"t_\text{v} = "
#         pre_text = r"t_\text{v}\lambda_\text{max} = "
#         fig.add_vline(x=vt, line_width=3, line_dash="dash", line_color="black",
#                       annotation_text="$" + pre_text + f"{vt}$",
#                       annotation_font_size=20,
#                       annotation_position="top",
#                       annotation_xshift = 10,
#                       # annotation_xshift = 0,
#                       annotation_yshift = 5,
#                       row=i_x + 1,
#                       col=1,
#                       opacity=0.75
#                       # line=dict(dash="dot")
#                       )
#     else:
#         fig.add_vline(x=vt,
#                       line_width=3,
#                       line_dash="dash",
#                       line_color="black",
#                       row=i_x + 1,
#                       col=1,
#                       opacity=0.75
#                       # line=dict(dash="dot")
#                       )

fig.update_layout(
    template="simple_white",
    font=dict(
        size=18,
        family="Times New Roman"
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.25,
        xanchor="right",
        x=1,
        font=dict(size=20)
    )
)


fig.update_layout(
    width=width,
    height=height,
)
fig.update_layout(
    margin=dict(l=15, r=40, t=10, b=50),
)

# SAVE
file_name = f"perturbed_vs_true_lorenz_1dim_all.png"
fig.write_image(file_name, scale=3)
