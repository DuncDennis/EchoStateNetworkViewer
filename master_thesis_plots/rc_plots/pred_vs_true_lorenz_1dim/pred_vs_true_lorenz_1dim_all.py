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
dt = 0.1
mle = 0.9059
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
    # "r_dim": 500,
    "r_dim": 60,
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
t_max = 100
x = np.arange(pred.shape[0])[:t_max] * dt * mle
# xaxis_title =  r'$\text{time } t \lambda_\text{max}$'
xaxis_title =  rf"$\{latex_text_size} " + r'\lambda_\text{max} \cdot t$'
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
    name = "True"
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
    name = "Predicted"
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
        title_standoff=5
    )

    fig.update_yaxes(
        showticklabels=False,
        row=i_x + 1,
        col=1,
    )

# Valid time:
    if i_x == 0:

        pre_text = r"t_\text{v} = "
        fig.add_vline(x=vt, line_width=3, line_dash="dash", line_color="black",
                      annotation_text="$" + pre_text + f"{vt}$",
                      annotation_font_size=20,
                      annotation_position="top",
                      row=i_x + 1,
                      col=1,
                      opacity=0.75
                      # line=dict(dash="dot")
                      )
    else:
        fig.add_vline(x=vt,
                      line_width=3,
                      line_dash="dash",
                      line_color="black",
                      row=i_x + 1,
                      col=1,
                      opacity=0.75
                      # line=dict(dash="dot")
                      )

fig.update_layout(
    template="simple_white",
    font=dict(
        size=18,
        family="Times New Roman"
    ),
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.2,
        xanchor="right",
        x=0.3,
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
file_name = f"pred_vs_true_lorenz_1dim_all.png"
fig.write_image(file_name, scale=3)

