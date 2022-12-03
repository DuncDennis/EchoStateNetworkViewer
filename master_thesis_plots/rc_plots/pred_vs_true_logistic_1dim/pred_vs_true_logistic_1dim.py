"""RC plot: train vs predict trajectory lorenz """
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities
import src.esn_src.measures as meas

# Create data:
sys_obj = sims.Logistic()

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
    "x_dim": 1,
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
seed = 1

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


linewidth = 3
height = 350
width = int(2 * height)
# width = 500

dim = 0
t_max = 100
x = np.arange(pred.shape[0])[:t_max]
fig = go.Figure()

# TRUE:
name = "True"
y = true_pred[:t_max, dim]
fig.add_trace(
    go.Scatter(x=x, y=y,
                 line=dict(
                     # color=color,
                     width=linewidth
                 ),
                 name=name,
                 mode="lines")
)

# TRUE:
name = "Predicted"
y = pred[:t_max, dim]
fig.add_trace(
    go.Scatter(x=x, y=y,
                 line=dict(
                     # color=color,
                     width=linewidth
                 ),
                 name=name,
                 mode="lines")
)

# Valid time:
print(vt)
pre_text = r"t_\text{v} = "
fig.add_vline(x=vt, line_width=3, line_dash="dash", line_color="black",
              annotation_text="$" + pre_text + f"{vt}$",
              annotation_font_size=20,
              annotation_position="bottom left"
              # line=dict(dash="dot")
              )

fig.update_layout(
    template="simple_white",
    # template="plotly_white",
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
        x=0.7,
        font=dict(size=15)
    )
)

x_axis_title = r"$\text{time steps}$"
y_axis_title = r"$x(t)$"
fig.update_xaxes(title=x_axis_title)
fig.update_yaxes(title=y_axis_title,
                 showgrid=True,
                 gridwidth=1,
                 gridcolor="black",
                 # title_font=dict(size=5),
                 title_standoff = 0
                 )

fig.update_layout(
    width=width,
    height=height,
)
fig.update_layout(
    # margin=dict(l=5, r=5, t=5, b=5),
)

# SAVE
file_name = f"pred_vs_true_logistic_1dim.png"
fig.write_image(file_name, scale=3)

