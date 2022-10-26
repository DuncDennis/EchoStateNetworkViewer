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
    "n_rad": 0.2,
    "n_avg_deg": 5.0,
    "n_type_opt": "erdos_renyi",
    "r_to_rgen_opt": "linear_r",
    "act_fct_opt": "tanh",
    "node_bias_opt": "random_bias",
    "node_bias_scale": 0.2,
    "w_in_opt": "random_sparse",
    "w_in_scale": 1.0,
    "x_train_noise_scale": 0.0,
    "reg_param": 1e-10,
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
error_thresh = 0.2

# Train (i.e. Drive):
# Train RC:
train_data = train_data_list[0]

# only one dim:
train_data = train_data[:, 0:1]

# Prediction data:
validate_data = validate_data_list_of_lists[0][0]
# only one dim:
validate_data = validate_data[:, 0:1]

results_vt = np.zeros((n_ens, build_args["r_dim"]))
ts_vt = np.zeros(n_ens)
for i in range(n_ens):
    print(i)
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

    error_series_ts = meas.error_over_time(y_pred=pred,
                                           y_true=true_pred,
                                           normalization="root_of_avg_of_spacedist_squared")
    vt = meas.valid_time_index(error_series_ts, error_threshold=error_thresh)
    ts_vt[i] = vt

    pca_pred = pca.transform(more_out_pred["r"])

    # As comparison run reservoir with prediction input:
    # sync:  Warning: drive one step less, because during predict the previous r_state is saved.
    esn_obj.drive(validate_data[:pred_sync_steps - 1, :])
    # drive:
    r_pred_true = esn_obj.drive(validate_data[pred_sync_steps - 1:-1, :])

    pca_pred_true = pca.transform(r_pred_true)

    for i_pca in range(build_args["r_dim"]):
        error_series_pca = meas.error_over_time(y_pred=pca_pred[:, i_pca][:, np.newaxis],
                                                y_true=pca_pred_true[:, i_pca][:, np.newaxis],
                                                normalization="root_of_avg_of_spacedist_squared")
        vt = meas.valid_time_index(error_series_pca, error_threshold=error_thresh)
        results_vt[i, i_pca] = vt

results_vt_mean = np.mean(results_vt, axis=0)
ts_vt_mean = np.mean(ts_vt)

# PLOT:
height = 500
width = int(1.4 * height)
font_size = 15
legend_font_size = 15
font_family = "Times New Roman"
linewidth = 1

yaxis_title = r"$\text{Valid Time index}$"
xaxis_title =  r'$\text{PC}$'

fig = go.Figure()

# add input signal
color = hex_to_rgba(next(col_pal_iterator), 1.0)

fig.add_trace(
    go.Scatter(y=results_vt_mean, name="vt rpca",
               line=dict(
                   width=linewidth, color=color
               ),
               mode="lines", showlegend=True,
               )
)
fig.add_hline(y=ts_vt_mean)

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
file_name = f"rpca_train_predict_valid_time.png"
# file_name = f"intro_pca_traj_{name}.pdf"
fig.write_image(file_name, scale=3)


w_out = esn_obj.get_w_out()
w_out_mod = w_out @ pca.components_.T

w_out_to_plot = np.sum(np.abs(w_out_mod), axis=0)
fig = go.Figure()
fig.add_trace(
    go.Scatter(y=w_out_to_plot)
)
fig.write_image("w_out.png", scale=3)

