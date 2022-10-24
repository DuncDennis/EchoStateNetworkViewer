"""Create the correlation plot between pca and input. """
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

# correlate:

# time delays:
tau_list = [0, 1, 2, 3, 4, 5]
inp = train_data[train_sync_steps:-1, :]

correlation_results = []

for tau in tau_list:
    correlation = meas.cross_correlate(inp, res_pca_states, tau)
    correlation_results.append(correlation)

# PLOT:
max_index = 30
height = 500
width = int(1.3 * height)
font_size = 15
legend_font_size = 20
font_family = "Times New Roman"

yaxis_title = r"$\text{absolute of } C[\boldsymbol{x}(i - \tau), \boldsymbol{r}_\text{pca}(i)]$"
xaxis_title =  r'$\text{Principal Component } \boldsymbol{p}_i$'

nr_taus = len(tau_list)
fig = make_subplots(rows=nr_taus, cols=1, shared_yaxes=True,
                    shared_xaxes=True, vertical_spacing=None,
                    print_grid=True,
                    x_title=xaxis_title,
                    y_title=yaxis_title,
                    # row_heights=[height] * nr_taus,
                    # column_widths=[width],
                    # subplot_titles=[str(x) for x in tau_list],
                    # row_titles=[str(x) for x in tau_list],
                    )

for i_tau, tau in enumerate(tau_list):

    corr = correlation_results[i_tau][:max_index, :]
    plot = np.sum(np.abs(corr), axis=1)
    x = np.arange(max_index)
    fig.add_trace(
        go.Bar(x=x, y=plot,
               name=fr"$\tau = {tau}$"
               ),
        row=i_tau+1, col=1
    )
fig.update_yaxes(range=[0, 1])

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
        # orientation="h",
        yanchor="top",
        y=1.01,
        xanchor="right",
        x=0.95,
        font=dict(size=legend_font_size)
    )
)

# SAVE
file_name = f"logistic_input_pca_correlation.png"
# file_name = f"intro_pca_traj_{name}.pdf"
fig.write_image(file_name, scale=3)

