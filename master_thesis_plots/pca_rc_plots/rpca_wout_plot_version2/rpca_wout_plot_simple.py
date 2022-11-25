"""Create Explained variance of pca states plot with error band.

NOT USED.
"""
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities

import itertools
import plotly.express as px
col_pal = px.colors.qualitative.Plotly
col_pal_iterator = itertools.cycle(col_pal)

def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return "rgba" + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))

# Create data:
sys_obj = sims.Lorenz63()
# sys_obj = sims.Logistic()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 5000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 1000,
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
    "x_dim": 3,
    "r_dim": 500,
    "n_rad": 0.4,
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
    # "ridge_regression_opt": "no_bias",
    "ridge_regression_opt": "bias",
    "scale_input_bool": False,
    "perform_pca_bool": True
}

reg_param = build_args["reg_param"]

# seeds:
seed = 1

# Do experiment:

# Build rc:
esn_obj = esn.ESN()
with utilities.temp_seed(seed):
    esn_obj.build(**build_args)

train_data = train_data_list[0]

fit, true, more_out = esn_obj.train(train_data,
                                    sync_steps=train_sync_steps,
                                    more_out_bool=True)

rfit_data = more_out["rfit"]

# center:
rfit_data = rfit_data - np.mean(rfit_data, axis=0)

# SVD:
u, s, v_t = np.linalg.svd(rfit_data, full_matrices=False)
d = np.diag(s)
v = v_t.T

# Transpose:
rfit_data = rfit_data.T
y_data = true.T

# Calculate wout with and without reg:
W_noreg = y_data @ u @ np.linalg.inv(d)

# with reg:
reg_series = s**2/(s**2 + reg_param)
reg_matrix = np.diag(reg_series)
W_reg = y_data @ u @ np.linalg.inv(d) @ reg_matrix

abs_w_noreg = np.abs(W_noreg)
abs_w_reg = np.abs(W_reg)

fig = go.Figure()
# for i in range(esn_obj.y_dim):
#     fig.add_trace(
#         go.Scatter(y=abs_w_reg[i, :],
#                    mode='none',
#                    stackgroup='one',
#                    name=fr"$j = {i+1}$",
#                    )
#     )

fig.add_trace(
    go.Scatter(y=np.sum(abs_w_reg, axis=0), name="reg"
               )
)

fig.add_trace(
    go.Scatter(y=np.sum(abs_w_noreg, axis=0), name="no reg"
               )
)
dy=1
fig.update_yaxes(range=( -dy, np.max(np.sum(abs_w_reg, axis=0) + dy)))

fig.add_trace(
    go.Scatter(y=reg_series,
              mode="lines")
)
fig.write_image(f"rpca_wout_test.png", scale=3)


# print(rfit_data.shape)
# print(true.shape)
#
# pca = PCA()
# pca.fit(rfit_data)
