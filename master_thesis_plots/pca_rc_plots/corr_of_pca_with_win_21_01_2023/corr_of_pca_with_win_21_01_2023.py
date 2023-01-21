"""Create the correlation plot between pca and input. """
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
import plotly.express as px

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities
import src.esn_src.measures as meas
col_pal = px.colors.qualitative.Plotly
col_pal_iterator = itertools.cycle(col_pal)

def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return "rgba" + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))

# Create data:
sys_obj = sims.Lorenz63(dt=0.1)
# sys_obj = sims.Logistic()
# sys_obj = sims.Lorenz96(sys_dim=10)
# sys_obj = sims.LinearSystem()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 2000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 2000,
                    "n_train_sects": 1,
                    "n_validate_sects": 1,
                    "normalize_and_center": False,
                    }

n_train = ts_creation_args["n_train_sects"]
train_sync_steps = ts_creation_args["t_train_sync"]
train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                         **ts_creation_args)
x_dim = sys_obj.sys_dim
# Build RC args:
build_args = {
    "x_dim": x_dim,
    "r_dim": 500,
    # "n_rad": 0.4,
    "n_rad": 0.0,
    "n_avg_deg": 5.0,
    "n_type_opt": "erdos_renyi",
    "r_to_rgen_opt": "linear_r",
    # "act_fct_opt": "tanh",
    "act_fct_opt": "linear",
    "node_bias_opt": "random_bias",
    "node_bias_scale": 0.0,
    # "w_in_opt": "random_sparse",
    # "w_in_opt": "ordered_sparse",
    # "w_in_opt": "random_dense_uniform",
    "w_in_opt": "random_sparse",
    "w_in_scale": 1.0,
    "x_train_noise_scale": 0.0,
    "reg_param": 1e-7,
    "ridge_regression_opt": "bias",
    "scale_input_bool": True,
    # "scale_input_bool": False,
}

x_dim = build_args["x_dim"]
r_dim = build_args["r_dim"]

# Ensemble size:
n_ens = 1

# seeds:
seed = 300
rng = np.random.default_rng(seed)
seeds = rng.integers(0, 10000000, size=n_ens)

# Do experiment:
for i_ens in range(n_ens):
    print(i_ens)

    # Build rc:
    esn_obj = esn.ESN()
    with utilities.temp_seed(seeds[i_ens]):
        esn_obj.build(**build_args)

    for i_train in range(n_train):
        # Train RC:
        train_data = train_data_list[i_train]
        inp = train_data[train_sync_steps:-1, :]
        _, _, more_out = esn_obj.train(train_data, sync_steps=train_sync_steps, more_out_bool=True)

        # states and w_in:
        res_states = more_out["r"]
        x_proc_states = more_out["xproc"]
        w_in = esn_obj.w_in

        # SVD tests:
        res_states_cent = res_states - np.mean(res_states)  # shape N x r_dim
        u_res, d_res, p_res = np.linalg.svd(res_states_cent, full_matrices=False)
        d_res = np.diag(d_res)

        u_inp, d_inp, p_inp = np.linalg.svd(x_proc_states, full_matrices=False)
        d_inp = np.diag(d_inp)

        # reservoir components
        pca = PCA()
        res_pca_states = pca.fit_transform(res_states)
        components = pca.components_ # n_components, n_features

        # input components:
        inp_pca = PCA()
        inp_pca_states = inp_pca.fit_transform(x_proc_states)
        inp_comps = inp_pca.components_

        if i_train == 0 and i_ens == 0:
            # explained variances:
            n_components = components.shape[0]

            w_in_results = np.zeros((n_ens, n_train, r_dim, x_dim))
            x_pca_results = np.zeros((n_ens, n_train, x_dim, x_dim))
            r_pca_results = np.zeros((n_ens, n_train, r_dim, r_dim))
            px_w_in_results = np.zeros((n_ens, n_train, x_dim, r_dim))
            # correlation_results = np.zeros((n_ens, n_train, n_components, x_dim))

        w_in_results[i_ens, i_train, :, :] = w_in
        x_pca_results[i_ens, i_train, :, :] = inp_comps
        r_pca_results[i_ens, i_train, :, :] = components
        px_w_in_results[i_ens, i_train, :, :] = inp_comps @ w_in.T
        # px_w_in_results[i_ens, i_train, :, :] = w_in.T

a = np.linalg.inv(d_res) @ u_res.T @ u_inp @ d_inp @ p_inp @ w_in.T
# a = np.linalg.inv(d_res) @ u_res.T @ u_inp @ d_inp @ p_inp @ w_in.T
# test:
# r_cent = res_states - np.mean(res_states, axis=0)
# res_states

fig = go.Figure()

for i in range(x_dim):
    fig.add_trace(
        go.Scatter(
            # x=px_w_in_results[0, 0, i, :],
            x=a[i, :],
            # y=r_pca_results[0, 0, i, :],
            y=p_res[i, :],
            mode="markers",
            name=f"{i}"
        )
    )

# SAVE
file_name = f"test.png"
fig.write_image(file_name, scale=3)
