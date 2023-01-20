"""
File that plots the w_out matrix of a hybrid-rc setup, in order to see the effect of
output-hybrid.

- See how the wout distribution changes if output hybrid is added:
"""
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities
import src.esn_src.measures as meas


# System:
# Lorenz:
# sigma = 10.0
# rho = 28.0
# beta = 8 / 3
# dt = 0.1
# sys_obj = sims.Lorenz63(dt=dt,
#                         sigma=sigma,
#                         rho=rho,
#                         beta=beta)

# sys_obj = sims.Thomas(dt=0.4, b=0.18)
# sys_obj = sims.Chen(dt=0.03)
# sys_obj = sims.WindmiAttractor(dt=0.2)
# sys_obj = sims.Halvorsen(dt=0.05)
sys_obj = sims.ChuaCircuit(dt=0.2)


# Hybrid model:

# epsilon model:
# eps = 0.1
# eps = 100
# # eps = 1e-4
# wrong_model = sims.Lorenz63(dt=dt,
#                             sigma=sigma,
#                             rho=rho*(1+eps),
#                             beta=beta).iterate

# Flow model:
# wrong_model = sys_obj.flow

# dim-selection model:
dim_select = 2
wrong_model = lambda x: sys_obj.iterate(x)[[dim_select]]

# sinus model:
# def wrong_model(x):
#     return np.sin(x)

kbm_dim = wrong_model(np.array([0, 0, 0])).size

# Time steps:
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 2000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 1000,
                    "n_train_sects": 15,
                    "n_validate_sects": 10, # keep it at 1 here.
                    "normalize_and_center": False,
                    }

n_train = ts_creation_args["n_train_sects"]
train_sync_steps = ts_creation_args["t_train_sync"]
pred_sync_steps = ts_creation_args["t_validate_sync"]
train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                         **ts_creation_args)
# x_dim:
x_dim = sys_obj.sys_dim

# No hybrid build RC args:
build_args = {
    "x_dim": x_dim,
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
    "x_train_noise_scale": 0.0, # 1e-6,
    "reg_param": 1e-7,
    "ridge_regression_opt": "bias",
    # "ridge_regression_opt": "bias",
    "scale_input_bool": True,
    "perform_pca_bool": False,
    "scale_input_model_bool": True,
    # "input_model": wrong_model,
    # "output_model": wrong_model,
}

# hybrid build args:
hybrid_build_args = build_args.copy()
# Input hybrid:
# hybrid_build_args["input_model"] = wrong_model
# Output hybrid:
hybrid_build_args["output_model"] = wrong_model


# Ensemble size:
n_ens = 15

# seeds:
seed = 300
rng = np.random.default_rng(seed)
seeds = rng.integers(0, 10000000, size=n_ens)

# Do experiment:
for i_ens in range(n_ens):
    print(i_ens)

    # # Build normal rc:
    # esn_obj = esn.ESNHybrid()
    # with utilities.temp_seed(seeds[i_ens]):
    #     esn_obj.build(**build_args)

    # Build hybrid rc:
    esn_hyb_obj = esn.ESNHybrid()
    with utilities.temp_seed(seeds[i_ens]):
        esn_hyb_obj.build(**hybrid_build_args)

    for i_train in range(n_train):
        train_data = train_data_list[i_train]

        # Train normal RC:
        # _, _ = esn_obj.train(train_data,
        #                         sync_steps=train_sync_steps,
        #                         more_out_bool=False)

        # Train hybrid RC:
        _, _ = esn_hyb_obj.train(train_data,
                                    sync_steps=train_sync_steps,
                                    more_out_bool=False)

        # Get wout matrices:
        # w_out_normal = esn_obj.get_w_out()  # shape: y_dim, rfit_dim
        w_out_hybrid = esn_hyb_obj.get_w_out()  # shape: y_dim, rfit_dim

        if i_train == 0 and i_ens == 0:
            # res_w_out_normal = np.zeros((n_ens,
            #                              n_train,
            #                              w_out_normal.shape[0],
            #                              w_out_normal.shape[1]))
            res_w_out_hybrid = np.zeros((n_ens,
                                         n_train,
                                         w_out_hybrid.shape[0],
                                         w_out_hybrid.shape[1]))


        # res_w_out_normal[i_ens, i_train, :, :] = w_out_normal
        res_w_out_hybrid[i_ens, i_train, :, :] = w_out_hybrid

# Get absolute value of w_out and calculate median:
# Normal:
# abs_res_w_out_normal = np.abs(res_w_out_normal)
# median_abs_res_w_out_normal = np.median(abs_res_w_out_normal, axis=(0, 1))

# Hybrid:
abs_res_w_out_hybrid = np.abs(res_w_out_hybrid)

# split to reservoir contribution:
abs_w_out_hyb_res = abs_res_w_out_hybrid[:, :, :, 0: -1 - kbm_dim]

# split hybrid contribution
abs_w_out_hyb_kbm = abs_res_w_out_hybrid[:, :, :, build_args["r_dim"]: -1]

abs_w_out_hyb_res_avg = np.mean(abs_w_out_hyb_res, axis=3)
abs_w_out_hyb_kbm_avg = np.mean(abs_w_out_hyb_kbm, axis=3)

median_abs_w_out_hyb_res_avg = np.median(abs_w_out_hyb_res_avg, axis=(0, 1))
median_abs_w_out_hyb_kbm_avg = np.median(abs_w_out_hyb_kbm_avg, axis=(0, 1))

low_abs_w_out_hyb_res_avg = np.quantile(abs_w_out_hyb_res_avg, q=0.25, axis=(0, 1))
low_abs_w_out_hyb_kbm_avg = np.quantile(abs_w_out_hyb_kbm_avg, q=0.25, axis=(0, 1))

high_abs_w_out_hyb_res_avg = np.quantile(abs_w_out_hyb_res_avg, q=0.75, axis=(0, 1))
high_abs_w_out_hyb_kbm_avg = np.quantile(abs_w_out_hyb_kbm_avg, q=0.75, axis=(0, 1))


# # Normal:
# fig = go.Figure()
# fig.add_trace(
#     go.Bar(x=np.arange(x_dim),
#            y=w_out_normal_avg)
# )
# fig.update_layout(title="normal")
# fig.show()

x = np.arange(x_dim) + 1
# Hybrid:
fig = go.Figure()
fig.add_traces([
    go.Bar(x=x,
           y=median_abs_w_out_hyb_res_avg,
           error_y=dict(symmetric=False,
                        thickness=2,
                        width=6,
                        array=high_abs_w_out_hyb_res_avg - median_abs_w_out_hyb_res_avg,
                        arrayminus=median_abs_w_out_hyb_res_avg - low_abs_w_out_hyb_res_avg),
           marker_color="#019355",
           name="Reservoir"
           ),
    go.Bar(x=x,
           y=median_abs_w_out_hyb_kbm_avg,
           error_y=dict(symmetric=False,
                        thickness=2,
                        width=6,
                        array=high_abs_w_out_hyb_kbm_avg - median_abs_w_out_hyb_kbm_avg,
                        arrayminus=median_abs_w_out_hyb_kbm_avg - low_abs_w_out_hyb_kbm_avg),
           marker_color="#F79503",
           name="KBM"
           )
])

# add hor lines:
for x_val in x[:-1]:
    fig.add_vline(x=x_val + 0.5,
                  line_width=1,
                  line_dash="dash",
                  line_color="black",
                  opacity=0.5,
                  )

fig.update_layout(
    # title="hybrid",
    barmode="group",
    template="simple_white",
    showlegend=True,
    # width=600,
    width=400,
    height=300,
    # yaxis_title=r"$\large \text{partial } |\text{W}_\text{out}|$",
    yaxis_title=r"$\large |\Omega_{\text{res} / \text{kbm},\, j}|$",
    xaxis_title=r"$\large \text{output dimension } j$",

    font=dict(
        size=23,
        family="Times New Roman"
    ),
    # yaxis=dict(dtick=0.4),  # use for thomas sinus.
    legend=dict(
        orientation="h",
        yanchor="top",
        y=1.3,
        xanchor="right",
        x=0.99,
        font=dict(size=23),
        entrywidth=0,
        bordercolor="grey",
        borderwidth=2,
    ),
    margin=dict(l=20, r=20, t=20, b=20),
)

# SAVE
fig.write_image(f"hybrid_wout_plot.png", scale=2.5)

# PLOT:

# w_out = median_abs_res_w_out_normal
# w_out = median_abs_res_w_out_hybrid
# fig = go.Figure()
#
# y_dim, rfit_dim = w_out.shape
# x = np.arange(rfit_dim)
# for i in range(x_dim):
#     fig.add_trace(
#         go.Bar(x=x, y=w_out[i, :])
#     )
#
# fig.update_layout(barmode="stack")
# fig.show()
