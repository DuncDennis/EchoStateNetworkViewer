"""Create Explained variance of pca states plot with error band."""
import numpy as np
from sklearn.decomposition import PCA
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import src.esn_src.esn_new_develop as esn
import src.esn_src.simulations as sims
import src.ensemble_src.sweep_experiments as sweep
import src.esn_src.utilities as utilities
import src.esn_src.measures as meas


import itertools
import plotly.express as px

# Color cycle:
col_pal = px.colors.qualitative.Plotly
col_pal_iterator = itertools.cycle(col_pal)

def hex_to_rgba(h, alpha):
    '''
    converts color value in hex format to rgba format with alpha transparency
    '''
    return "rgba" + str(tuple([int(h.lstrip('#')[i:i+2], 16) for i in (0, 2, 4)] + [alpha]))


def factor(ev: np.ndarray,
           regparam: float,
           train_steps: int):
    """PC suppression factor."""
    cutoff_ev = regparam/train_steps
    return ev/(ev + cutoff_ev)


# Create data:
sys_obj = sims.Lorenz63(dt=0.1)
# sys_obj = sims.Lorenz63(dt=0.05)
# sys_obj = sims.Halvorsen(dt=0.05)
# sys_obj = sims.Logistic()
# sys_obj = sims.Henon()
# sys_obj = sims.ComplexButterfly()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 2000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 2000,
                    "n_train_sects": 1,
                    "n_validate_sects": 1, # keep it at 1 here.
                    "normalize_and_center": False,
                    }

n_train = ts_creation_args["n_train_sects"]
train_sync_steps = ts_creation_args["t_train_sync"]
pred_sync_steps = ts_creation_args["t_validate_sync"]
train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                         **ts_creation_args)

x_dim = sys_obj.sys_dim

# Build RC args:
build_args = {
    "x_dim": x_dim,
    "r_dim": 500,
    "n_rad": 0.4,
    "n_avg_deg": 5,
    "n_type_opt": "erdos_renyi",
    "r_to_rgen_opt": "linear_r",
    "act_fct_opt": "tanh",
    "node_bias_opt": "random_bias",
    "node_bias_scale": 0.4,
    "w_in_opt": "random_sparse",
    # "w_in_opt": "random_dense_uniform",
    "w_in_scale": 1.0,
    "x_train_noise_scale": 0.0,
    # "reg_param": 1e-7,
    "reg_param": 1e-4,
    "ridge_regression_opt": "bias",
    "scale_input_bool": True,
    "perform_pca_bool": False
    # "perform_pca_bool": True
}


# Ensemble size:
n_ens = 1  # 10

# seeds:
seed = 300
rng = np.random.default_rng(seed)
seeds = rng.integers(0, 10000000, size=n_ens)

# Check if pc-transform part of RC or not:
pca_rc_bool = False
if "perform_pca_bool" in build_args:
    if build_args["perform_pca_bool"]:
        pca_rc_bool = True

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
        _, _, more_out = esn_obj.train(train_data,
                                       sync_steps=train_sync_steps,
                                       more_out_bool=True)

        # PCA train:
        if pca_rc_bool:
            res_pca_states = more_out["rfit"]

        else:
            res_states = more_out["rfit"]  # get rfit states
            pca = PCA()
            res_pca_states = pca.fit_transform(res_states)  # pc transform rfit states.

        var_train = np.var(res_pca_states, axis=0)

        var_nopca_train = np.var(more_out["r"], axis=0)

        w_out = esn_obj.get_w_out()

        if build_args["ridge_regression_opt"] == "bias":
            w_out = w_out[:, :-1]

        if pca_rc_bool:
            w_out_pca = w_out
        else:
            w_out_pca = w_out @ pca.components_.T

        # Prediction:
        validate_data = validate_data_list_of_lists[i_train][0]
        pred, true_pred, more_out_pred = esn_obj.predict(validate_data,
                                                         sync_steps=pred_sync_steps,
                                                         more_out_bool=True)

        # pred pca states:
        if pca_rc_bool:
            res_pca_states_pred = more_out_pred["rfit"]
        else:

            res_states_pred = more_out_pred["rfit"]
            res_pca_states_pred = pca.transform(res_states_pred)  # PCA transform the res_states_pred

        # var_nopca_pred = np.var(more_out_pred["r"], axis=0)

        var_pred = np.var(res_pca_states_pred, axis=0)

        ### VALID TIME? MAYBE?
        error_series_ts = meas.error_over_time(y_pred=pred,
                                               y_true=true_pred,
                                               normalization="root_of_avg_of_spacedist_squared")
        vt = meas.valid_time_index(error_series_ts, error_threshold=0.4)


        if i_train == 0 and i_ens == 0:
            # explained variances:
            n_components = var_train.size
            # expl_var_ratio_results = np.zeros((n_ens, n_train, n_components))
            # rpca_train_std_results = np.zeros((n_ens, n_train, n_components))
            # rpca_pred_std_results = np.zeros((n_ens, n_train, n_components))
            rpca_train_var_results = np.zeros((n_ens, n_train, n_components))
            rpca_pred_var_results = np.zeros((n_ens, n_train, n_components))

            # rnopca_train_var_results = np.zeros((n_ens, n_train, n_components))
            # rnopca_pred_var_results = np.zeros((n_ens, n_train, n_components))

            # w_out_nopca_results = np.zeros((n_ens,
            #                                 n_train,
            #                                 w_out.shape[0],
            #                                 w_out.shape[1]))

            w_out_pca_results = np.zeros((n_ens,
                                          n_train,
                                          w_out.shape[0],
                                          w_out.shape[1]))

            # valid time??
            vt_results = np.zeros((n_ens, n_train))

        w_out_pca_results[i_ens, i_train, :, :] = w_out_pca

        # Rpca std for train and predict:
        rpca_train_var_results[i_ens, i_train, :] = var_train
        rpca_pred_var_results[i_ens, i_train, :] = var_pred

        # No pca:
        # w_out_nopca_results[i_ens, i_train, :, :] = w_out

        # Rpca std for train and predict:
        # rnopca_train_var_results[i_ens, i_train, :] = var_nopca_train
        # rnopca_pred_var_results[i_ens, i_train, :] = var_nopca_pred

        #valid time??
        vt_results[i_ens, i_train]  = vt


# Decide to plot pc-transformed quantities or not:

w_out_results = w_out_pca_results
pred_var_results = rpca_pred_var_results
train_var_results = rpca_train_var_results
xaxis_title = r"$\Large \text{principal component } i$"
yaxis_title1 = r"$\Large \text{Var of}\;\, \boldsymbol{r}_\text{pc}(t)$"
# yaxis_title1 = r"$\Large \lambda_i^{(t/p)}$"
# yaxis_title2 = r"$\Large |W_{\text{out, PC}, ij}|$"
# yaxis_title2 = r"$\Large |W_{\text{pc}, ij}|$"
yaxis_title2 = r"$\Large |\mathrm{W}_{\text{pc}, ji}|$"
# yaxis_title3 = r"$\Large f(\lambda_i)$"
yaxis_title3 = r"$\Large f(\phi_i)$"
f_name = "pca"

# wout
abs_w_out = np.abs(w_out_results)
mean_abs_w_out = np.median(abs_w_out, axis=(0, 1))  # median:
error_low_w_out = np.quantile(abs_w_out, q=0.25, axis=(0, 1))
error_high_w_out = np.quantile(abs_w_out, q=0.75, axis=(0, 1))

# train
mean_rpca_train_var = np.median(train_var_results, axis=(0, 1))
low_rpca_train_var = np.quantile(train_var_results, q=0.25, axis=(0, 1))
high_rpca_train_var = np.quantile(train_var_results, q=0.75, axis=(0, 1))

# predict
mean_rpca_pred_var = np.median(pred_var_results, axis=(0, 1))
low_rpca_pred_var = np.quantile(pred_var_results, q=0.25, axis=(0, 1))
high_rpca_pred_var = np.quantile(pred_var_results, q=0.75, axis=(0, 1))

# low_rpca_pred_var = np.min(pred_var_results, axis=(0, 1))
# high_rpca_pred_var = np.max(pred_var_results, axis=(0, 1))

# Plot:
x = np.arange(1, mean_rpca_train_var.size + 1)

fig = make_subplots(rows=2,
                    cols=1,
                    shared_xaxes=True,
                    x_title=xaxis_title,
                    # subplot_titles=["Variance of reservoir states training vs. prediction",
                    #                 "Wout"]
                    specs=[[{"secondary_y": False}],
                           [{"secondary_y": True}]]
                    )

# first variance plot:
# data_list = ((mean_rpca_train_var, low_rpca_train_var, high_rpca_train_var, r"$\Large \text{Train } (\lambda_i)$"),
#              (mean_rpca_pred_var, low_rpca_pred_var, high_rpca_pred_var, r"$\Large\text{Predict}$"))
data_list = ((mean_rpca_train_var, low_rpca_train_var, high_rpca_train_var, r"$\Large \text{Train } (\phi_i)$"),
             (mean_rpca_pred_var, low_rpca_pred_var, high_rpca_pred_var, r"$\Large\text{Predict } (\tilde{\phi}_i)$"))
# data_list = ((mean_rpca_pred_var, low_rpca_pred_var, high_rpca_pred_var, r"$\Large\text{Predict}$"),
#              (mean_rpca_train_var, low_rpca_train_var, high_rpca_train_var, r"$\Large \text{Train}$"))
for data in data_list:

    mean, low, high, name = data
    print("max ev: ", mean[0], "min_ev", mean[-1])
    color = next(col_pal_iterator)
    color_fade = hex_to_rgba(color, 0.6)
    fig.add_trace(
        go.Scatter(x=x, y=mean,
                   showlegend=True,
                   line=dict(color=color,
                             width=3),
                   name=name,
                   ), row=1, col=1,
        # secondary_y=False
    )

    high = high.tolist()
    low = low.tolist()
    x_list = x.tolist()
    fig.add_trace(
        go.Scatter(x=x_list + x_list[::-1], y=high + low[::-1],
                   line=dict(color='rgba(255,255,255,0)'),
                   fill="toself",
                   fillcolor=color_fade,
                   showlegend=False,
                   ), row=1, col=1,
        # secondary_y=False
    )

# secondly w_out and factor f:
# Wout:
for i in range(esn_obj.y_dim):
    color = next(col_pal_iterator)
    fig.add_trace(
        go.Scatter(x=x, y=mean_abs_w_out[i, :],
                   mode='none',
                   stackgroup='one',
                   fillcolor=color,
                   # name=fr"$j = {i+1}$",
                   showlegend=False
                   ),
        row=2, col=1, secondary_y=False
    )

# Factor:
train_steps = res_pca_states.shape[0]
y = factor(mean_rpca_train_var,
           regparam=build_args["reg_param"],
           train_steps=train_steps)
fig.add_trace(
    go.Scatter(x=x,
               y=y,
               line=dict(color="black",
                         # dash="dot",
                         width=2),
               showlegend=False,
               # yaxis="y3",
               # yaxis="y1",
               ),
    secondary_y=True,
    row=2,
    col=1
)

pc_cutoff = np.argmin(np.abs(y - 0.5)) + 1
print("pc_cutoff", pc_cutoff)
# # point:
# fig.add_trace(
#     go.Scatter(x=[pc_cutoff],
#                y=[0.5],
#                showlegend=False,
#                marker=dict(color="red")
#                ),
#     secondary_y=True,
#     row=2,
#     col=1
# )

# Vertical line:

for row in [1, 2]:

    if row == 2:
        # annotation = dict(
            # text=r"$\large i_\text{co} = " + fr"{pc_cutoff}$",
            # xshift=-43,
            # text=r"$\large i_\text{co}$",
            # align="center",
            # yshift=20)
        annotation = dict(
            text=r"$\large i_\text{co}$",
            align="center",
            xshift=10,
            yshift=-10)
    else:
        annotation = dict(text="")

    fig.add_vline(
        x=pc_cutoff,
        line_width=2,
        line_dash="dash",
        line_color="black",
        opacity=1.0,
        annotation=annotation,
        row=row,
    )

# add hline for factor plot:
fig.add_trace(
    go.Scatter(x=[1, build_args["r_dim"]],
               y=[0.5, 0.5],
               line=dict(color="black", dash="dash", width=1),
               marker=dict(size=0),
               showlegend=False,
               mode="lines",
               ),
    secondary_y=True,
    row=2, col=1
)

# add hline for ev plot:
y_hline = build_args["reg_param"]/train_steps
fig.add_trace(
    go.Scatter(x=[1, build_args["r_dim"]],
               y=[y_hline, y_hline],
               line=dict(color="black", dash="dash", width=1),
               marker=dict(size=0),
               showlegend=False,
               mode="lines",
               # mode="lines+text",
               # # text=[r"$\large \beta / N_\text{T}$", ""]
               # text=["$latextest$", ""],
               # textposition="bottom right",
               ),
    secondary_y=False,
    row=1, col=1
)
# fig.add_annotation(x=1, y=y_hline,
#                    # text=r"$\large \beta / N_\text{T}$",
#                    text=r"$\large \phi_\text{co} = \beta / N_\text{T}$",
#                    showarrow=False,
#                    yshift=-125,
#                    xshift=90,
#                    row=1, col=1)

fig.add_annotation(x=1, y=np.log10(y_hline),
                   # text=r"$\large \beta / N_\text{T}$",
                   # text=r"$\large \phi_\text{co} = \beta / N_\text{T}$",
                   text=r"$\large \phi_\text{co}$",
                   showarrow=False,
                   yshift=-15,
                   xshift=50,
                   row=1, col=1)


fig.update_layout(
    yaxis1=dict(
        title=yaxis_title1,
        type="log",
        exponentformat='power',
        # showgrid=True,
        # gridwidth=1,
        # gridcolor="gray"
    ),

    yaxis2=dict(
        title=yaxis_title2,
        range=[0, 90]
    ),

    yaxis3=dict(
        title=yaxis_title3,
        dtick=0.5,
        range=[0.0, 1.1]
    ),
    xaxis=dict(range=[0, build_args["r_dim"]])
)


fig.update_layout(
    template="simple_white",
    width=700,
    height=600,
    # legend_traceorder="reversed",
    legend_orientation="h",
    legend_yanchor="top",
    legend_y=1,
    legend_x=0.47,
    legend=dict(bordercolor="grey",
                borderwidth=2,
                # traceorder="reversed",
                font=dict(size=15)),
    font=dict(
        size=18,
        family="Times New Roman"
    ),
    margin=dict(l=10, r=15, t=10, b=85),
)
# print(fig.layout)
# fig.show()

# SAVE
# fig.write_image("intro_expl_var_w_error.pdf", scale=3)
fig.write_image(f"rpca_wout_plot_seperate_09_01_2023__{f_name}.png", scale=3)
