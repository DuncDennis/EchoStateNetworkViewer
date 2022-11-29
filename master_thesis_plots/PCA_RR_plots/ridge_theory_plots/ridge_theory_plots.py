import numpy as np
import plotly.graph_objects as go

import streamlit as st

import src.esn_src.utilities as utilities


log_reg_param = st.number_input('log_reg_param',
                                    value=1.,
                                    step=1.,
                                    format="%f",)
reg_param = 10 ** (log_reg_param)

# Set seed
seed = 1

# Parameters
n = int(st.number_input("nsample", value=300))

# get R:
data_mean = 10
rng = np.random.default_rng(seed=seed)
R = rng.normal(loc=data_mean, scale=5, size=n)[np.newaxis, :]

center_r = st.checkbox("Center R", value=False)
if center_r:
    R = R - np.mean(R, axis=1)

# mean of R:
mean_r = np.mean(R, axis=1)

# get Y:
true_intersect = 10
true_sloap = 0.5
Y = true_sloap * R + true_intersect
error = rng.normal(size=(1, n))*1.5
Y += error
Y = Y

if st.checkbox("Center Y", value=False):
    Y = Y - np.mean(Y, axis=1)

mean_y = np.mean(Y, axis=1)
st.write("Mean of r: ", mean_r)
st.write("Mean of y: ", mean_y)

############### FITS ###################

#### Fit with "traditional" formula: ###

# # Perform normal Linear fit (i.e. reg_param = 0).
# rfit_array = utilities.vectorize(utilities.add_one, (R.T,)).T
# # W_lr = np.linalg.inv(rfit_array.T @ rfit_array) @ rfit_array.T @ Y
# W_lr = Y @ rfit_array.T @ np.linalg.inv(rfit_array @ rfit_array.T)
# b_lr = W_lr[0, -1]
# W_lr_out = W_lr[:, :1]

# Perform the RR fit no intercept:
rfit_array = R
eye = np.eye(rfit_array.shape[0])
W_rr = Y @ rfit_array.T @ np.linalg.inv(rfit_array @ rfit_array.T + reg_param * eye)
W_rr_noint_out = W_rr[:, :1]

# Perform the RR fit:
rfit_array = utilities.vectorize(utilities.add_one, (R.T,)).T
eye = np.eye(rfit_array.shape[0])
eye[-1, -1] = 0  # No regularization for bias term.
W_rr = Y @ rfit_array.T @ np.linalg.inv(rfit_array @ rfit_array.T + reg_param * eye)
b_rr = W_rr[0, -1]
W_rr_out = W_rr[:, :1]

# Perform the RR fit alt:
rfit_array = utilities.vectorize(utilities.add_one, (R.T,)).T
eye = np.eye(rfit_array.shape[0])
W_rr_alt = Y @ rfit_array.T @ np.linalg.inv(rfit_array @ rfit_array.T + reg_param * eye)
b_rr_alt = W_rr_alt[0, -1]
W_rr_alt_out = W_rr_alt[:, :1]

# #### Fit with equivalent centered data and no bias:
# R_centered = R - np.mean(R, axis=0)
# Y_centered = Y - np.mean(Y, axis=0)
#
# # Perform the LR fit:
# W_lr_out_cent = np.linalg.inv(R_centered.T @ R_centered) @ R_centered.T @ Y_centered
# b_lr_cent = mean_y - mean_r @ W_lr_out_cent
#
# # Perform the RR fit:
# W_rr_out_cent = np.linalg.inv(R_centered.T @ R_centered +
#                               reg_param * np.eye(2)) @ R_centered.T @ Y_centered
# b_rr_cent = mean_y - mean_r @ W_rr_out_cent

######### PREDICTIONS: ############

# # Predictor of lr fit:
# Y_lr_hat = W_lr_out @ R + b_lr
#
# # Predictor of rr fit:
# Y_rr_hat = W_rr_out @ R + b_rr

# # Predictor of lr fit centered:
# Y_lr_cent_hat = R @ W_lr_out_cent + b_lr_cent
#
# # Predictor of rr fit centered:
# Y_rr_cent_hat = R @ W_rr_out_cent + b_rr_cent

#############  SINGULAR VALUE DECOMPOSITION: ############
# Singular value decomposition values:
# Shapes: u (n x p), d (p x p), v (p x p).
# u, s, v_t = np.linalg.svd(R_centered, full_matrices=False)
# d = np.diag(s)
# v = v_t.T
# -> Now: R_centered = u @ d @ v.T

### Predictors with SVD:

# LR with SVD:
# y_lr_cent_svd_hat = u @ u.T @ Y_centered + mean_y
#
# # RR with SVD:
# diag_rr = np.diag(s**2 / (s**2 + reg_param))
# y_rr_cent_svd_hat = u @ diag_rr @ u.T @ Y_centered + mean_y

### Further investigations with SVD:
# Y_in_u = u.T @ Y_centered
# st.write("Y_in_u: ", Y_in_u)
#
# Y_in_u_rr = diag_rr @ u.T @ Y_centered
# st.write("Y_in_u rr: ", Y_in_u)

# # Theoretical biases:
# # theoretical bias term:
# b_rr_th = mean_y - mean_r @ W_rr_out
# b_lr_th = mean_y - mean_r @ W_lr_out
#
#
# # Singular value decomposition values:
# # Shapes: u (n x p), d (p x p), v (p x p).
# u, s, v_t = np.linalg.svd(R, full_matrices=False)
# d = np.diag(s)
# v = v_t.T
# st.write("svd: ", u.shape, d.shape, v.shape)
# st.write("Max deviation between svd and R", np.max(np.abs(u @ d @ v.T - R)))
#
# st.write("Test weather the theoretical bias is the same as the real bias: ",
#          np.max(np.abs(b_rr - b_rr_th)),
#          np.max(np.abs(b_lr - b_lr_th)),
#          )
#
#
#
# # Linear fit with svd:
# Y_lr_svd_hat = u @ (u.T @ Y) + b_lr_th
#
# # RR fit with svd:
# diag_rr = np.diag(s**2 / (s**2 + reg_param))
# Y_rr_svd_hat = u @ diag_rr @ (u.T @ Y) + b_rr_th

#################### PLOTS: ###############

# Plot the points
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=R[0, :], y=Y[0, :],
               mode="markers",
               marker=dict(
                   size=5,
                   # color="red",
                   opacity=0.5,
                   line=dict(
                       width=1
                   )
                   # opacity=0.8
               ),
               # name=r"$\text{Trainings data: }(\boldsymbol{k}[i], \boldsymbol{y}[i])$"
               name=r"$\text{Train data: }(\boldsymbol{k}[i], \boldsymbol{y}[i])$"
               )
)

x_min = np.min(R)
x_max = np.max(R)

# # LR with bias:
# y_min = b_lr + W_lr_out[0, 0] * x_min
# y_max = b_lr + W_lr_out[0, 0] * x_max
# fig.add_trace(
#     go.Scatter(x=[x_min, x_max], y=[y_min, y_max],
#                mode="lines",
#                marker=dict(
#                    size=3,
#                    # color="red"
#                    # opacity=0.8
#                ),
#                name="LR with bias"
#                )
# )

# RR with bias:
y_min = b_rr + W_rr_out[0, 0] * x_min
y_max = b_rr + W_rr_out[0, 0] * x_max
fig.add_trace(
    go.Scatter(x=[x_min, x_max], y=[y_min, y_max],
               mode="lines",
               marker=dict(
                   size=3,
                   # color="red"
                   # opacity=0.8
               ),
               # name="intercept"
line=dict(dash="dashdot", width=4),
               name=r"$\hat{\boldsymbol{y}}\text{ unpenalized intercept}$"
               )
)

# RR with bias penalized:
y_min = b_rr_alt + W_rr_alt_out[0, 0] * x_min
y_max = b_rr_alt + W_rr_alt_out[0, 0] * x_max
fig.add_trace(
    go.Scatter(x=[x_min, x_max], y=[y_min, y_max],
               mode="lines",
               marker=dict(
                   size=3,
                   # color="red"
                   # opacity=0.8
               ),
               line=dict(dash="dot", width=4),
               name=r"$\hat{\boldsymbol{y}}\text{ penalized intercept}$"
               # name=r"$\hat{\boldsymbol{y}} = \mathbf{W}_\text{c} \boldsymbol{k} + \boldsymbol{w}_\text{c}$"
               )
)

# RR with bias penalized:
y_min = W_rr_noint_out[0, 0] * x_min
y_max = W_rr_noint_out[0, 0] * x_max
fig.add_trace(
    go.Scatter(x=[x_min, x_max], y=[y_min, y_max],
               mode="lines",
               marker=dict(
                   size=3,
                   # color="red"
                   # opacity=0.8
               ),
# line=dict(dash="dash"),
               # name="No intercept",
line=dict(width=4),
               name=r"$\hat{\boldsymbol{y}}\text{ no intercept}$",
               )
)

fig.add_vline(x=0, line_dash="dash")
fig.add_hline(y=0, line_dash="dash")

fig.update_layout(scene=dict(aspectmode="data"))
fig.update_layout(template="plotly_white",
                  showlegend=True,
                  )

fig.update_xaxes(title=r"$\text{Input } \boldsymbol{k}$")
fig.update_yaxes(title=r"$\text{Output } \boldsymbol{y}$")

fig.update_layout(
    legend=dict(
        # orientation="h",
        yanchor="bottom",
        y=0.85,  # 0.99
        # xanchor="left",
        x=0.04,
        # x=0,
        # font=dict(size=12)),
        font=dict(size=15)),
    width=500,
    height=500,
    font=dict(
        size=15,
    )
)

st.plotly_chart(fig)

log_reg_param_str = str(int(log_reg_param))
if "-" in log_reg_param_str:
    log_reg_param_str = log_reg_param_str.replace("-", "m")

if center_r:
    file_name = f"ridge_theory_centered_{log_reg_param_str}.png"
else:
    file_name = f"ridge_theory_uncentered_{log_reg_param_str}.png"
fig.write_image(file_name, scale=3)
