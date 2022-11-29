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
n = 100  # nr samples

# get R:
data_mean = np.array([5, 5])
# data_mean = np.array([0, 0])
# data_cov = np.array([[2, 1], [1, 1]])
data_cov = np.array([[2, 1], [1.9, 1]])
rng = np.random.default_rng(seed=seed)
R = rng.multivariate_normal(mean=data_mean, cov=data_cov, size=n)

if st.checkbox("Center R", value=False):
    R = R - np.mean(R, axis=0)

# mean of R:
mean_r = np.mean(R, axis=0)

# get Y:
W_real = np.ones((2, 1)) # r_dim x y_dim
b_real = 5
Y = np.zeros((n, 1))
for i in range(n):
    Y[i, 0] = R[i, :] @ W_real + b_real
error = rng.normal(size=(n, 1))
Y += error
mean_y = np.mean(Y)

st.write("Mean of r: ", mean_r)
st.write("Mean of y: ", mean_y)

############### FITS ###################

#### Fit with "traditional" formula: ###

# Perform normal Linear fit (i.e. reg_param = 0).
rfit_array = utilities.vectorize(utilities.add_one, (R,))
W_lr = np.linalg.inv(rfit_array.T @ rfit_array) @ rfit_array.T @ Y
b_lr = W_lr[-1, 0]
W_lr_out = W_lr[:2, :]

# Perform the RR fit:
rfit_array = utilities.vectorize(utilities.add_one, (R,))
eye = np.eye(rfit_array.shape[1])
eye[-1, -1] = 0  # No regularization for bias term.
W_rr = np.linalg.inv(rfit_array.T @ rfit_array + reg_param * eye) @ rfit_array.T @ Y
b_rr = W_rr[-1, 0]
W_rr_out = W_rr[:2, :]

#### Fit with equivalent centered data and no bias:
R_centered = R - np.mean(R, axis=0)
Y_centered = Y - np.mean(Y, axis=0)

# Perform the LR fit:
W_lr_out_cent = np.linalg.inv(R_centered.T @ R_centered) @ R_centered.T @ Y_centered
b_lr_cent = mean_y - mean_r @ W_lr_out_cent

# Perform the RR fit:
W_rr_out_cent = np.linalg.inv(R_centered.T @ R_centered +
                              reg_param * np.eye(2)) @ R_centered.T @ Y_centered
b_rr_cent = mean_y - mean_r @ W_rr_out_cent

######### PREDICTIONS: ############

# Predictor of lr fit:
Y_lr_hat = R @ W_lr_out + b_lr

# Predictor of rr fit:
Y_rr_hat = R @ W_rr_out + b_rr

# Predictor of lr fit centered:
Y_lr_cent_hat = R @ W_lr_out_cent + b_lr_cent

# Predictor of rr fit centered:
Y_rr_cent_hat = R @ W_rr_out_cent + b_rr_cent

#############  SINGULAR VALUE DECOMPOSITION: ############
# Singular value decomposition values:
# Shapes: u (n x p), d (p x p), v (p x p).
u, s, v_t = np.linalg.svd(R_centered, full_matrices=False)
d = np.diag(s)
v = v_t.T
# -> Now: R_centered = u @ d @ v.T

### Predictors with SVD:

# LR with SVD:
y_lr_cent_svd_hat = u @ u.T @ Y_centered + mean_y

# RR with SVD:
diag_rr = np.diag(s**2 / (s**2 + reg_param))
y_rr_cent_svd_hat = u @ diag_rr @ u.T @ Y_centered + mean_y

### Further investigations with SVD:
Y_in_u = u.T @ Y_centered
st.write("Y_in_u: ", Y_in_u)

Y_in_u_rr = diag_rr @ u.T @ Y_centered
st.write("Y_in_u rr: ", Y_in_u)

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

# add 3d points (real)
fig.add_trace(
    go.Scatter3d(x=R[:, 0], y=R[:, 1], z=Y[:, 0],
                 mode='markers',
                 marker=dict(
                     size=3,
                     color="red"
                     # opacity=0.8
                 ),
                 name="Real data"
                 )
)

# # add 3d points (rr traditional)
# fig.add_trace(
#     go.Scatter3d(x=R[:, 0], y=R[:, 1], z=Y_rr_hat[:, 0],
#                  mode='markers',
#                  marker=dict(
#                      size=2,
#                      color="blue"
#                  ),
#                  name="RR traditional"
#                  )
# )
#
# # add 3d points (lr traditional)
# fig.add_trace(
#     go.Scatter3d(x=R[:, 0], y=R[:, 1], z=Y_lr_hat[:, 0],
#                  mode='markers',
#                  marker=dict(
#                      size=2,
#                      color="green"
#                  ),
#                  name="LR traditional"
#                  )
# )
#
# # add 3d points (rr centered)
# fig.add_trace(
#     go.Scatter3d(x=R[:, 0], y=R[:, 1], z=Y_rr_cent_hat[:, 0],
#                  mode='markers',
#                  marker=dict(
#                      size=2,
#                      # color="blue"
#                  ),
#                  name="RR centered"
#                  )
# )
#
# # add 3d points (lr centered)
# fig.add_trace(
#     go.Scatter3d(x=R[:, 0], y=R[:, 1], z=Y_lr_cent_hat[:, 0],
#                  mode='markers',
#                  marker=dict(
#                      size=2,
#                      color="green"
#                  ),
#                  name="LR centered"
#                  )
# )

# add 3d points (rr centered and svd)
fig.add_trace(
    go.Scatter3d(x=R[:, 0], y=R[:, 1], z=y_rr_cent_svd_hat[:, 0],
                 mode='markers',
                 marker=dict(
                     size=2,
                     # color="blue"
                 ),
                 name="RR svd"
                 )
)

# add 3d points (lr centered and svd)
fig.add_trace(
    go.Scatter3d(x=R[:, 0], y=R[:, 1], z=y_lr_cent_svd_hat[:, 0],
                 mode='markers',
                 marker=dict(
                     size=2,
                     # color="green"
                 ),
                 name="LR svd"
                 )
)

# add 2d points
z_display = np.min(Y) - 2
fig.add_trace(
    go.Scatter3d(x=R[:, 0], y=R[:, 1], z=np.ones(n) * z_display,
                 mode='markers',
                 marker=dict(
                     size=1,
                     color="black"
                     # opacity=0.8
                 ),
                 name="R"
                 )
)

# Add principle components:
start_x = mean_r[0]
start_y = mean_r[1]

# PC1
end_x = v[0, 0] + start_x
end_y = v[1, 0] + start_y
z_display = np.min(Y) - 2
fig.add_trace(
    go.Scatter3d(x=[start_x, end_x], y=[start_y, end_y], z=np.ones(2) * z_display,
                 mode='lines',
                 name="PC1",
                 line=dict(width=5)
                 )
)

# PC2
end_x = v[0, 1] + start_x
end_y = v[1, 1] + start_y
z_display = np.min(Y) - 2
fig.add_trace(
    go.Scatter3d(x=[start_x, end_x], y=[start_y, end_y], z=np.ones(2) * z_display,
                 mode='lines',
                 name="PC2",
                 line=dict(width=5)
                 )
)

# Add y in U basis:
# Corresponding to first component:
end = v[:, 0] * Y_in_u[0] + mean_r
fig.add_trace(
    go.Scatter3d(x=[end[0]], y=[end[1]], z=np.ones(1) * z_display,
                 mode='markers',
                 marker=dict(
                     size=5,
                 ),
                 name="y in u pc1"
                 )
)

# Corresponding to second component:
end = v[:, 1] * Y_in_u[1] + mean_r
fig.add_trace(
    go.Scatter3d(x=[end[0]], y=[end[1]], z=np.ones(1) * z_display,
                 mode='markers',
                 marker=dict(
                     size=5,
                 ),
                 name="y in u pc2"
                 )
)

# Add y in U basis with RR:
# Corresponding to first component:
end = v[:, 0] * Y_in_u_rr[0] + mean_r
fig.add_trace(
    go.Scatter3d(x=[end[0]], y=[end[1]], z=np.ones(1) * z_display,
                 mode='markers',
                 marker=dict(
                     size=5,
                 ),
                 name="y in u pc1 RR"
                 )
)

# Corresponding to second component:
end = v[:, 1] * Y_in_u_rr[1] + mean_r
fig.add_trace(
    go.Scatter3d(x=[end[0]], y=[end[1]], z=np.ones(1) * z_display,
                 mode='markers',
                 marker=dict(
                     size=5,
                 ),
                 name="y in u pc2 RR"
                 )
)

fig.update_layout(scene=dict(aspectmode="data"))

st.plotly_chart(fig)


# more write:

# st.write(b_rr_th)
# st.write(b_rr)
#
# st.write(b_lr_th)
# st.write(b_lr)

st.write("UT @ U")
st.write(u.T @ u)

st.write("U @ UT")
st.write(u @ u.T)


# # TEST:
# R_new = R.T
# Y_new = Y.T
#
# a = Y_new  @ R_new.T @ np.linalg.inv(R_new @ R_new.T + reg_param * np.eye(2))
# b = np.linalg.inv(R.T @ R + reg_param * np.eye(2)) @ R.T @ Y
#
# st.write(a, b)
