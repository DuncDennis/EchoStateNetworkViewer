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
data_cov = np.array([[2, 1], [1, 1]])
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



# Predictor of lr fit:
Y_lr_hat = R @ W_lr_out + b_lr

# Predictor of rr fit:
Y_rr_hat = R @ W_rr_out + b_rr

# Theoretical biases:
# theoretical bias term:
b_rr_th = mean_y - mean_r @ W_rr_out
b_lr_th = mean_y - mean_r @ W_lr_out

# Alternative fit: Get W by performing a fit without bias on centered R and Y, then calculate the
R_centered = R - np.mean(R, axis=0)
# Y_centered = Y - np.mean(Y, axis=0)
Y_centered = Y

# LR centered:
W_lr_out_cent = np.linalg.inv(R_centered.T @ R_centered) @ R_centered.T @ Y_centered

# LR centered:
W_rr_out_cent = np.linalg.inv(R_centered.T @ R_centered + reg_param * np.eye(2)) @ R_centered.T @ Y_centered

st.write("Difference between W_lrs: ")
st.write(np.max(np.abs(W_lr_out_cent - W_lr_out)))
st.write("Difference between W_rrs: ")
st.write(np.max(np.abs(W_rr_out_cent - W_rr_out)))

# Singular value decomposition values:
# Shapes: u (n x p), d (p x p), v (p x p).
u, s, v_t = np.linalg.svd(R, full_matrices=False)
d = np.diag(s)
v = v_t.T
st.write("svd: ", u.shape, d.shape, v.shape)
st.write("Max deviation between svd and R", np.max(np.abs(u @ d @ v.T - R)))

st.write("Test weather the theoretical bias is the same as the real bias: ",
         np.max(np.abs(b_rr - b_rr_th)),
         np.max(np.abs(b_lr - b_lr_th)),
         )



# Linear fit with svd:
Y_lr_svd_hat = u @ (u.T @ Y) + b_lr_th

# RR fit with svd:
diag_rr = np.diag(s**2 / (s**2 + reg_param))
Y_rr_svd_hat = u @ diag_rr @ (u.T @ Y) + b_rr_th

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

# add 3d points (rr fitted)
fig.add_trace(
    go.Scatter3d(x=R[:, 0], y=R[:, 1], z=Y_rr_hat[:, 0],
                 mode='markers',
                 marker=dict(
                     size=2,
                     color="blue"
                 ),
                 name="RR"
                 )
)

# add 3d points (lr fitted)
fig.add_trace(
    go.Scatter3d(x=R[:, 0], y=R[:, 1], z=Y_lr_hat[:, 0],
                 mode='markers',
                 marker=dict(
                     size=2,
                     color="green"
                 ),
                 name="LR"
                 )
)

# add 3d points (lr svd fitted)
fig.add_trace(
    go.Scatter3d(x=R[:, 0], y=R[:, 1], z=Y_lr_svd_hat[:, 0],
                 mode='markers',
                 marker=dict(
                     size=2,
                 ),
                 name="LR svd"
                 )
)

# add 3d points (rr svd fitted)
fig.add_trace(
    go.Scatter3d(x=R[:, 0], y=R[:, 1], z=Y_rr_svd_hat[:, 0],
                 mode='markers',
                 marker=dict(
                     size=2,
                 ),
                 name="RR svd"
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

# Plot u.T @ Y:
y_projected = u.T @ Y
# st.write("here", y_projected.shape)
# z_display = np.min(Y) - 2
# fig.add_trace(
#     go.Scatter3d(x=y_projected[:, 0], y=y_projected[:, 1], z=np.ones(n) * z_display,
#                  mode='markers',
#                  marker=dict(
#                      size=1,
#                      # opacity=0.8
#                  ),
#                  name="u.T @ Y"
#                  )
# )


st.plotly_chart(fig)


# more write:

st.write(b_rr_th)
st.write(b_rr)

st.write(b_lr_th)
st.write(b_lr)
