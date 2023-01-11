import numpy as np
import plotly.graph_objects as go

# import streamlit as st

import src.esn_src.utilities as utilities
from sklearn.decomposition import PCA


# log_reg_param = st.number_input('log_reg_param',
#                                     value=1.,
#                                     step=1.,
#                                     format="%f",)
# reg_param = 10 ** (log_reg_param)

# Set seed
seed = 1

# Parameters
n = 600  # nr samples

# get 2D R:
data_mean_2d = np.array([5, 5])
# data_mean = np.array([0, 0])
data_cov = np.array([[2, 1], [1, 1]])
data_cov = np.array([[2, 1], [1, 1]])
# data_cov = np.array([[2, 1], [1.9, 1]])
rng = np.random.default_rng(seed=seed)
R_2d = rng.multivariate_normal(mean=data_mean_2d, cov=data_cov, size=n)

# Get 3d R:
R = np.zeros((n, 3))
R[:, :2] = R_2d
W_real = 0.5 *np.ones((2, 1)) # r_dim x y_dim
b_real = 5
for i in range(n):
    R[i, 2] = R_2d[i, :] @ W_real + b_real # + 0.1

#################### PLOTS: ###############

# Plot the points
fig = go.Figure()



# Plot surface:
R_mean = np.mean(R, axis=0)
# x range:
dx = 2
x_start = np.min(R, axis=0)[0] - dx
x_end = np.max(R, axis=0)[0] + dx
x_data = [x_start, x_end]

# y range:
dy = 2
y_start = np.min(R, axis=0)[1] - dy
y_end = np.max(R, axis=0)[1] + dy
y_data = [y_start, y_end]

# z_data:
z_data = np.ones((2, 2))
for i_x, x in enumerate(x_data):
    for i_y, y in enumerate(y_data):
        # z_data[i_x, i_y] = np.array([x, y]) @ W_real + b_real
        z_data[i_y, i_x] = np.array([x, y]) @ W_real + b_real

colorscale=[[0, 'black'], [1, 'black']]  #black

fig.add_trace(
    go.Surface(x=x_data, y=y_data, z = z_data,
               opacity=0.2,
               showlegend=False,
               showscale=False,
               colorscale=colorscale
               # surfacecolor="b"
               )
)

## PCA:
pca = PCA()
pca.fit(R)
p1 = pca.components_[0, :]
p2 = pca.components_[1, :]
p3 = pca.components_[2, :]

d = 5
start = np.mean(R, axis=0)
end1 = start + p1*d
end2 = start + p2*d
end3 = start + p3*d

ends = [end1, end2, end3]
names = ["PC1", "PC2", "PC3"]
names = [r"$\boldsymbol{p}_1$", r"$\boldsymbol{p}_2$", r"$\boldsymbol{p}_3$"]
text_ds = [-0.4, +0.5, 0.5]
annotations = []
for i in range(3):
    end = ends[i]
    fig.add_trace(
        go.Scatter3d(x=[start[0], end[0]],
                     y=[start[1], end[1]],
                     z=[start[2], end[2]],
                     mode="lines",
                     line=dict(width=3,
                               color="black"),
                     showlegend=False)
    )



    text_d = text_ds[i]
    name = names[i]
    annot_dict = dict(
                x=end[0] + text_d, y=end[1] + text_d, z=end[2] + text_d,
                text=name,
                showarrow=False,
                font=dict(
                    color="black",
                    size=12
                )
            )
    annotations.append(annot_dict)

fig.update_layout(
    scene=dict(
        annotations=annotations
    )
)


fig.add_trace(
    go.Scatter3d(x=R[:, 0], y=R[:, 1], z=R[:, 2],
                 mode='markers',
                 marker=dict(
                     color="blue",
                     # size=0.9,
                     # opacity=0.7,
                     size=1.5,
                     opacity=0.3,
                     line=dict(
                         color="black",
                         width=0.5,
                     )
                 ),
                 name="Real data"
                 )
)
# add 3d points (real)

height = 300
# width = int(1.4 * height)
width = 300

fig.update_layout(template="plotly_white",
                  showlegend=False,
                  )
a = 0.68
camera = dict(eye=dict(x=1.25, y=-1.25, z=1.25))
camera = dict(eye=dict(x=0.4, y=-2.3, z=0.3))
camera = dict(eye=dict(x=-3, y=-0.5, z=0.2))
camera = dict(eye=dict(x=-2.0, y=-0.5, z=0.2))
camera = dict(eye=dict(x=-2.0*a, y=-3*a, z=0.2*a))

fig.update_scenes(
    xaxis_showticklabels=False,
    yaxis_showticklabels=False,
    zaxis_showticklabels=False
)

fig.update_layout(scene_camera=camera,
                  )
# fig.update_layout(scene=dict(aspectmode="data"))
fig.update_layout(scene=dict(aspectmode="data"))

fig.update_layout(
    # autosize=False,
    # margin=dict(l=20, r=20, t=20, b=20),
    margin=dict(l=0, r=0, t=0, b=0, pad=0),
    width=width,
    height=height,
)

# st.plotly_chart(fig)


# SAVE
# fig.write_image("intro_expl_var_w_error.pdf", scale=3)
file_name = f"pca_theory_3d.png"
fig.write_image(file_name, scale=7)

# EXPLAINED VARIANCE PLOT:
R_pca = pca.transform(R)
var = np.var(R_pca, axis=0)

# x = [r"$\lambda_1$", r"$\lambda_2$", r"$\lambda_3$"]
x = [r"$\boldsymbol{p}_1$", r"$\boldsymbol{p}_2$", r"$\boldsymbol{p}_3$"]
y = var
fig = go.Figure()
fig.add_trace(
    go.Bar(x=x, y=y)
)

# yaxis_title = r"$\text{Explained variance  } \lambda_i$"
# yaxis_title = r"$\text{Explained variance  } \phi_i$"
# yaxis_title = r"$\text{explained variance  } \phi_i$"
yaxis_title = r"$\text{expl. variance  } \phi_i$"
# xaxis_title =  r'$\text{Principal component}$'
xaxis_title =  r'$\text{principal component } i$'
height = 300
width = int(1.4 * height)
font_size = 15
legend_font_size = 15
font_family = "Times New Roman"

fig.update_layout(
    width=width,
    height=height,
    yaxis_title=yaxis_title,
    xaxis_title=xaxis_title,

    font=dict(
        size=font_size,
        family=font_family
    ),
    )

fig.update_layout(template="plotly_white",
                  )

fig.update_layout(
    margin=dict(l=20, r=20, t=20, b=20),
)

# st.plotly_chart(fig)
file_name = f"pca_theory_barplot.png"
fig.write_image(file_name, scale=5)
