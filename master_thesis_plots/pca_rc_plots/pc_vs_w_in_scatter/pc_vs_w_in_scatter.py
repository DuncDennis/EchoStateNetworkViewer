"""Create the correlation plot between pca and input.
IDEA: The first principal components p_1, p_2, .. seem to be specified by the input, i.e. by w_in.
Here p_1 is scatter plotted vs. w_in. To simplify, a 1-dim input is used.
"""
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
# sys_obj = sims.Lorenz96(sys_dim=10)
# sys_obj = sims.LinearSystem()
ts_creation_args = {"t_train_disc": 1000,
                    "t_train_sync": 100,
                    "t_train": 1000,
                    "t_validate_disc": 1000,
                    "t_validate_sync": 100,
                    "t_validate": 1000,
                    "n_train_sects": 1,
                    "n_validate_sects": 1,
                    "normalize_and_center": False,
                    }

n_train = ts_creation_args["n_train_sects"]
train_sync_steps = ts_creation_args["t_train_sync"]
train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                         **ts_creation_args)

# Only one dimension
dim_select = 0  # Choose dimension here
train_data_list = [x[:, [dim_select]] for x in train_data_list]
validate_data_list_of_lists_new = []
for val_data_list in validate_data_list_of_lists:
    temp = [x[:, [dim_select]] for x in val_data_list]
    validate_data_list_of_lists_new.append(temp.copy())
validate_data_list_of_lists = validate_data_list_of_lists_new
x_dim = 1

# x_dim = sys_obj.sys_dim
# Build RC args:
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
    "x_train_noise_scale": 0.0,
    "reg_param": 1e-7,
    "ridge_regression_opt": "bias",
    "scale_input_bool": True,
}

x_dim = build_args["x_dim"]

# Ensemble size:
n_ens = 1

# seeds:
seed = 304
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
        _, _, more_out = esn_obj.train(train_data,
                                       sync_steps=train_sync_steps,
                                       more_out_bool=True)
        res_states = more_out["r"]
        pca = PCA()
        pca.fit(res_states)
        components = pca.components_ # n_components, n_features

        if i_train == 0 and i_ens == 0:
            # explained variances:
            n_components = components.shape[0]
            w_in_results = np.zeros((n_ens, n_train, n_components))
            pc_results = np.zeros((n_ens, n_train, n_components, build_args["r_dim"]))
        w_in = esn_obj.w_in
        w_in_results[i_ens, i_train, :] = w_in.flatten()
        pc_results[i_ens, i_train, :, :] = components


# PLOT:
height = 300
width = int(1 * height)
font_size = 25
legend_font_size = 25
font_family = "Times New Roman"

# xaxis_title = r"$\text{Principal component } \boldsymbol{p}_i$"
xaxis_title = r"$\boldsymbol{p}_i$"
# yaxis_title =  r"$\sum_i |\text{corr}_{ij}|$"
# yaxis_title =  r"$\text{Summed correlation between } W_\text{in} \text{ and } \boldsymbol{p}_i$"
yaxis_title =  r"$W_\text{in}$"

# plot median:
pc_to_show = 0
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=w_in_results[0, 0, :],
        y=pc_results[0, 0, pc_to_show, :],
        mode="markers",
        showlegend=False)
)


fig.update_layout(template="simple_white",
                  showlegend=False,
                  font=dict(
                      size=font_size,
                      family=font_family
                  ),
                  xaxis_title=xaxis_title,
                  yaxis_title=yaxis_title,
                  title="1st principal component"
                  )

# fig.update_layout(
#     margin=dict(l=5, r=5, t=5, b=5),
# )

# SAVE
file_name = f"pc_vs_w_in_scatter.png"
fig.write_image(file_name, scale=3)
