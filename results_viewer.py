from __future__ import annotations

from typing import Any
import streamlit as st

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np

import src.streamlit_src.app_fragments.streamlit_utilities as utils

with st.sidebar:

    # LOAD FILE:
    file = st.file_uploader("Choose File", type="pkl", accept_multiple_files=False)
    df: None | pd.DataFrame = None
    if file is not None:
        df = pd.read_pickle(file)

        n_ens = df["i ens"].max() + 1
        st.markdown(f"Ensemble size: {n_ens}")
        n_train = df["i train sect"].max() + 1
        st.markdown(f"Train sections: {n_train}")
        n_validate = df["i val sect"].max() + 1
        st.markdown(f"Validation sections: {n_validate}")
        if "i test sect" in df.columns:
            n_test = df["i test sect"].max() + 1
            st.markdown(f"Test sections: {n_validate}")

        utils.st_line()

        # FILTER DF:
        with st.expander("Filter by parameters: "):
            parameter_cols = [x for x in df.columns if x.startswith("P ")]
            selections = {}
            for param_col in parameter_cols:
                unique_params = df[param_col].unique()
                selection = utils.st_selectbox_with_all(param_col,
                                                        unique_params,
                                                        default_select_all_bool=True,
                                                        disable_if_only_one_opt=True,
                                                        key=f"{param_col}")
                selections[param_col] = selection

            filtered_df = df.copy()
            for key, val in selections.items():
                filtered_df = filtered_df[filtered_df[key].isin(val)]

        utils.st_line()

st.header("Show data: ")
with st.expander("Show data"):
    st.write(filtered_df)

st.header("Plot quantities: ")

# Choose Metrics:
metrics_cols = [x for x in filtered_df.columns if x.startswith("M ")]
chosen_metrics = utils.st_selectbox_with_all("Select metrics",
                                             metrics_cols,
                                             disable_if_only_one_opt=True)

# Choose averaging:
avg_mode = st.selectbox("Averaging", ["mean and std", "median and quartile"])
if avg_mode == "mean and std":
    avg_str = "mean"
    error_high = "std_high"
    error_low = "std_low"
elif avg_mode == "median and quartile":
    avg_str = "median"
    error_high = "quartile_high"
    error_low = "quartile_low"


# Sweep param:
parameter_cols = [x for x in df.columns if x.startswith("P ")]
sweep_params = [x for x in parameter_cols if len(df[x].unique()) > 1]
sweep_param = st.selectbox("Sweep parameter", sweep_params)
non_sweep_params = [x for x in parameter_cols if x != sweep_param]

### statistical functions:
def mean(x):
    return np.mean(x)

def std_low(x):
    return np.std(x) * 0.5

def std_high(x):
    return np.std(x) * 0.5

def median(x):
    return np.median(x)

def quartile_low(x):
    return np.median(x) - np.quantile(x, q=0.25)

def quartile_high(x):
    return np.quantile(x, q=0.75) - np.median(x)

stat_funcs = [mean, std_low, std_high, median, quartile_low, quartile_high]

# Do statistics:
group_obj = df.groupby(parameter_cols, as_index=False)
res = group_obj[chosen_metrics].agg(stat_funcs).reset_index(inplace=False)
res


# non_sweep_params
out = np.unique(res[non_sweep_params].values)
# out = res[non_sweep_params].unique_values()

# out

# px.line(res, x=sweep_param, y="")

out = res.value_counts(non_sweep_params).index
unique_non_sweep_param_values = out.values
st.write(unique_non_sweep_param_values.shape)

for metric in chosen_metrics:
    fig = go.Figure()
    for value_combination in unique_non_sweep_param_values:

        # Group by other sweep variables -> maybe do with GroupBy?.
        condition_df: None | pd.DataFrame = None
        for non_sweep_param, value in zip(non_sweep_params, value_combination):
            condition_series = res[non_sweep_param] == value
            if condition_df is None:
                condition_df = condition_series
            else:
                condition_df = condition_df & condition_series
        sub_df = res[condition_df]
        name = str(list(zip(non_sweep_params, value_combination)))
        fig.add_trace(
            go.Scatter(x = sub_df[sweep_param],
                       y=sub_df[(metric, avg_str)],
                       error_y={"array": sub_df[(metric, error_high)],
                                "arrayminus": sub_df[(metric, error_low)]},
                       name=name
                       )
        )
    fig.update_yaxes(title=metric)
    fig.update_xaxes(title=sweep_param)
    fig.update_layout(title=avg_mode)
    st.plotly_chart(fig)

# unique_non_sweep_param_values
#
# for value_combination in unique_non_sweep_param_values:
#     st.write("val comb: ", value_combination)
#     condition_df: None | pd.DataFrame = None
#     for non_sweep_param, value in zip(non_sweep_params, value_combination):
#         st.write(non_sweep_param, value)
#         st.write(res[non_sweep_param] == value)
#         condition_series = res[non_sweep_param] == value
#         if condition_df is None:
#             condition_df = condition_series
#         else:
#             condition_df = condition_df & condition_series
#     condition_df
#     res[condition_df]
# condition_df
# res[condition_df]



#
# utils.st_line()
# # 1. Get all parameter columns
# parameter_cols
#
# # 2. Get all Metric columns:
# metric_cols = [x for x in df.columns if x.startswith("M ")]
#
# # 3. Average over all "i .." columns.
# group_obj = df.groupby(parameter_cols, as_index=False)
# # res = group_obj[metric_cols].mean()
#
# # res = group_obj[metric_cols].transform("mean")
# # res
#
#
# res = group_obj[metric_cols].agg(["mean", "median", "std", "quantile"]).reset_index(inplace=False)
#
# res.columns = [" ".join(a) if a[0].startswith("M ") else a[0] for a in res.columns.to_flat_index()]
# res
# #
# # # 4. Plot:
# # fig = px.line(res, x="P r_dim")
# # st.plotly_chart(fig)
# #
# # res.columns
