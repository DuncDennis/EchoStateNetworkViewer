from __future__ import annotations

import streamlit as st

import plotly.graph_objects as go
import pandas as pd
import numpy as np

import src.streamlit_src.app_fragments.streamlit_utilities as utils

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

# SHOW TOTAL FILTERED DF:
st.header("Show data: ")
with st.expander("Show data"):
    st.write(filtered_df)
    st.write(filtered_df.shape)

# AGGREGATE RESULTS:
st.header("Aggregate data: ")

# All paramter columns
parameter_cols = [x for x in filtered_df.columns if x.startswith("P ")]

# Choose Metrics:
metrics_cols = [x for x in filtered_df.columns if x.startswith("M ")]
chosen_metrics = utils.st_selectbox_with_all("Select metrics",
                                             metrics_cols,
                                             disable_if_only_one_opt=True)

# Do statistics:
group_obj = filtered_df.groupby(parameter_cols, as_index=False)
df_agg = group_obj[chosen_metrics].agg(stat_funcs).reset_index(inplace=False)
st.write(df_agg)

# Plotting:
st.header("Plotting: ")

sweep_params = [x for x in parameter_cols if len(filtered_df[x].unique()) > 1]
sweep_param = st.selectbox("Sweep parameter", sweep_params)
non_sweep_params = [x for x in parameter_cols if x != sweep_param]

other_params = df_agg.value_counts(non_sweep_params).index
unique_non_sweep_param_values = other_params.values
# st.write(unique_non_sweep_param_values.shape)

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

for metric in chosen_metrics:
    fig = go.Figure()
    for value_combination in unique_non_sweep_param_values:

        # Group by other sweep variables -> maybe do with GroupBy?.
        condition_df: None | pd.DataFrame = None
        for non_sweep_param, value in zip(non_sweep_params, value_combination):
            condition_series = df_agg[non_sweep_param] == value
            if condition_df is None:
                condition_df = condition_series
            else:
                condition_df = condition_df & condition_series
        sub_df = df_agg[condition_df]
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


