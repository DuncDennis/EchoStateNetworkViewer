from __future__ import annotations

from typing import Any
import streamlit as st

import plotly.express as px
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

st.header("Show data: ")
with st.expander("Show data"):
    st.write(filtered_df)

st.header("Plot quantities: ")


# 1. Get all parameter columns
parameter_cols = [x for x in df.columns if x.startswith("P ")]
parameter_cols

# 2. Get all Metric columns:
metric_cols = [x for x in df.columns if x.startswith("M ")]

# 3. Average over all "i .." columns.
group_obj = df.groupby(parameter_cols, as_index=False)
# res = group_obj[metric_cols].mean()

# res = group_obj[metric_cols].transform("mean")
# res


res = group_obj[metric_cols].agg(["mean", "median", "std", "quantile"]).reset_index(inplace=False)

res.columns = [" ".join(a) if a[0].startswith("M ") else a[0] for a in res.columns.to_flat_index()]
res
#
# # 4. Plot:
# fig = px.line(res, x="P r_dim")
# st.plotly_chart(fig)
#
# res.columns
