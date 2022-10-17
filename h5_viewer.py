from __future__ import annotations

from typing import Any
import copy
import streamlit as st

import plotly.express as px
import pandas as pd
import numpy as np

from sklearn.model_selection import TimeSeriesSplit

import src.streamlit_src.app_fragments.streamlit_utilities as utils
import src.streamlit_src.app_fragments.raw_data as raw
import src.streamlit_src.app_fragments.esn_app_utilities as esnutils
import src.streamlit_src.app_fragments.preprocess_data as preproc
import src.streamlit_src.app_fragments.esn_build_train_predict as esn

import src.ensemble_src.sweep_experiments as sweep

file_path = "test2.h5"

metric_results = sweep.metrics_from_hdf5(file_path)

df_big = sweep.PredModelSweeper(None).to_big_pandas(metric_results)
st.write(df_big)

parameter_cols = [x for x in df_big.columns if x.startswith("P ")]
st.write(parameter_cols)

for param_col in parameter_cols:
    a = df_big[param_col].unique()
    selection = st.multiselect(param_col, a)
    st.write(a)



# fig = px.line(df_big, y="VALIDATE VT", x="r_dim", color="sync_steps")
# st.plotly_chart(fig)
# st.write(metric_results)


params_dict = sweep.results_to_param_sweep(metric_results, stripped=True)
# st.write(params_dict)


def get_data_selection(multiselect_dict: dict[str, Any],
                       metric_results: list[tuple[Any, object]]
                       ):
    """
    Get the data and parameters corresponding to the parameters specified in the
    multiselect_dict. -> Function is like a filter.

    Args:
        multiselect_dict:
        metric_results:

    Returns:

    """
    for key, val in multiselect_dict.items():
        if len(val) == 0:
            st.warning(f"Choose at least one option for {key}")
            raise Exception(f"Choose at least one option for {key}")

    data_points = []
    params_to_show = []
    for data_point in metric_results:
        select_traj = True
        params_for_point = data_point[0]
        metric_df = data_point[1]
        for key, val in params_for_point.items():
            if val not in multiselect_dict[key]:
                select_traj = False
                break
        if select_traj:
            params_this_traj = {}
            data_points.append(metric_df)
            for key, val in multiselect_dict.items():
                if len(val)>1:
                    params_this_traj[key] = params_for_point[key]
            params_to_show.append(params_this_traj)

    return data_points, params_to_show

#
# multiselect_dict = {}
# for i, (key, val) in enumerate(params_dict.items()):
#     disabled = True if len(val) == 1 else False
#     if not disabled:
#         container = st.container()
#         all = st.checkbox("Select all", key=str(i))
#         if all:
#             default = val
#         else:
#             default = val[0]
#         multiselect_dict[key] = container.multiselect(key, val, default=default, disabled=disabled)
#     else:
#         multiselect_dict[key] = st.multiselect(key, val, default=val[0], disabled=disabled)
#
# # st.write(multiselect_dict)
#
# out = get_data_selection(multiselect_dict, metric_results)
# st.write(out[0], out[1])
#
#
# def make_plottable(data_points, params_to_show):
#     df = pd.DataFrame()
#
#     for i_data, metric_df in enumerate(data_points):
#         params = params_to_show[i_data]
#
#         metric_cols = ["TRAIN MSE", "VALIDATE VT"]
#         temp = metric_df[metric_cols].mean(axis=0)
#         statistic_metric_df = temp
#
#         st.write(statistic_metric_df)
#         st.write(type(statistic_metric_df))
#         # statistic_metric_df[]
#
#         st.write(params)
#
#         # statistic_metric_df.insert
#
#
# make_plottable(data_points=out[0], params_to_show=out[1])
