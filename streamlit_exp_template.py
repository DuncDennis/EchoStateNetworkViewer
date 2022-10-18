"""A test experiment template to show how to start an ensemble experiment. """
from __future__ import annotations

from typing import Any
import copy
import streamlit as st

import plotly.express as px
import pandas as pd
import numpy as np

import src.streamlit_src.app_fragments.streamlit_utilities as utils
import src.streamlit_src.app_fragments.raw_data as raw
import src.streamlit_src.app_fragments.esn_app_utilities as esnutils
import src.streamlit_src.app_fragments.preprocess_data as preproc
import src.streamlit_src.app_fragments.esn_build_train_predict as esn

import src.ensemble_src.sweep_experiments as sweep

# GET DATA:
st.set_page_config("RC Sweep Experimentor", page_icon="🧹")
with st.sidebar:
    st.header("Reservoir Computing for Time Series Prediction")

    status_container = st.container()

    status_dict = {"seed_bool": False,
                   "raw_data_bool": False,
                   "preproc_data_bool": False,
                   "build_bool": False}

    # Ensemble:
    st.header("Ensemble settings: ")
    n_ens = st.number_input("Ensemble size", value=5, min_value=1)

    # Random seed:
    utils.st_line()
    st.header("1. 🌱 Random seed: ")
    status_name = "seed_bool"
    try:
        seed = utils.st_seed()
        status_dict[status_name] = True
    except Exception as e:
        st.exception(e)

    # Raw data:
    utils.st_line()
    st.header("2. 📼 Create raw data: ")
    status_name = "raw_data_bool"
    try:
        data_source, (data, data_name, data_parameters, dt) = raw.st_raw_data_source()
        if data is not None:
            status_dict[status_name] = True
    except Exception as e:
        st.exception(e)

    # Preprocess data:
    utils.st_line()
    st.header("3. 🌀 Preprocess data: ")
    status_name = "preproc_data_bool"
    try:
        if esnutils.check_if_ready_to_progress(status_dict, status_name):
            preproc_data = preproc.st_all_preprocess(data, noise_seed=seed)
            status_dict[status_name] = True
        else:
            st.info(esnutils.create_needed_status_string(status_dict, status_name))
    except Exception as e:
        st.exception(e)

    # Build RC:
    utils.st_line()
    st.header("5. 🛠️ Build RC: ")
    status_name = "build_bool"
    try:
        if esnutils.check_if_ready_to_progress(status_dict, status_name):
            esn_type = esn.st_select_esn_type()

            with st.expander("Basic parameters: "):
                basic_build_args = esn.st_basic_esn_build()
            with st.expander("Network parameters: "):
                build_args = basic_build_args | esn.st_network_build_args()
            if esn_type == "ESN_r_process":
                with st.expander("Reservoir post-process layer:"):
                    build_args = build_args | esn.st_esn_r_process_args(build_args["r_dim"])
            x_dim = preproc_data.shape[1]

            build_args = {"x_dim": x_dim} | build_args
            esn_class = esn.ESN_DICT[esn_type]

            status_dict[status_name] = True
        else:
            st.info(esnutils.create_needed_status_string(status_dict, status_name))
    except Exception as e:
        st.exception(e)


# NON-TRACKED PARAMETERS:
train_data_list = [preproc_data[:1000],
                   preproc_data[1000:2000]]
validate_data_list_of_lists = [[preproc_data[1000:1500], preproc_data[1500:2000], preproc_data[2000:2500]],
                               [preproc_data[2000:2500], preproc_data[2500:3000], preproc_data[3000:3500]]]
# test_data_list = [preproc_data[3000: 3500],
#                   preproc_data[3500: 4000]]

train_sync_steps = 150
pred_sync_steps = 150

# TRACKED PARAMETERS:
parameters={
    "r_dim": [100, 200, 300, 400, 500, 600, 700, 800],
    "n_rad": [0.1, 0.5, 0.9],
}

# PARAMETER TO ARGUMENT TRANSFOMER FUNCTION:
def parameter_transformer(parameters: dict[str, float | int | str]):
    """Transform the parameters to be usable by PredModelEnsembler.

    Args:
        parameters: The parameter dict defining the sweep experiment.
            Each key value pair must be like: key is a string, value is either a string,
            int or float.

    Returns:
        All the data needed for PredModelEnsembler.
    """

    build_args["r_dim"] = parameters["r_dim"]
    build_args["n_rad"] = parameters["n_rad"]

    build_models_args = {"model_class": esn_class,
                         "build_args": build_args,
                         "n_ens": n_ens,
                         "seed": seed}

    train_validate_test_args = {
        "train_data_list": train_data_list,
        "validate_data_list_of_lists": validate_data_list_of_lists,
        "train_sync_steps": train_sync_steps,
        "pred_sync_steps": pred_sync_steps,
        "opt_validate_metrics_args": {"VT": {"dt": dt, "lle": 0.9}}
    }

    return build_models_args, train_validate_test_args

sweeper = sweep.PredModelSweeper(parameter_transformer)

if st.checkbox("RUN"):
    results_df = sweeper.sweep(parameters)
    # results_df.to_pickle("test_xyz")

    file_path = sweep.save_pandas_to_pickles(df=results_df,
                                 # name="test"
                                 )

st.write(file_path)

df = pd.read_pickle(file_path)
df
