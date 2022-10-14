from typing import Any

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

# TEMP to use decorator:
@st.experimental_memo
def build_train_predict_ensemble(n_ens: int,
                                 build_args: dict[str, Any],
                                 _esn_class: Any,
                                 seed: int,
                                 preproc_data: np.ndarray,
                                 nr_splits: int,
                                 t_train_disc: int,
                                 t_train_sync: int,
                                 t_train: int,
                                 t_pred_disc: int,
                                 t_pred_sync: int,
                                 t_pred: int,
                                 ) -> pd.DataFrame:

    ensembler = sweep.PredModelEnsemble()
    ensembler.build_models(_esn_class, n_ens=n_ens, seed=seed, **build_args)

    valid_data = preproc_data[t_train_disc:, :]
    tscv = TimeSeriesSplit(n_splits=nr_splits,
                           max_train_size=t_train_sync + t_train,
                           test_size=t_pred_sync + t_pred,
                           gap=t_pred_disc)

    print("Train Test Splits: ")
    train_list = []
    pred_list = []
    for train_index, test_index in tscv.split(valid_data):
        train, pred = valid_data[train_index], valid_data[test_index]
        train_list.append(train)
        pred_list.append(pred)
        print(train.shape, pred.shape)
        print("train range", train_index[0], train_index[-1])
        print("test range", test_index[0], test_index[-1])

    print("Train predict")
    df_list = []
    for i in range(nr_splits):
        print(f"Fold {i}")
        train_data = train_list[i]
        pred_data = pred_list[i]
        ensembler.train_models(train_data=train_data,
                               sync_steps=t_train_sync)
        ensembler.predict_models(predict_data=pred_data,
                                 sync_steps=t_pred_sync)
        metric_df = ensembler.return_pandas()
        metric_df.insert(loc=0, column="Data Fold", value=i)
        metric_df.insert(loc=0, column="ESN", value=metric_df.index)
        df_list.append(metric_df)
    metric_df_fold = pd.concat(df_list, axis=0, ignore_index=True)
    return metric_df_fold

if __name__ == '__main__':
    st.set_page_config("Reservoir Computing", page_icon="🧹")
    with st.sidebar:
        st.header("Reservoir Computing for Time Series Prediction")

        status_container = st.container()

        status_dict = {"seed_bool": False,
                       "raw_data_bool": False,
                       "preproc_data_bool": False,
                       "tp_split_bool": False,
                       "build_bool": False,
                       "train_bool": False,
                       "predict_bool": False}

        # Ensemble:
        st.header("Ensemble settings: ")
        n_ens = st.number_input("Ensemble size", value=5, min_value=1)
        nr_splits = st.number_input("Nr of splits", value=5, min_value=1)

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

        # Train-Predict split:
        utils.st_line()
        st.header("4. ✂ Train-Predict split:")
        status_name = "tp_split_bool"
        try:
            if esnutils.check_if_ready_to_progress(status_dict, status_name):
                total_steps = preproc_data.shape[0]
                split_out = \
                    esn.st_select_split_up_relative(
                        total_steps=int(0.5*total_steps),
                        default_t_train_rel=5000,
                        default_t_pred_rel=400
                    )
                if split_out is not None:
                    status_dict[status_name] = True
                    section_names = ["train disc", "train sync", "train",
                                     "pred disc", "pred sync", "pred"]
                    section_steps = list(split_out)
                    t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred = split_out
                    x_train, x_pred = esn.split_time_series_for_train_pred(preproc_data,
                                                                           *split_out)

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
                # ensembler = sweep.PredModelEnsemble()
                #
                # ensembler.build_models(esn_class, n_ens=n_ens, seed=seed, **build_args)

                status_dict[status_name] = True
            else:
                st.info(esnutils.create_needed_status_string(status_dict, status_name))
        except Exception as e:
            st.exception(e)

        # Train RC:
        utils.st_line()
        st.header("6. 🦾 Train RC: ")
        status_name = "train_bool"
        try:
            if esnutils.check_if_ready_to_progress(status_dict, status_name):
                if st.checkbox("Perform training",
                               key="Train Checkbox",
                               help="Drive the reservoir with training data and fit the "
                                    "(generalized) reservoir states to the next data step."
                               ):
                    # ensembler.train_models(train_data=x_train,
                    #                        sync_steps=t_train_sync)

                    status_dict[status_name] = True
            else:
                st.info(esnutils.create_needed_status_string(status_dict, status_name))
        except Exception as e:
            st.exception(e)


        # Predict ESN:
        utils.st_line()
        st.header("7. 🔮 Predict with RC: ")
        status_name = "predict_bool"
        try:
            if esnutils.check_if_ready_to_progress(status_dict, status_name):
                if st.checkbox("Perform prediction",
                               key="Predict Checkbox",
                               help="Synchronize the trained reservoir with real data and then "
                                    "predict the following steps. "):
                    # ensembler.predict_models(predict_data=x_pred,
                    #                          sync_steps=t_pred_sync)

                    status_dict[status_name] = True
            else:
                st.info(esnutils.create_needed_status_string(status_dict, status_name))
        except Exception as e:
            st.exception(e)

        utils.st_line()

        # Write status:
        with status_container:
            esnutils.st_write_status(status_dict)


    if st.checkbox("Do it"):
        metric_df = build_train_predict_ensemble(n_ens=n_ens,
                                                 build_args=build_args,
                                                 _esn_class=esn_class,
                                                 seed=seed,
                                                 preproc_data=preproc_data,
                                                 nr_splits=nr_splits,
                                                 t_train_disc=t_train_disc,
                                                 t_train_sync=t_train_sync,
                                                 t_train=t_train,
                                                 t_pred_disc=t_pred_disc,
                                                 t_pred_sync=t_pred_sync,
                                                 t_pred=t_pred
                                                 )
        st.write(metric_df)

        for metric in metric_df.columns:
            if metric.startswith("TRAIN") or metric.startswith("PREDICT"):
                fig = px.histogram(metric_df, x=metric, color="Data Fold", barmode="group",
                                   opacity=1)
                st.plotly_chart(fig)
                fig = px.histogram(metric_df, x=metric, color="ESN", barmode="group",
                                   opacity=1)
                st.plotly_chart(fig)
    # st.write(preproc_data.shape)
    # valid_data = preproc_data[t_pred_disc:, :]
    # nr_splits = 5
    # tscv = TimeSeriesSplit(n_splits=nr_splits,
    #                        max_train_size=5000,
    #                        test_size=500,
    #                        gap=t_pred_disc)
    # train_list = []
    # pred_list = []
    # for train_index, test_index in tscv.split(valid_data):
    #     train, pred = valid_data[train_index], valid_data[test_index]
    #     train_list.append(train)
    #     pred_list.append(pred)
    #     st.write(train.shape, pred.shape)
    #     st.write("train range", train_index[0], train_index[-1])
    #     st.write("test range", test_index[0], test_index[-1])
    #     utils.st_line()
    #
    #
    # t_train_sync = 50
    # t_pred_sync = 50
    # if st.checkbox("train and predict for ensemble and validation"):
    #     df_list = []
    #     for i in range(nr_splits):
    #         print(f"Fold {i}")
    #         train_data = train_list[i]
    #         pred_data = pred_list[i]
    #         ensembler.train_models(train_data=train_data,
    #                                sync_steps=t_train_sync)
    #         ensembler.predict_models(predict_data=pred_data,
    #                                  sync_steps=t_pred_sync)
    #         metric_df = ensembler.return_pandas()
    #         metric_df.insert(loc=0, column="Data Fold", value=i)
    #         metric_df.insert(loc=0, column="ESN", value=metric_df.index)
    #         # metric_df["ESN"] = metric_df.index
    #         # metric_df["Data Fold"] = i
    #         df_list.append(metric_df)
    #
    #     metric_df_fold = pd.concat(df_list, axis=0, ignore_index=True)
    #     st.table(metric_df_fold)

    # metric_df = PredEnsemble.return_pandas()
    # st.table(metric_df)



