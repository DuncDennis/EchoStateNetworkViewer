"""Streamlit app to predict a timeseries with an Echo State Network.
Author: Dennis Duncan [DuncDennis@gmail.com]"""

import copy

import streamlit as st

import src.streamlit_src.app_fragments.esn_app_utilities as esnutils
import src.streamlit_src.app_fragments.system_simulation as syssim
import src.streamlit_src.app_fragments.timeseries_measures as measures
import src.streamlit_src.app_fragments.pred_vs_true_plotting as pred_vs_true
import src.streamlit_src.app_fragments.system_measures as sysmeas
import src.streamlit_src.app_fragments.streamlit_utilities as utils
import src.streamlit_src.app_fragments.timeseries_plotting as plot
import src.streamlit_src.app_fragments.esn_build_train_predict as esn
import src.streamlit_src.app_fragments.esn_plotting as esnplot
import src.streamlit_src.app_fragments.preprocess_data as preproc
import src.streamlit_src.app_fragments.raw_data as raw

if __name__ == '__main__':
    st.set_page_config("Reservoir Computing Viewer", page_icon="⚡")

    with st.sidebar:
        st.header("Reservoir Computing")
        # raw_data_bool, preproc_data_bool, build_bool, train_bool, predict_bool = esnutils.st_main_checkboxes()

        status_container = st.container()

        status_dict = {"seed_bool": False,
                       "raw_data_bool": False,
                       "preproc_data_bool": False,
                       "tp_split_bool": False,
                       "build_bool": False,
                       "train_bool": False,
                       "predict_bool": False}

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
                        total_steps=total_steps,
                        default_t_train_disc_rel=2500,
                        default_t_train_sync_rel=200,
                        default_t_train_rel=5000,
                        default_t_pred_disc_rel=2500,
                        default_t_pred_sync_rel=200,
                        default_t_pred_rel=5000)
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

                x_dim = preproc_data.shape[1]
                esn_obj = esn.build(esn_type,
                                    seed=seed,
                                    x_dim=x_dim,
                                    build_args=build_args)
                esn_obj = copy.deepcopy(esn_obj)
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
                if st.checkbox("Perform training", key="Train Checkbox"):
                    y_train_fit, y_train_true, res_train_dict, esn_obj = esn.train_return_res(
                        esn_obj,
                        x_train,
                        t_train_sync,
                        )
                    esn_obj = copy.deepcopy(esn_obj)
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
                if st.checkbox("Perform prediction", key="Predict Checkbox"):
                    y_pred, y_pred_true, res_pred_dict, esn_obj = esn.predict_return_res(esn_obj,
                                                                                         x_pred,
                                                                                         t_pred_sync)
                    esn_obj = copy.deepcopy(
                        esn_obj)  # needed for the streamlit caching to work correctly.
                    status_dict[status_name] = True
            else:
                st.info(esnutils.create_needed_status_string(status_dict, status_name))
        except Exception as e:
            st.exception(e)

        utils.st_line()

        # Write status:
        with status_container:
            esnutils.st_write_status(status_dict)

    # Main Tabs:
    main_tabs = st.tabs(
        ["📼 Raw data",
         "🌀 Preprocessed data",
         "✂ Data split",
         "🛠️ Build",
         "🦾 Training",
         "🔮 Prediction",
         # "🔬 Look-under-hood"
         ])

    # raw_tab, preproc_tab, split_tab, build_tab, train_tab, predict_tab, more_tab = main_tabs
    raw_tab, preproc_tab, split_tab, build_tab, train_tab, predict_tab = main_tabs

    with raw_tab:
        status_name = "raw_data_bool"
        if status_dict[status_name]:
            time_series_dict = {"time series": data}
            st.markdown("Plot and measure the **raw data**.")
            plot_tab, measure_tab, lyapunov_tab = st.tabs(["Plot",
                                                           "Measures",
                                                           "Lyapunov Exponent"])
            with plot_tab:
                plot.st_all_timeseries_plots(time_series_dict, key="raw")

            with measure_tab:
                measures.st_all_data_measures(time_series_dict, dt=dt, key="raw")

            with lyapunov_tab:
                if data_source == "Simulate":
                    if st.checkbox("Calculate Lyapunov exponent of system"):
                        system_name = data_name
                        system_parameters = data_parameters
                        sysmeas.st_largest_lyapunov_exponent(system_name, system_parameters)
                else:
                    st.info("This feature is only available if the data is simulated from a "
                            "dynamical system. ")

        else:
            st.info(esnutils.create_needed_status_string_tab(status_name))

    with preproc_tab:
        status_name = "preproc_data_bool"
        if status_dict[status_name]:

            time_series_dict = {"time series": preproc_data}

            st.markdown("Plot and measure the **preprocessed data**.")

            plot_tab, measure_tab = st.tabs(["Plot",
                                             "Measures"])
            with plot_tab:
                plot.st_all_timeseries_plots(time_series_dict,
                                             key="preproc")

            with measure_tab:
                measures.st_all_data_measures(time_series_dict,
                                              dt=dt,
                                              key="preproc")

        else:
            st.info(esnutils.create_needed_status_string_tab(status_name))

    with split_tab:
        status_name = "tp_split_bool"
        if status_dict[status_name]:
            st.markdown("Show the **Train-Predict split:**")
            if st.checkbox("Show train-predict split"):
                plot.st_one_dim_time_series_with_sections(preproc_data,
                                                          section_steps=section_steps,
                                                          section_names=section_names)
        else:
            st.info(esnutils.create_needed_status_string_tab(status_name))

    with build_tab:
        status_name = "build_bool"
        if status_dict[status_name]:

            st.markdown("Explore the Echo State Network architecture.")
            tabs = st.tabs(["Dimensions", "Input matrix", "Network"])
            with tabs[0]:
                st.markdown("**Layer dimensions:**")
                x_dim, r_dim, r_gen_dim, y_dim = esn_obj.get_dimensions()
                esnplot.st_plot_architecture(x_dim=x_dim,
                                             r_dim=r_dim,
                                             r_gen_dim=r_gen_dim,
                                             y_dim=y_dim)
            with tabs[1]:
                w_in = esn_obj._w_in
                if st.checkbox("Input matrix as heatmap", key=f"build_tab__input_heatmap"):
                    esnplot.st_input_matrix_as_heatmap(w_in)
            with tabs[2]:
                network = esn_obj.return_network()
                esnplot.st_all_network_architecture_plots(network)

        else:
            st.info(esnutils.create_needed_status_string_tab(status_name))

    with train_tab:
        status_name = "train_bool"
        if status_dict[status_name]:
            train_data_dict = {"train true": y_train_true,
                               "train fitted": y_train_fit}
            st.markdown(
                "Compare the **training data** with the **fitted data** produced during training.")

            with st.expander("More info ..."):
                st.write(
                    "During training, the true training data and the fitted data should be very "
                    "similar. Otherwise the Echo State Network prediction is very likely to fail.")

            plot_tab, measure_tab, difference_tab = st.tabs(["Plot", "Measures", "Difference"])

            with plot_tab:
                plot.st_all_timeseries_plots(train_data_dict, key="train")
            with measure_tab:
                measures.st_all_data_measures(train_data_dict, dt=dt, key="train")
            with difference_tab:
                pred_vs_true.st_all_difference_measures(y_pred_traj=y_train_fit,
                                                        y_true_traj=y_train_true,
                                                        dt=dt,
                                                        train_or_pred="train",
                                                        key="train")
        else:
            st.info(esnutils.create_needed_status_string_tab(status_name))

    with predict_tab:
        status_name = "predict_bool"
        if status_dict[status_name]:

            pred_data_dict = {"true": y_pred_true,
                              "pred": y_pred}
            st.markdown("Compare the Echo State Network **prediction** with the **true data**.")
            plot_tab, measure_tab, difference_tab = st.tabs(["Plot", "Measures", "Difference"])
            with plot_tab:
                plot.st_all_timeseries_plots(pred_data_dict, key="predict")
            with measure_tab:
                measures.st_all_data_measures(pred_data_dict, dt=dt, key="predict")
            with difference_tab:
                pred_vs_true.st_all_difference_measures(y_pred_traj=y_pred,
                                                        y_true_traj=y_pred_true,
                                                        dt=dt,
                                                        train_or_pred="predict",
                                                        key="predict")
        else:
            st.info(esnutils.create_needed_status_string_tab(status_name))


    # with more_tab:
    #     status_name = "predict_bool"
    #     if status_dict[status_name]:
    #         st.markdown("Explore internal quantities of the Echo State Network. ")
    #
    #         res_states_tab, w_out_r_gen_tab, res_time_tab, res_dyn = st.tabs(
    #             ["Internal reservoir states", "W_out and R_gen",
    #              "Reservoir time series", "Pure reservoir dynamics"])
    #
    #         res_train_dict_no_rgen = {k: v for k, v in res_train_dict.items() if k != "r_gen"}
    #         res_pred_dict_no_rgen = {k: v for k, v in res_pred_dict.items() if k != "r_gen"}
    #
    #         with res_states_tab:
    #             esnplot.st_reservoir_state_formula()
    #
    #             if st.checkbox("Node value histograms"):
    #                 act_fct = esn_obj.get_act_fct()
    #                 esnplot.st_reservoir_states_histogram(res_train_dict_no_rgen,
    #                                                       res_pred_dict_no_rgen,
    #                                                       act_fct)
    #             utils.st_line()
    #             if st.checkbox("Node value time series", key=f"res_train_dict_no_rgen__checkbox"):
    #                 esnplot.st_reservoir_node_value_timeseries(res_train_dict_no_rgen,
    #                                                            res_pred_dict_no_rgen, )
    #
    #         with w_out_r_gen_tab:
    #             w_out = esn_obj.get_w_out()
    #             if st.checkbox("Output coupling", key="output_coupling_cb"):
    #                 st.markdown("Sum the absolute value of the W_out matrix over all generalized "
    #                             "reservoir indices, to see which output dimension has the "
    #                             "strongest coupling to the reservoir states.")
    #                 esnplot.st_plot_output_w_out_strength(w_out)
    #             utils.st_line()
    #             if st.checkbox("W_out matrix as barchart", key="w_out_as_bar"):
    #                 st.markdown(
    #                     "Show the w_out matrix as a stacked barchart, where the x axis is the "
    #                     "index of the generalized reservoir dimension.")
    #                 esnplot.st_plot_w_out_as_barchart(w_out)
    #             utils.st_line()
    #             if st.checkbox("R_gen std", key="r_gen_std"):
    #                 st.markdown(
    #                     "Show the standard deviation of the generalized reservoir state (r_gen) "
    #                     "during training and prediction.")
    #                 esnplot.st_r_gen_std_barplot(r_gen_train=res_train_dict["r_gen"],
    #                                              r_gen_pred=res_pred_dict["r_gen"])
    #             utils.st_line()
    #             if st.checkbox("R_gen std times w_out", key="r_gen_std_wout"):
    #                 st.markdown(
    #                     "Show the standard deviation of the generalized reservoir state (r_gen) "
    #                     "times w_out during training and prediction.")
    #                 esnplot.st_r_gen_std_times_w_out_barplot(r_gen_train=res_train_dict["r_gen"],
    #                                                          r_gen_pred=res_pred_dict["r_gen"],
    #                                                          w_out=w_out)
    #
    #         with res_time_tab:
    #             if st.checkbox("Reservoir states", key="r_states_3d"):
    #                 time_series_dict = {"r_train": res_train_dict["r"],
    #                                     "r_pred": res_pred_dict["r"]}
    #                 plot.st_timeseries_as_three_dim_plot(time_series_dict, key="r")
    #             utils.st_line()
    #             if st.checkbox("Generalized reservoir states", key="r_gen_states_3d"):
    #                 time_series_dict = {"r_gen_train": res_train_dict["r_gen"],
    #                                     "r_gen_pred": res_pred_dict["r_gen"]}
    #                 plot.st_timeseries_as_three_dim_plot(time_series_dict, key="r_gen")
    #
    #         with res_dyn:
    #             if st.checkbox("Largest lyapunov exponent of reservoir", key="lle_res"):
    #                 st.markdown(
    #                     "Calculate the largest lyapunov exponent from the trained reservoir "
    #                     "update equation, looping the output back into the reservoir.")
    #                 # TODO: Say that the last training reservoir state is used.
    #                 # TODO: Add Latex formula for reservoir update equation.
    #                 res_update_func = esn_obj.get_res_iterator_func()
    #                 res_starting_point = res_train_dict["r"][-1, :]
    #                 sysmeas.st_largest_lyapunov_exponent_custom(res_update_func,
    #                                                             res_starting_point,
    #                                                             dt=dt,
    #                                                             using_str="the reservoir update "
    #                                                                       "equation")
    #     else:
    #         st.info(esnutils.create_needed_status_string_tab(status_name))

