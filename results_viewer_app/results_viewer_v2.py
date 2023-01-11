"""View the dataframe which was created with
src.ensemble_src.sweep_experiments.PredModelSweeper(<parameter_transformer>).sweep(<parameters>)

The dataframe has the following columns:
- Parameter columns of the form "P <parameter_name>".
- The three ensemble columns: "i ens", "i train sect", "i val sect"
- The measure columns:
    - "M TRAIN <train_measure_name>"
    - "M VALIDATE <predict_measure_name>"
    - Optional: "M TEST <predict_measure_name>"
"""

from __future__ import annotations

import streamlit as st

import plotly.graph_objects as go
import plotly.express as px
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
    raw_df: None | pd.DataFrame = None  # the dataframe as read from the pkl file.
    df: None | pd.DataFrame = None  # the dataframe after the sidebar prefiltering
    if file is not None:
        # The file name without the file type:
        file_name = file.name.split(".")[0]

        # Read the panbdas dataframe:
        raw_df = pd.read_pickle(file)

        # Calculate some n_ens, n_train and n_validate:
        n_ens = raw_df["i ens"].max() + 1
        n_train = raw_df["i train sect"].max() + 1
        n_validate = raw_df["i val sect"].max() + 1

        # Write ensemble sizes:
        utils.st_line()
        st.markdown(f"Ensemble size: {n_ens}")
        st.markdown(f"Train sections: {n_train}")
        st.markdown(f"Validation sections: {n_validate}")
        if "i test sect" in raw_df.columns:
            n_test = raw_df["i test sect"].max() + 1
            st.markdown(f"Test sections: {n_validate}")
        utils.st_line()

        # FILTER DF:
        st.subheader("Prefilter data: ")
        with st.expander("Filter by parameters: "):
            parameter_cols = [x for x in raw_df.columns if x.startswith("P ")]
            selections = {}
            for param_col in parameter_cols:
                unique_params = raw_df[param_col].unique()
                selection = utils.st_selectbox_with_all(param_col,
                                                        unique_params,
                                                        default_select_all_bool=True,
                                                        disable_if_only_one_opt=True,
                                                        key=f"{param_col}")
                selections[param_col] = selection

            df = raw_df.copy()
            for key, val in selections.items():
                df = df[df[key].isin(val)]
        utils.st_line()


if df is not None:

    # Tabs:
    tabs = st.tabs(["Data",
                    "Data aggregation",
                    "Plot preview (sweep)",
                    "Plot preview (violin)",
                    "PP (M vs. M scatter)",
                    "Pub plots",
                    # "Histogram"
                    ])

    # Data tab:
    with tabs[0]:
        # Write file name:
        st.write("**file name:**", file_name)

        # SHOW TOTAL FILTERED DF:
        st.header("Show data: ")
        # st.write("Shows only non-constant param cols")

        # Remove columns with only one value and save to df_use.
        df_use = df.copy()
        parameter_cols = [x for x in df_use.columns if x.startswith("P ")]
        for col in parameter_cols:
            if len(df_use[col].unique()) == 1:
                df_use.drop(col, inplace=True, axis=1)

        with st.expander("Show raw data"):
            st.write(raw_df.shape)
            st.write(raw_df)

        with st.expander("Show pre-filtered data"):
            st.write(df.shape)
            st.write(df)

        with st.expander("Show pre-filtered data with constant columns removed"):
            st.write(df_use.shape)
            st.write(df_use)

    # Data aggregation tab:
    with tabs[1]:
        # get parameter columns again:
        df_agg = df_use.copy()
        parameter_cols = [x for x in df_agg.columns if x.startswith("P ")]
        cols = st.columns(2)

        with cols[0]:

            # Choose metric:
            metrics_cols = [x for x in df_use.columns if x.startswith("M ")]
            try:
                chosen_metric = st.selectbox("Choose metric", metrics_cols, index=2)
            except:
                chosen_metric = st.selectbox("Choose metric", metrics_cols)

            df_agg = df_use.copy()

            # Aggregate all stat funcs:
            # Do statistics and save in df_agg (will have multiindex for measures):
            group_obj = df_use.groupby(parameter_cols, as_index=False)
            df_agg = group_obj[chosen_metric].agg(stat_funcs).reset_index(inplace=False)

        with cols[1]:

            # Choose averaging mode:
            avg_mode = st.selectbox("Averaging", ["median and quartile", "mean and std"])
            if avg_mode == "mean and std":
                avg_str = "mean"
                error_high_str = "std_high"
                error_low_str = "std_low"
            elif avg_mode == "median and quartile":
                avg_str = "median"
                error_high_str = "quartile_high"
                error_low_str = "quartile_low"

            avg_mode_rename = {
                avg_str: "avg",
                error_high_str: "error_high",
                error_low_str: "error_low",
            }

            df_agg.rename(avg_mode_rename, inplace=True, axis=1)

            # remove error which is not needed:
            cols_to_keep = parameter_cols
            cols_to_keep += ["avg", "error_high", "error_low"]
            df_agg = df_agg[cols_to_keep]

        st.header("Aggregated data: ")
        st.write(df_agg.shape)
        st.write(df_agg)


    # Plot preview (sweep):
    with tabs[2]:
        df_plot = df_agg.copy()
        st.write("**Chosen Metric:**", chosen_metric)
        st.write("**Chosen Average:**", avg_mode)

        parameter_cols = [x for x in df_plot.columns if x.startswith("P ")]

        cols = st.columns(2)

        with cols[0]:
            # choose x axis:
            x_param = st.selectbox("x axis param", parameter_cols)

        with cols[1]:
            # choose color:
            par_cols_remaining = parameter_cols.copy()
            par_cols_remaining.remove(x_param)
            if len(par_cols_remaining) == 0:
                color_axis = None
            else:
                color_axis = st.selectbox("color param", par_cols_remaining)

        # If there is more than 2 parameter values in total in df_agg:
        if len(parameter_cols) > 2:

            st.write("**Filter data more to have 2 remaining parameters**")
            par_cols_after_x_and_col = parameter_cols.copy()
            par_cols_after_x_and_col.remove(x_param)
            par_cols_after_x_and_col.remove(color_axis)
            cols = st.columns(len(par_cols_after_x_and_col))
            for i, p_name in enumerate(par_cols_after_x_and_col):
                with cols[i]:
                    unique_for_p = df_plot[p_name].value_counts().index
                    selected_p_val = st.selectbox(p_name, unique_for_p)
                    df_plot = df_plot[df_plot[p_name] == selected_p_val]

        with st.expander("Data to be plotted"):
            st.write(df_plot)



        fig = px.line(df_plot,
                      x=x_param,
                      error_y="error_high",
                      error_y_minus="error_low",
                      y="avg",
                      color=color_axis,
                      )

        if st.checkbox("log x", key="sweep_log_x"):
            fig.update_xaxes(type="log",
                             exponentformat="power")

        st.plotly_chart(fig)

    # Plot preview (violin):
    with tabs[3]:
        df_plot_violin = df_use.copy()
        parameter_cols = [x for x in df_use.columns if x.startswith("P ")]

        cols = st.columns(2)
        with cols[1]:
            # x axis:
            x_param = st.selectbox("x axis param", parameter_cols, key="violin")
        with cols[0]:
            # Choose metric:
            metrics_cols = [x for x in df_plot_violin.columns if x.startswith("M ")]
            try:
                chosen_metric = st.selectbox("Choose metric", metrics_cols, index=2, key="violin1")
            except:
                chosen_metric = st.selectbox("Choose metric", metrics_cols, key="violin2")

        # If there is more than 1 parameter values in total in df_agg:
        if len(parameter_cols) > 1:

            par_cols_after_x = parameter_cols.copy()
            par_cols_after_x.remove(x_param)
            cols = st.columns(len(par_cols_after_x))
            for i, p_name in enumerate(par_cols_after_x):
                with cols[i]:
                    unique_for_p = df_plot_violin[p_name].value_counts().index
                    selected_p_val = st.selectbox(p_name, unique_for_p, key=f"violin__{p_name}")
                    df_plot_violin = df_plot_violin[df_plot_violin[p_name] == selected_p_val]

        with st.expander("Data to be plotted"):
            st.write(df_plot_violin)

        # Violin plot:
        fig = go.Figure()

        if st.checkbox("log x", key="violin_log_x"):
            fig.update_xaxes(type="log",
                             exponentformat="power")

        for i, val in enumerate(df[x_param].unique()):
            sub_df = df_plot_violin[df_plot_violin[x_param] == val]
            fig.add_trace(
                go.Violin(x=sub_df[x_param],
                          y=sub_df[chosen_metric],
                          box_visible=True,
                          # line_color=line_colors[i],
                          points="all",
                          marker_size=3,
                          name=str(val),
                          # points=False
                          ))

        st.plotly_chart(fig)

    # Scatter metric vs metric and parameter as color:
    with tabs[4]:
        df_plot_scatter = df_use.copy()
        parameter_cols = [x for x in df_plot_scatter.columns if x.startswith("P ")]
        metric_cols = [x for x in df_plot_scatter.columns if x.startswith("M ")]
        df_scatter_agg = None
        if len(metric_cols) >= 2:

            # Choose metric for x and y:
            cols = st.columns(2)
            st.write("**Choose metrics for x and y axis:**")
            with cols[0]:
                x_metric = st.selectbox("xaxis metric", metric_cols, key="m_vs_m_x")
                st.write(x_metric)
            with cols[1]:
                y_metric_choice = [y for y in metric_cols if y != x_metric]
                y_metric = st.selectbox("yaxis metric", y_metric_choice, key="m_vs_m_y")



            # choose parameter for color:
            st.write("**Choose parameter for color:**")
            color_param = st.selectbox("color param", parameter_cols, key="m_vs_m_color")
            if len(parameter_cols) > 1:
                par_cols_after_col = parameter_cols.copy()
                par_cols_after_col.remove(color_param)
                cols = st.columns(len(par_cols_after_col))
                for i, p_name in enumerate(par_cols_after_col):
                    with cols[i]:
                        unique_for_p = df_plot_scatter[p_name].value_counts().index
                        selected_p_val = st.selectbox(p_name,
                                                      unique_for_p,
                                                      key=f"m_vs_m_sel__{p_name}")
                        df_plot_scatter = df_plot_scatter[df_plot_scatter[p_name] == selected_p_val]

            # aggregate the data for the chosen metrics:
            chosen_metrics = [x_metric, y_metric]
            group_obj = df_plot_scatter.groupby(parameter_cols, as_index=False)
            df_scatter_agg = group_obj[chosen_metrics].agg(stat_funcs).reset_index(inplace=False)

            # choose avg mode:
            st.write("**Choose averaging mode:**")
            avg_mode = st.selectbox("Averaging",
                                    ["median and quartile", "mean and std"],
                                    key="m_vs_m_avgmode")
            if avg_mode == "mean and std":
                avg_str = "mean"
                error_high_str = "std_high"
                error_low_str = "std_low"
            elif avg_mode == "median and quartile":
                avg_str = "median"
                error_high_str = "quartile_high"
                error_low_str = "quartile_low"

            avg_mode_rename = {
                avg_str: "avg",
                error_high_str: "error_high",
                error_low_str: "error_low",
            }

            df_scatter_agg = df_scatter_agg.iloc[:, df_scatter_agg.columns.get_level_values(
                1).isin(
                ["",
                 avg_str,
                 error_high_str,
                 error_low_str])]

            # rename columns:
            df_scatter_agg.columns = df_scatter_agg.columns.map('|'.join).str.strip('|')
            prev_cols = df_scatter_agg.columns
            new_cols = []
            for x in prev_cols:
                if str(x).endswith(avg_str):
                    new_cols.append(str(x).replace(avg_str, avg_mode_rename[avg_str]))
                elif str(x).endswith(error_high_str):
                    new_cols.append(str(x).replace(error_high_str, avg_mode_rename[error_high_str]))
                elif str(x).endswith(error_low_str):
                    new_cols.append(str(x).replace(error_low_str, avg_mode_rename[error_low_str]))
                else:
                    new_cols.append(str(x))

            df_scatter_agg.rename(columns=dict(zip(prev_cols, new_cols)), inplace=True)
            st.write(df_scatter_agg)
            to_plot = df_scatter_agg.copy()
            if st.checkbox("log color", key="m_vs_m_logcol"):
                to_plot[color_param] = np.log(to_plot[color_param])

            st.write("**Aggregated:**")
            fig = px.scatter(to_plot,
                             x=x_metric + "|" + "avg",
                             y=y_metric + "|" + "avg",
                             color=color_param,
                             # size=[marker_size] * df_scatter_agg.index.size,
                             error_x=x_metric + "|" + "error_high",
                             error_x_minus=x_metric + "|" + "error_low",
                             error_y=y_metric + "|" + "error_high",
                             error_y_minus=y_metric + "|" + "error_low",
                             color_continuous_scale="portland"
                             )

            fig.update_traces(
                marker=dict(size=12,
                            line=dict(width=1,
                                      color='DarkSlateGrey')
                            ),
                error_x=dict(thickness=1),
                error_y=dict(thickness=1)
            )

            # fig.update_layout(
            #     template="simple_white"
            # )

            st.plotly_chart(fig)

            st.write("**Each point:**")
            fig = px.scatter(df_plot_scatter,
                             x=x_metric,
                             y=y_metric,
                             color=color_param,
                             )
            st.plotly_chart(fig)


        else:
            st.write("This tab can also be used if there is a minimum of 2 Metric columns")

    # Publication plots:
    with tabs[5]:
        with st.expander("Data Frame (df_plot)"):
            st.write(df_plot)

        with st.expander("Data Frame (df_plot_violin)"):
            st.write(df_plot_violin)

        pub_tabs = st.tabs([
            "1d_valid_time_sweep",
            "2d_valid_time_sweep",
            "1d_valid_time_violin",
            "m_vs_m_scatter_plot",
        ])

        parameter_cols = [x for x in df_plot.columns if x.startswith("P ")]

        # Pub plot: 1d sweep
        with pub_tabs[0]:

            if len(parameter_cols) == 1:

                import results_viewer_app.pub_plots_src.oned_valid_time_sweep as pub_oned

                save_bool = st.button("SAVE PLOT", key="1d")

                st.write("**Preview**")
                img, path = pub_oned.onedim_vt(df_plot, name=file_name, save_bool=save_bool)

                # Preview img:
                st.image(img)

                if save_bool:
                    st.write("**Saved to:**")
                    st.write(path)
            else:
                st.write("**Only works if there is only one parameter column**")

        # Pub plot: 2d sweep
        with pub_tabs[1]:
            if len(parameter_cols) == 2:

                x_param = st.selectbox("x_param", parameter_cols, key="pub_2d")

                import results_viewer_app.pub_plots_src.twod_valid_time_sweep as pub_twod

                save_bool = st.button("SAVE PLOT", key="2d")

                plot_args = {}
                cols = st.columns(2)
                with cols[0]:
                    hide_xaxis_title = st.checkbox("Hide xaxis title", key="hide_x")
                    if hide_xaxis_title:
                        plot_args["xaxis_title"] = None
                with cols[1]:
                    yaxis_dtick = int(st.number_input("Y-axis Dtick", value=5, step=1))
                    plot_args["yaxis_dtick"] = yaxis_dtick
                # plot_args = dict(xaxis_title=None,
                #                  yaxis_dtick=2)

                st.write("**Preview**")
                img, path = pub_twod.twodim_vt(df_plot,
                                               x_param=x_param,
                                               name=file_name,
                                               save_bool=save_bool,
                                               plot_args=plot_args)

                # Preview img:
                st.image(img)

                if save_bool:
                    st.write("**Saved to:**")
                    st.write(path)
            else:
                st.write("**Only works if there are two parameter columns**")

        # Pub plot: violin 1d
        with pub_tabs[2]:

            if len(parameter_cols) == 1:

                import results_viewer_app.pub_plots_src.oned_valid_time_violin as pub_oned_violin

                save_bool = st.button("SAVE PLOT", key="1d_violin")

                st.write("**Preview**")

                img, path = pub_oned_violin.onedim_vt_violin(df_plot_violin,
                                                             name=file_name,
                                                             save_bool=save_bool)

                # Preview img:
                st.image(img)

                if save_bool:
                    st.write("**Saved to:**")
                    st.write(path)
            else:
                st.write("**Only works if there is only one parameter column**")

        # Pub plto: metric vs metric scatter plot:
        with pub_tabs[3]:
            # take df_scatter_agg and plot nicely:
            if df_scatter_agg is not None:
                import results_viewer_app.pub_plots_src.m_vs_m_scatter_plot as m_vs_m_scatter
                save_bool = st.button("SAVE PLOT", key="m_vs_m_pubsave")
                st.write("**Preview**")
                cols = st.columns(2)
                # with cols[0]:
                auto_color_ticks = st.checkbox("auto color ticks", value=True, key="m_vs_m_autotick")
                # with cols[1]:
                if not auto_color_ticks:
                    color_dtick = int(st.number_input("color dtick", value=100))
                else:
                    color_dtick = None


                img, path = m_vs_m_scatter.m_vs_m_scatter_plot(df_scatter_agg,
                                                               x_metric=x_metric,
                                                               y_metric=y_metric,
                                                               color_param=color_param,
                                                               name=file_name,
                                                               save_bool=save_bool,
                                                               color_dtick=color_dtick)

                # Preview img:
                st.image(img)

                if save_bool:
                    st.write("**Saved to:**")
                    st.write(path)
            else:
                st.write("This tab can also be used if there is a minimum of 2 Metric columns")
