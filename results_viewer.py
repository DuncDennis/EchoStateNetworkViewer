from __future__ import annotations

import streamlit as st
from PIL import Image

import plotly.graph_objects as go
import pandas as pd
import numpy as np

import src.streamlit_src.app_fragments.streamlit_utilities as utils

# Def to modify the figure:

# This is for hybrid / nonhybrid results.
def nice_fig(fig):
    yaxis_title = r'$\text{Valid Time }(\lambda_\mathrm{max} t_v)$'
    # yaxis_title = r'Valid Time'
    # xaxis_title = "r_dim"
    xaxis_title = r"$\text{Reservoir dimension}$"
    title = None

    new_legend = " "
    legend_order = ["Model only",
                    "Reservoir only",
                    "Input hybrid",
                    "Output hybrid",
                    "Full hybrid",
                    ]



    height = 500
    width = int(1.4 * height)
    # figsize = (y_size, x_size)

    xtick0 = 0
    xdtick = 500
    ytick0 = 0
    ydtick = 5
    xrange = [0, 1250]
    xrange = [0, 1050]
    # xrange = [0, 1250]
    yrange = None # [0, 15]
    yrange = [0, 15.5]
    yrange = [0, 13.5]

    font_size = 15
    legend_font_size = 11

    fig.update_yaxes(range=yrange)
    fig.update_xaxes(range=xrange)

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        yaxis_title=yaxis_title,
        xaxis_title=xaxis_title,
        xaxis=dict(
            tickmode='linear',
            tick0=xtick0,
            dtick=xdtick
        ),
        yaxis=dict(
            tickmode='linear',
            tick0=ytick0,
            dtick=ydtick
        ),

        font=dict(
            size=font_size,
            family="Times New Roman"
        ),

        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.01,  # 0.99
            xanchor="left",
            # x=0.01,
            font=dict(size=legend_font_size))
    )

    fig.write_image("test_fig.png", scale=3)
    # fig.write_image("test_fig.pdf")
    image = Image.open('test_fig.png')
    st.image(image)


# For 1D sweeps:
def onedim_vt_sweep_nice(fig, sweep_var, name=""):
    sweep_var_trans = {
        "P node_bias_scale": r"\text{Node bias scale } \sigma_\text{b}$",
        "P t_train": r"\text{Train size } N_\text{T}$",
        "P r_dim": r"\text{Reservoir dimension } r_\text{dim}$",
        "P reg_param": r"\text{Regularization parameter } \beta$",
        "P w_in_scale": r"\text{Input strength } \sigma$",
        "P n_avg_deg": r"\text{Average degree } d$",
        "P n_rad": r"\text{Spectral radius } \rho_0$",
        "P dt": r"\text{Time step of system } \Delta t$",
        "P x_train_noise_scale": r"\text{Train noise scale } \sigma_\text{T}$",
    }

    sweep_var_defaults = {
        "P node_bias_scale": 0.4,
        "P t_train": 1000,
        "P r_dim": 500,
        "P reg_param": 1e-7,
        "P w_in_scale": 1.0,
        "P n_avg_deg": 5.0,
        "P n_rad": 0.4,
        "P dt": 0.1,
        "P x_train_noise_scale": 0.0,
    }

    if sweep_var not in sweep_var_trans.keys():
        raise ValueError("sweep var not accounted for.")

    # SETTINGS:

    latex_text_size = "large"
    latex_text_size = "Large"
    # latex_text_size = "huge"
    # latex_text_size = "normalsize"

    y_range = [-0.5, 8.5]
    y0_tick = 0
    dy_tick = 2

    fig.update_yaxes(range=y_range,
                     tick0=y0_tick,
                     dtick=dy_tick)

    # vertical line:
    default_x = sweep_var_defaults[sweep_var]
    fig.add_vline(
        x=default_x,
        line_width=5,
        line_dash="dash",
        line_color="green",
        opacity=0.6
    )

    x_axis_title = sweep_var_trans[sweep_var]
    x_axis_title = rf"$\{latex_text_size}" + x_axis_title

    # y_axis_title = r"{\text{Valid time } (t_\text{v} \lambda_\text{max})}$"
    # y_axis_title = r"{\text{Valid time  }\, t_\text{v} \lambda_\text{max}}$"

    y_axis_title = r" t_\text{v} \lambda_\text{max}$"
    y_axis_title = rf"$\{latex_text_size}" + y_axis_title
    linewidth = 3
    height = 350
    width = int(2.1 * height)

    fig.update_layout(
        template="simple_white",
        # showlegend=False,
        font=dict(
            size=29,
            family="Times New Roman"
        ),
        xaxis_title=x_axis_title,
        yaxis_title=y_axis_title,
        title=None,
        width=width,
        height=height,
        margin=dict(l=5, r=5, t=5, b=5)
    )

    # fig.update_xaxes(
    #     showgrid=True,
    #     gridwidth=1,
    #     gridcolor="gray",
    # )
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="gray",
    )

    fig.update_traces(line=dict(color="Black", width=linewidth))

    # Preview:
    prev_name = "temp_preview_pic"
    prev_path = prev_name + ".png"
    fig.write_image(prev_path, scale=3)
    image = Image.open(prev_path)
    st.image(image)

    if st.button("Save fig"):

        file_path = f"results_viewer_saved_plots/lorenz_dt_0p1/{name}.png"
        fig.write_image(file_path, scale=3)
        st.write("Saved")

def transform_func(input_str: str) -> str:
    if "full" in input_str:
        out = "Full hybrid"
    elif "input" in input_str:
        out = "Input hybrid"
    elif "output" in input_str:
        out = "Output hybrid"
    elif "no" in input_str:
        out = "Reservoir only"
    elif "model" in input_str:
        out = "Model only"
    else:
        raise Exception(f"Not implemented: {input_str}")
    return out

def format_name(non_sweep_params: list, value_combination: list) -> str:
    # name = str(list(zip(non_sweep_params, value_combination)))
    name = ""
    for param_name, param_value in zip(non_sweep_params, value_combination):
        param_name_stripped = param_name.split()[-1]
        # name += f"{param_name_stripped}: {param_value}, "

        # Transform model type:
        param_value = transform_func(param_value)

        name += f"{param_value}"


    return name

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
    filtered_df: None | pd.DataFrame = None
    if file is not None:
        name_to_use = file.name.split(".")[0]

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

if filtered_df is not None:

    data_tab, sweep_tab, histogram_tab = st.tabs(["Data", "Sweep", "Histogram"])

    with data_tab:
        # SHOW TOTAL FILTERED DF:
        st.header("Show data: ")

        # Remove columns with only one value:
        parameter_cols = [x for x in filtered_df.columns if x.startswith("P ")]
        for col in parameter_cols:
            if len(filtered_df[col].unique()) == 1:
                filtered_df.drop(col, inplace=True, axis=1)

        with st.expander("Show data"):
            st.write(filtered_df)
            st.write(filtered_df.shape)

    with sweep_tab:
        try:
            # AGGREGATE RESULTS:
            st.header("Aggregate data: ")

            # All parameter columns
            parameter_cols = [x for x in filtered_df.columns if x.startswith("P ")]

            # Optionally remove all cols that only have one unique value:
            if st.checkbox("Remove all non-sweep variables", value=True):
                parameter_cols_multiple_vals = []
                for col in parameter_cols:
                    if len(filtered_df[col].unique()) == 1:
                        filtered_df.drop(col, inplace=True, axis=1)
                    else:
                        parameter_cols_multiple_vals.append(col)
                parameter_cols = parameter_cols_multiple_vals

            # Choose Metrics:
            metrics_cols = [x for x in filtered_df.columns if x.startswith("M ")]
            chosen_metrics = utils.st_selectbox_with_all("Select metrics",
                                                         metrics_cols,
                                                         default_select_all_bool=True,
                                                         disable_if_only_one_opt=True)
            # Do statistics:
            group_obj = filtered_df.groupby(parameter_cols, as_index=False)
            df_agg = group_obj[chosen_metrics].agg(stat_funcs).reset_index(inplace=False)

            with st.expander("Show aggregated data: "):
                st.table(df_agg)

            # Plotting:
            st.header("Plotting: ")

            sweep_params = [x for x in parameter_cols if len(filtered_df[x].unique()) > 1]
            sweep_param = st.selectbox("Sweep parameter", sweep_params)
            non_sweep_params = [x for x in parameter_cols if x != sweep_param]

            if len(non_sweep_params) > 0:
                other_params = df_agg.value_counts(non_sweep_params).index
                unique_non_sweep_param_values = other_params.values
            else:
                unique_non_sweep_param_values = [None]
            # Choose averaging:
            avg_mode = st.selectbox("Averaging", ["median and quartile", "mean and std"])
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

                    if value_combination is not None:
                        # Group by other sweep variables -> maybe do with GroupBy?.
                        condition_df: None | pd.DataFrame = None
                        for non_sweep_param, value in zip(non_sweep_params, value_combination):
                            condition_series = df_agg[non_sweep_param] == value
                            if condition_df is None:
                                condition_df = condition_series
                            else:
                                condition_df = condition_df & condition_series
                        sub_df = df_agg[condition_df]
                        # name = format_name(non_sweep_params, value_combination)  # ONLY ADD FOR HYBRID
                        # name="test"
                        name = str(list(zip(non_sweep_params, value_combination)))
                    else:
                        sub_df = df_agg
                        name=None
                    fig.add_trace(
                        go.Scatter(x=sub_df[sweep_param],
                                   y=sub_df[(metric, avg_str)],
                                   error_y={"array": sub_df[(metric, error_high)],
                                            "arrayminus": sub_df[(metric, error_low)]},
                                   name=name
                                   )
                    )

                fig.update_yaxes(title=metric)
                fig.update_xaxes(title=sweep_param)
                fig.update_layout(title=avg_mode)

                log_x = st.checkbox("log_x", key=metric + "log_x")
                if log_x:
                    fig.update_xaxes(type="log",
                                     exponentformat="E")

                # modify_fig(fig)
                st.plotly_chart(fig)

                if metric == "M VALIDATE VT":
                    # if st.checkbox("Plot nicely: ", key=metric):
                    #     nice_fig(fig)

                    if st.checkbox("Plot nicely 1D sweep: ", key=metric + "1d", value=True):
                        utils.st_line()
                        onedim_vt_sweep_nice(fig, sweep_var=sweep_param, name=name_to_use)
                        utils.st_line()
                utils.st_line()
        except Exception as e:
            st.error(e)

    with histogram_tab:
        df = filtered_df
        # check for parameter columns:
        remaining_param_cols = [x for x in df.columns if x.startswith("P ")]
        if len(remaining_param_cols) != 0:
            st.error("Not working")
        else:
            import plotly.figure_factory as ff

            if st.checkbox("Total hist"):
                data_list = df["M VALIDATE VT"]
                fig = ff.create_distplot([data_list],
                                         group_labels=[r"VT"],
                                         show_hist=True,
                                         show_rug=False)
                fig.update_xaxes(
                    range=[0, 14]
                )

                fig.update_layout(
                    template="simple_white",
                    showlegend=False,
                                  )

                # st.plotly_chart(fig)
                # if st.button("Save fig"):
                file_path = f"temp_hist_preview.png"
                fig.write_image(file_path, scale=3)
                # st.write("Saved")
                image = Image.open(file_path)
                st.image(image)

            if st.checkbox("sub-experiment histograms"):
                label_list = []
                data_list = []
                by = "i ens"
                by = "i train sect"
                for val in df[by].unique():
                    data_array = df[df[by] == val]["M VALIDATE VT"]
                    data_list.append(data_array)
                    label_list.append(str(val))

                fig = ff.create_distplot(data_list, label_list, show_hist=False)
                st.plotly_chart(fig)

        if st.checkbox("Violin Plot (test)", value=False):
            st.write(remaining_param_cols)
            if len(remaining_param_cols) == 1:
                x = remaining_param_cols[0]

                fig = go.Figure()
                line_colors = ["black", "green", "black"]
                for i, val in enumerate(df[x].unique()):
                    sub_df = df[df[x] == val]
                    fig.add_trace(
                        go.Violin(x=sub_df[x],
                                  y=sub_df["M VALIDATE VT"],
                                  box_visible=True,
                                  line_color=line_colors[i],
                                  points="all",
                                  marker_size=3
                                  # points=False
                    )             )
                    #
                    # data=go.Violin(
                    #     x=df[x],
                    #     y=df["M VALIDATE VT"],
                    #     box_visible=True,
                    #     line_color='black',
                    #     # line_color=['black', "green", "black"],
                    #     # line_color=['black', "green", "black"],
                    #
                    #     # meanline_visible=False
                    #     points=False,
                    #     # fillcolor='lightseagreen',
                    #     fillcolor='white',
                    #     opacity=0.6,
                    #     # x0='Total Bill'
                    # )
                # )
                # fig.add_vline(
                #     x="b",
                #     line_width=5,
                #     line_dash="dash",
                #     line_color="green",
                #     opacity=0.6
                # )
                fig.update_layout(yaxis_zeroline=False,
                                  showlegend=False, 
                                  template="simple_white",
                                  xaxis_title=r"$\Large\text{ridge regression type}$",
                                  yaxis_title=r"$\Large t_\text{v}\lambda_\text{max}$",
                                  font=dict(size=25,
                                            family="timesnewroman")
                                  )

                file_path = f"temp_violin_preview.png"
                fig.write_image(file_path, scale=3)
                # st.write("Saved")
                image = Image.open(file_path)
                st.image(image)

                st.plotly_chart(fig)

