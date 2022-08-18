"""Python file that includes Streamlit elements used for plotting esn quantities."""

from __future__ import annotations

from typing import Callable

import numpy as np
import networkx as nx
import streamlit as st

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

from src.streamlit_src.generalized_plotting import plotly_plots as plpl
from src.streamlit_src.app_fragments import streamlit_utilities as utils
from src.streamlit_src.app_fragments import timeseries_measures as meas_app
from src.streamlit_src.app_fragments import esn_app_utilities as esnutils
from src.streamlit_src.app_fragments import timeseries_plotting as plot
import src.streamlit_src.latex_formulas.esn_formulas as esn_latex


def st_plot_w_out_as_barchart(w_out: np.ndarray, key: str | None = None) -> None:
    """Streamlit element to plot w_out as a barchart with r_gen_dim as x_axis.

    TODO: add bargap as a option in matrix_as_barchart?

    Args:
        w_out: The w_out matrix of shape (output dimension, r_gen dimension).
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    log_y = st.checkbox("log y", key=f"{key}__st_plot_w_out_as_barchart__logy")
    fig = plpl.matrix_as_barchart(w_out.T, x_axis="r_gen index", y_axis="out dim",
                                  value_name="w_out", log_y=log_y)
    fig.update_layout(bargap=0.0)
    st.plotly_chart(fig)


def st_plot_output_w_out_strength(w_out: np.ndarray, key: str | None = None) -> None:
    """Streamlit element to plot the strength of w_out for each output component.
    TODO: remove key argument?
    Args:
        w_out: The w_out matrix of shape (output dimension, r_gen dimension).
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    utils.st_line()
    cols = st.columns(2)
    with cols[0]:
        st.latex(esn_latex.w_out_sum_over_r_gen_left)
    with cols[1]:
        st.latex(esn_latex.w_out_sum_over_r_gen_right)
    utils.st_line()

    out_dim = w_out.shape[0]

    w_out_summed = np.sum(np.abs(w_out), axis=1)
    x = np.arange(out_dim)
    fig = px.bar(x=x, y=w_out_summed)
    fig.update_xaxes(title="Output dimension i", tickvals=x)
    fig.update_yaxes(title="a_i")
    fig.update_layout(title="Summed abs. w_out entries vs. output dimension")

    st.plotly_chart(fig)


def st_plot_architecture(x_dim: int, r_dim: int, r_gen_dim: int, y_dim: int) -> None:
    """Streamlit element to plot dimensions of the layers in the esn.

    Args:
        x_dim: The input dimension of the esn.
        r_dim: The reservoir dimension.
        r_gen_dim: The generalized reservoir dimension.
        y_dim: The output dimension.

    """
    utils.st_line()
    cols = st.columns(7)
    cols[0].markdown("**Input:**")

    cols[1].latex(r"\rightarrow")

    cols[2].markdown("**Reservoir states:**")

    cols[3].latex(r"\rightarrow")

    cols[4].markdown("**Generalized res. states:**")

    cols[5].latex(r"\rightarrow")
    cols[6].markdown("**Output:**")

    cols = st.columns(7)

    cols[0].markdown(f"**{x_dim}**")

    cols[2].markdown(f"**{r_dim}**")

    cols[4].markdown(f"**{r_gen_dim}**")

    cols[6].markdown(f"**{y_dim}**")

    utils.st_line()


def st_reservoir_state_formula() -> None:
    """Streamlit element to plot the reservoir update equation and describe the terms. """
    st.markdown("**Reservoir update equation:**")
    st.latex(esn_latex.w_in_and_network_update_equation_with_explanation)
    utils.st_line()


def st_reservoir_states_histogram(res_train_dict: dict[str, np.ndarray],
                                  res_pred_dict: dict[str, np.ndarray],
                                  act_fct: Callable[[np.ndarray], np.ndarray] | None,
                                  key: str | None = None) -> None:
    """Streamlit element to show histograms of reservoir state quantities.

    TODO: Bad practice that res_train_dict has one item more than actually needed?
    TODO: Also include bias and r_to_r_gen?

    Show value histograms of:
    - Res. input: W_in * INPUT
    - Res. internal update: Network * PREVIOUS_RES_STATE
    - Act. fct. argument: W_in * INPUT + Network * PREVIOUS_RES_STATE + BIAS.
    - Res. states: The reservoir states.

    All values are first flattened over all nodes and then the histogram is created.

    Args:
        res_train_dict: A dictionary containing "r_input", "r_internal", "r_act_fct_inp", "r"
                        corresponding to the reservoir state quantities during training.
        res_pred_dict: A dictionary containing "r_input", "r_internal", "r_act_fct_inp", "r"
                       corresponding to the reservoir state quantities during prediction.
        act_fct: The activation function used in the esn.
        key: Provide a unique key if this streamlit element is used multiple times.

    """

    st.markdown("**Histograms of terms in update equation:**")
    cols = st.columns(3)
    with cols[0]:
        train_or_predict = esnutils.st_train_or_predict_select(
            key=f"{key}__st_reservoir_states_histogram")
        # train_or_predict = st.selectbox("Train or predict", ["train", "predict"],
        #                                 key=f"{key}__st_reservoir_states_histogram__top")
    with cols[1]:
        bins = int(st.number_input("Bins", min_value=2, value=50,
                                   key=f"{key}__st_reservoir_states_histogram__bins"))
    with cols[2]:
        share_x = st.checkbox("Share x", key=f"{key}__st_reservoir_states_histogram__sharex")
        share_y = st.checkbox("Share y", key=f"{key}__st_reservoir_states_histogram__sharey")
        if share_x:
            share_x = "all"
        if share_y:
            share_y = "all"

    if train_or_predict == "train":
        res_state_dict = res_train_dict
    elif train_or_predict == "predict":
        res_state_dict = res_pred_dict
    else:
        raise ValueError("This train or predict option is not accounted for.")

    res_state_dict_flattened = {key: val.flatten()[:, np.newaxis] for key, val in
                                res_state_dict.items()}  # if key != "r_gen"
    df = meas_app.get_histograms(res_state_dict_flattened, dim_selection=[0], bins=bins)

    fig = make_subplots(rows=2, cols=2, shared_xaxes=share_x, shared_yaxes=share_y,
                        subplot_titles=["Res. input", "Res. internal update", "Act. fct. argument",
                                        "Res. states"],
                        specs=[[{}, {}],
                               [{"secondary_y": True}, {}]],
                        horizontal_spacing=0.1,
                        vertical_spacing=0.2)

    df_sub = df[df["label"] == "r_input"]
    fig.add_trace(
        go.Bar(x=df_sub["bins"], y=df_sub["histogram"], showlegend=False),
        row=1, col=1
    )

    df_sub = df[df["label"] == "r_internal"]
    fig.add_trace(
        go.Bar(x=df_sub["bins"], y=df_sub["histogram"], showlegend=False),
        row=1, col=2
    )

    df_sub = df[df["label"] == "r_act_fct_inp"]
    fig.add_trace(
        go.Bar(x=df_sub["bins"], y=df_sub["histogram"], showlegend=False),
        row=2, col=1
    )
    bins = df_sub["bins"]
    fig.add_trace(
        go.Scatter(x=bins, y=act_fct(bins), showlegend=True, name="activation function",
                   mode="lines"),
        secondary_y=True,
        row=2, col=1
    )

    df_sub = df[df["label"] == "r"]
    fig.add_trace(
        go.Bar(x=df_sub["bins"], y=df_sub["histogram"], showlegend=False),
        row=2, col=2
    )

    fig.update_layout(bargap=0.0)
    fig.update_layout(legend=dict(
        orientation="h",
        yanchor="bottom",
        y=-0.15,
        xanchor="left")
    )
    fig.update_layout(width=750, height=500)

    st.plotly_chart(fig)


def st_reservoir_node_value_timeseries(res_train_dict: dict[str, np.ndarray],
                                       res_pred_dict: dict[str, np.ndarray],
                                       key: str | None = None) -> None:
    """Streamlit element to plot reservoir node value time series.

    # TODO: Make nicer?

    One can select the dimension to plot for train and predict.

    Args:
        res_train_dict: A dictionary containing "r_input", "r_internal", "r_act_fct_inp", "r"
                        corresponding to the reservoir state quantities during training.
        res_pred_dict: A dictionary containing "r_input", "r_internal", "r_act_fct_inp", "r"
                       corresponding to the reservoir state quantities during prediction.
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    st.markdown("**Training**")
    plot.st_plot_dim_selection(res_train_dict,
                               key=f"{key}__st_reservoir_node_value_timeseries__train")
    st.markdown("**Prediction**")
    plot.st_plot_dim_selection(res_pred_dict,
                               key=f"{key}__st_reservoir_node_value_timeseries__predict")


def st_esn_network_as_heatmap(network: np.ndarray) -> None:
    """Streamlit element to plot the network matrix as a heatmap.

    Args:
        network: The numpy array of shape (reservoir dimension, reservoir dimension).
    """
    st.markdown("**The networks adjecency matrix:**")
    fig = px.imshow(network)
    fig.update_xaxes(title="reservoir dimension")
    fig.update_yaxes(title="reservoir dimension")
    st.plotly_chart(fig)


def st_esn_network_measures(network: np.ndarray) -> None:
    """Streamlit element to measure some network quantities of the provided network.

    Args:
        network: The numpy array of shape (reservoir dimension, reservoir dimension).
    """
    nw_nx = nx.from_numpy_matrix(network, create_using=nx.DiGraph)
    indeg = np.mean([x[1] for x in nw_nx.in_degree])
    outdeg = np.mean([x[1] for x in nw_nx.out_degree])

    cols = st.columns(2)
    with cols[0]:
        st.markdown("**Mean network in-degree:**")
        st.write(indeg)
    with cols[1]:
        st.markdown("**Mean network out-degree:**")
        st.write(outdeg)


def st_esn_network_eigenvalues(network: np.ndarray) -> None:
    """Streamlit element to plot the network eigenvalues.

    # TODO: Maybe add caching?

    Args:
        network: The numpy array of shape (reservoir dimension, reservoir dimension).
    """
    eigenvalues = np.linalg.eig(network)[0]
    abs_eigenvalues = np.abs(eigenvalues)
    # st.write(eigenvalues)

    st.markdown(f"**Largest eigenvalue = {np.round(abs_eigenvalues[0], 4)}**")

    fig = px.line(abs_eigenvalues)
    fig.update_xaxes(title="eigenvalue number")
    fig.update_yaxes(title="absoulte value of eigenvalue")
    st.plotly_chart(fig)


def st_input_matrix_as_heatmap(w_in: np.ndarray) -> None:
    """Streamlit element to plot the input matrix as a heatmap.

    Args:
        w_in: The numpy array of shape (reservoir dimension, input dimension).
    """
    st.markdown("**The input matrix W_in:**")
    fig = px.imshow(w_in, aspect="auto")
    fig.update_xaxes(title="input dimension", tickvals=np.arange(w_in.shape[1]))
    fig.update_yaxes(title="reservoir dimension")
    st.plotly_chart(fig)


def st_all_network_architecture_plots(network: np.ndarray,
                                      key: str | None = None) -> None:
    """Streamlit element to show all network architecture plots.

    Args:
        network: The numpy array of shape (reservoir dimension, reservoir dimension).
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    if st.checkbox("Network matrix as heatmap",
                   key=f"{key}__st_all_network_architecture_plots__hm"):
        st_esn_network_as_heatmap(network)
    utils.st_line()
    if st.checkbox("Network degree",
                   key=f"{key}__st_all_network_architecture_plots__deg"):
        st_esn_network_measures(network)
    utils.st_line()
    if st.checkbox("Network eigenvalues",
                   key=f"{key}__st_all_network_architecture_plots__eig"):
        st_esn_network_eigenvalues(network)


def st_r_gen_std_barplot(r_gen_train: np.ndarray, r_gen_pred: np.ndarray, key: str | None = None
                         ) -> None:
    """Streamlit element to show the std of r_gen_train and r_gen_pred.

    Args:
        r_gen_train: The generalized reservoir states during training of shape (train time steps,
                    r_gen dimension).
        r_gen_pred: The generalized reservoir states during prediction of shape (predict time
                    steps, r_gen dimension).
        key: Provide a unique key if this streamlit element is used multiple times.

    """

    log_y = st.checkbox("log y", key=f"{key}__st_r_gen_std_barplot__logy")
    r_gen_dict = {"r_gen_train": r_gen_train, "r_gen_pred": r_gen_pred}
    out = meas_app.get_statistical_measure(r_gen_dict, mode="std")

    fig = px.bar(out, x="x_axis", y="std", color="label", log_y=log_y, barmode="group")
    fig.update_xaxes(title="r_gen index")
    fig.update_layout(bargap=0.0)
    if log_y:
        fig.update_yaxes(type="log", exponentformat="E")
    st.plotly_chart(fig)


def st_r_gen_std_times_w_out_barplot(r_gen_train: np.ndarray, r_gen_pred: np.ndarray,
                                     w_out: np.ndarray, key: str | None = None
                                     ) -> None:
    """Streamlit element to show the std of r_gen_train and r_gen_pred.

    # TODO: add latex explanation.

    Args:
        r_gen_train: The generalized reservoir states during training of shape (train time steps,
                    r_gen dimension).
        r_gen_pred: The generalized reservoir states during prediction of shape (predict time
                    steps, r_gen dimension).
        w_out: The output matrix w_out of shape (output dimension, r_gen dimension).
        key: Provide a unique key if this streamlit element is used multiple times.

    """

    log_y = st.checkbox("log y", key=f"{key}__st_r_gen_std_times_w_out_barplot__logy")
    abs_w_out_per_r_gen_dim = np.sum(np.abs(w_out), axis=0)

    r_gen_times_wout_dict = {"r_gen_train_times_wout": r_gen_train * abs_w_out_per_r_gen_dim,
                             "r_gen_pred_times_wout": r_gen_pred * abs_w_out_per_r_gen_dim}
    out = meas_app.get_statistical_measure(r_gen_times_wout_dict, mode="std")

    fig = px.bar(out, x="x_axis", y="std", color="label", log_y=log_y, barmode="group")
    fig.update_layout(bargap=0.0)
    fig.update_xaxes(title="r_gen index")

    if log_y:
        fig.update_yaxes(type="log", exponentformat="E")
    st.plotly_chart(fig)
