"""Python file that includes Streamlit elements used for plotting esn quantities."""

from __future__ import annotations

import copy
from typing import Callable

import numpy as np
import networkx as nx
import streamlit as st
from sklearn.decomposition import PCA
import pandas as pd

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.express as px

import src.esn_src.measures as resmeas
import src.esn_src.utilities

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

    cols[4].markdown("**Generalized res. states (readout):**")

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

def st_plot_network(network: np.ndarray) -> None:
    import matplotlib.pyplot as plt

    # net_dim = network.shape[0]
    # network_mask = np.ones((net_dim, net_dim))
    # network_mask[network == 0] = 0
    # if np.all(network_mask.T == network_mask): # if directed
    #     directed = False
    #     nw_nx = nx.from_numpy_matrix(network_mask, create_using=nx.Graph)
    # else:
    #     directed = True
    #     nw_nx = nx.from_numpy_matrix(network_mask, create_using=nx.DiGraph)


    # nw_nx = nx.from_numpy_matrix(network, create_using=nx.DiGraph)
    nw_nx = nx.from_numpy_matrix(network, create_using=nx.Graph)

    degrees = np.array([x[1] for x in nw_nx.degree()])
    degrees_color = list(3.0 * degrees)
    degrees_size = list(10.0 * degrees)
    fig = plt.figure()
    ax = plt.gca()
    with src.esn_src.utilities.temp_seed(1):
        nx.draw(nw_nx,
                ax=ax,
                # node_size=10,
                # node_color=degrees_color,
                node_size=degrees_size,
                alpha=0.5)
    st.pyplot(fig)

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
    if st.checkbox("Plot network",
                   key=f"{key}__st_all_network_architecture_plots__plot"):
        st_plot_network(network)
    utils.st_line()
    if st.checkbox("Network degree",
                   key=f"{key}__st_all_network_architecture_plots__deg"):
        st_esn_network_measures(network)
    utils.st_line()
    if st.checkbox("Network eigenvalues",
                   key=f"{key}__st_all_network_architecture_plots__eig"):
        st_esn_network_eigenvalues(network)



# def st_r_gen_std_barplot(r_gen_train: np.ndarray, r_gen_pred: np.ndarray, key: str | None = None
#                          ) -> None:
#     """Streamlit element to show the std of r_gen_train and r_gen_pred.
#     # TODO: maybe make more general?
#     Args:
#         r_gen_train: The generalized reservoir states during training of shape (train time steps,
#                     r_gen dimension).
#         r_gen_pred: The generalized reservoir states during prediction of shape (predict time
#                     steps, r_gen dimension).
#         key: Provide a unique key if this streamlit element is used multiple times.
#
#     """
#
#     log_y = st.checkbox("log y", key=f"{key}__st_r_gen_std_barplot__logy")
#     r_gen_dict = {"r_gen_train": r_gen_train, "r_gen_pred": r_gen_pred}
#     out = meas_app.get_statistical_measure(r_gen_dict, mode="std")
#
#     fig = px.bar(out, x="x_axis", y="std", color="label", log_y=log_y, barmode="group")
#     fig.update_xaxes(title="r_gen index")
#     fig.update_layout(bargap=0.0)
#     if log_y:
#         fig.update_yaxes(type="log", exponentformat="E")
#     st.plotly_chart(fig)


def get_r_gen_times_w_out_terms(r_gen_states: np.ndarray,
                                w_out: np.ndarray) -> np.ndarray:
    """Function to calculate the individual terms in the output prediction.

    Every r_gen dimension is connected with its w_out entry for each output dimension.

    Args:
        r_gen_states: The generalized reservoir states of shape (time_steps, r_gen_dim).
        w_out: The w_out matrix of shape (out_dim, r_gen_dim).

    Returns:
        The individual output terms of shape (steps, out_dim, r_gen_dim).
    """
    steps, r_gen_dim = r_gen_states.shape
    out_dim = w_out.shape[0]
    results = np.zeros((steps, out_dim, r_gen_dim))
    for i_r_gen in range(r_gen_dim):
        for i_out_dim in range(out_dim):
            results[:, i_out_dim, i_r_gen] = r_gen_states[:, i_r_gen] * w_out[i_out_dim, i_r_gen]
    return results


def st_r_gen_times_w_out_stat_measure(r_gen_dict: dict[str, np.ndarray],
                                      w_out: np.ndarray,
                                      key: str | None = None) -> None:
    """Streamlit element to plot statistical quantities of r_gen times w_out.

    Args:
        r_gen_dict: A dictionary containing r_gen_states of shape (time steps, r_gen dim).
        w_out: The output matrix of shape (out_dim, r_gen_dim).
        key: Provide a unique key if this streamlit element is used multiple times.

    """
    out_dim = w_out.shape[0]
    cols = st.columns(2)
    with cols[0]:
        mode = st.selectbox("Choose data or dimension", ["data", "dimension"],
                            key=f"{key}__st_r_gen_stat_measure_times_w_out__overlay")

    r_gen_w_out_dict = {key: get_r_gen_times_w_out_terms(r_gen, w_out) for key, r_gen in
                        r_gen_dict.items()}

    if mode == "dimension":
        with cols[1]:
            out_dim = int(st.number_input("Select output dimension",
                                          value=0,
                                          max_value=out_dim - 1,
                                          min_value=0,
                                          key=f"{key}__st_r_gen_stat_measure_times_w_out__dim"))
        dict_use = {k: v[:, out_dim, :] for k, v in r_gen_w_out_dict.items()}

    elif mode == "data":
        with cols[1]:
            selection = st.selectbox("Select data", list(r_gen_dict.keys()),
                                     key=f"{key}__st_r_gen_stat_measure_times_w_out__tp")
        r_gen_w_out = r_gen_w_out_dict[selection]

        dict_use = {}
        for i_out_dim in range(out_dim):
            dict_use[f"out dim {i_out_dim}"] = r_gen_w_out[:, i_out_dim, :]
    else:
        raise ValueError("This mode is not accounted for. ")

    meas_app.st_statistical_measures(dict_use,
                                     key=f"{key}__st_r_gen_stat_measure_times_w_out__meas",
                                     bar_or_line="line",
                                     x_label="r_gen_dim",
                                     default_measure="std",
                                     default_abs=False,
                                     default_log_y=True)


def st_scatter_matrix_plot(res_train_dict: dict[str, np.ndarray],
                           res_pred_dict: dict[str, np.ndarray],
                           key: str | None = None) -> None:
    """Streamlit element to produce a scatter matrix plot for train/pred reservoir states.

    One can decide weather to plot the train or predict reservoir states.
    One can decide which reservoir states to plot: "r_act_fct_inp", "r_internal", "r_input", "r"
    or "r_gen".
    One can decide weather to perform a pca on the data or not.

    Args:
        res_train_dict: The dictionary of train reservoir states with the keys: "r_act_fct_inp",
                        "r_internal", "r_input", "r" and "r_gen".
        res_pred_dict: The dictionary of prediction reservoir states with the keys:
                        "r_act_fct_inp", "r_internal", "r_input", "r" and "r_gen".
        key: Provide a unique key if this streamlit element is used multiple times.

    """

    st.markdown("Plot a selection of dimension (or pca dimensions) for a reservoir state type.")

    cols = st.columns(2)
    with cols[0]:
        train_or_predict = st.selectbox("Train / predict", ["train", "predict"],
                                        key=f"{key}__st_scatter_matrix_plot__train_pred")
    with cols[1]:
        res_type = st.selectbox("Reservoir state type", ["r", "r_input", "r_internal",
                                                         "r_act_fct_inp", "r_gen"],
                                key=f"{key}__st_scatter_matrix_plot__res_type")

    if train_or_predict == "train":
        res_dict = res_train_dict
    elif train_or_predict == "predict":
        res_dict = res_pred_dict
    else:
        raise ValueError("This train or predict option should not exist. ")

    res_states = res_dict[res_type]
    res_state_dim = res_states.shape[1]
    cols = st.columns(2)
    with cols[0]:
        min_dim = st.number_input("Min dimension", value=0, min_value=0,
                                  max_value=res_state_dim - 1,
                                  key=f"{key}__st_scatter_matrix_plot__min_dim")
    with cols[1]:
        max_dim = st.number_input("Max dimension", value=min_dim + 4, min_value=min_dim + 1,
                                  max_value=res_state_dim - 1,
                                  key=f"{key}__st_scatter_matrix_plot__max_dim")
    pca_bool = st.checkbox("Perform pca", value=True,
                           key=f"{key}__st_scatter_matrix_plot__pcabool")

    fig = get_scatter_matrix_plot(states=res_states, min_dim=min_dim, max_dim=max_dim,
                                  pca_bool=pca_bool)
    st.plotly_chart(fig)


@st.experimental_memo
def get_scatter_matrix_plot(states: np.ndarray, min_dim: int = 0, max_dim: int = 4,
                            pca_bool: bool = False) -> go.Figure:
    """Function to produce a scatter matrix plot of a selection of dims/pca comps of states.

    Args:
        states: The input array of shape (time_steps, state_dimension).
        min_dim: The minimal dimension to plot.
        max_dim: The maximal dimension to plot.
        pca_bool: Weather to perform PCA or not before plotting the dimensions.

    Returns:
        A plotly figure with (max_dim - min_dim)^2 subplots, each a 2D scatter plot.
    """
    if pca_bool:
        pca = PCA()
        components = pca.fit_transform(states)
        labels = {
            str(i): f"PC {i + 1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }
        to_plot = components

    else:
        labels = {
            str(i): f"Dim {i + 1}" for i in range(states.shape[1])
        }
        to_plot = states
    dimensions = range(min_dim, max_dim)

    fig = px.scatter_matrix(
        to_plot,
        labels=labels,
        dimensions=dimensions,
    )
    fig.update_traces(marker={'size': 1})
    fig.update_traces(diagonal_visible=False)
    return fig


def st_all_w_out_r_gen_plots(r_gen_dict: dict[str, np.ndarray],
                             w_out: np.ndarray,
                             key: str | None = None,
                             ) -> None:
    """Streamlit element to plot all w_out and r_gen plots on one page.

    Args:
        r_gen_dict: A dictionary containing r_gen_states of shape (time steps, r_gen dim).
        w_out: The output matrix of shape (out_dim, r_gen_dim).
        key: Provide a unique key if this streamlit element is used multiple times.

    """

    if st.checkbox("Output coupling", key=f"{key}__st_all_w_out_r_gen_plots__occ"):
        st.markdown("Sum the absolute value of the W_out matrix over all generalized "
                    "reservoir indices, to see which output dimension has the "
                    "strongest coupling to the reservoir states.")
        st_plot_output_w_out_strength(w_out, key=f"{key}__st_all_w_out_r_gen_plots__os")

    utils.st_line()
    if st.checkbox("Wout matrix as barchart", key=f"{key}__st_all_w_out_r_gen_plots__wbp"):
        st.markdown(
            r"Show the $W_\text{out}$ matrix as a stacked barchart, where the x axis is the "
            "index of the generalized reservoir dimension.")
        st_plot_w_out_as_barchart(w_out, key=f"{key}__st_all_w_out_r_gen_plots__wbp2")

    utils.st_line()
    if st.checkbox("Statistical measures on Rgen", key=f"{key}__st_all_w_out_r_gen_plots__smr"):
        st.markdown(
            "Show the standard deviation of the generalized reservoir state (r_gen) "
            "during training and prediction.")

        meas_app.st_statistical_measures(r_gen_dict,
                                         bar_or_line="line",
                                         x_label="r_gen_dim",
                                         default_abs=False,
                                         default_log_y=True,
                                         default_measure="std",
                                         key=f"{key}__st_all_w_out_r_gen_plots__smr1")

    utils.st_line()
    if st.checkbox("Statistical measures on Rgen times Wout",
                   key=f"{key}__st_all_w_out_r_gen_plots__smrw"):
        st.markdown(
            "Show the standard deviation of the generalized reservoir state (r_gen) "
            "times w_out during training and prediction.")
        st.latex(
            r"""
            r_\text{gen}[i] \times W_\text{out}[i, j],\qquad  i: \text{r gen dim}, j:
            \text{out dim}
            """)
        st_r_gen_times_w_out_stat_measure(r_gen_dict, w_out,
                                          key=f"{key}__st_all_w_out_r_gen_plots__smr2")

    utils.st_line()
    if st.checkbox("Mean frequency of Rgen states",
                   key=f"{key}__st_all_w_out_r_gen_plots__mf"):
        st.markdown("**Plot the mean frequency of the r_gen components.**")
        meas_app.st_mean_frequency(r_gen_dict,
                                   x_label="r_gen_dim")


def st_dist_in_std_for_r_gen_states(r_gen_train: np.ndarray,
                                    r_gen_pred: np.ndarray,
                                    save_session_state: bool = False):

    dist_in_std_log = resmeas.distance_in_std(x=r_gen_train,
                                              y=r_gen_pred,
                                              log_bool=True)
    dist_in_std = resmeas.distance_in_std(x=r_gen_train,
                                          y=r_gen_pred,
                                          log_bool=False)
    if save_session_state:
        utils.st_add_to_state_category("dist in std", "MEASURES", dist_in_std)
        utils.st_add_to_state_category("dist in std log", "MEASURES", dist_in_std_log)

    st.markdown(
        r"""
        Calculate the difference in the standard deviation of r_gen_pred and r_gen_train. 
        """
    )

    st.latex(
        r"""
        \|f(\text{std}_\text{time}(r_\text{gen, train})) - f(\text{std}_\text{time}(r_\text{gen, pred}))\|
        """
    )

    st.markdown(
        r"""
        When $f = \log$:
        """
    )
    st.write("Dist in std with log", dist_in_std_log)

    st.markdown(
        r"""
        When $f = \text{Id}$:
        """
    )
    st.write("Dist in std without log", dist_in_std)


def st_investigate_partial_w_out_influence(r_gen_train: np.ndarray,
                                           x_train: np.ndarray,
                                           t_train_sync: int,
                                           w_out: np.ndarray,
                                           key: str | None = None) -> None:

    st.markdown(
        r"""
        Take the esn r_gen states used for training and split the states at some 
        r_gen dimensions. For each side of the split (i.e. the first and the last r_gen 
        states) calculate the partial output that they produce. 
        """
    )

    r_gen_states = r_gen_train
    r_gen_dim = r_gen_states.shape[1]
    sys_dim = x_train.shape[1]
    inp = x_train[t_train_sync:-1, :]
    out = x_train[t_train_sync + 1:, :]
    diff = out - inp
    w_out = copy.deepcopy(w_out)

    i_rgen_dim_split = int(
        st.number_input("split r_gen_dim", value=sys_dim, min_value=0,
                        max_value=r_gen_dim - 1,
                        key=f"{key}__st_investigate_partial_w_out_influence__srgd"))

    r_gen_first = r_gen_states[:, :i_rgen_dim_split]
    r_gen_last = r_gen_states[:, i_rgen_dim_split:]
    esn_output_first = (w_out[:, :i_rgen_dim_split] @ r_gen_first.T).T
    esn_output_last = (w_out[:, i_rgen_dim_split:] @ r_gen_last.T).T

    # to_corr_esn_dict = {"esn output first": esn_output_first,
    #                     "esn output last": esn_output_last}

    st.write("nr of first r_gen dims", r_gen_first.shape[1])
    st.write("nr of last r_gen dims", r_gen_last.shape[1])

    st.markdown(r"**Partial ESN output:**")

    to_plot_esn_dict = {"esn output first": esn_output_first,
                        "esn output last": esn_output_last,
                        "esn output summed": esn_output_first + esn_output_last}
    with st.expander("Show: "):
        plot.st_default_simulation_plot_dict(to_plot_esn_dict)

    st.markdown(r"**Real input, output and difference:**")
    to_plot_real_dict = {"real input": inp,
                         "real output": out,
                         "real difference": diff}
    with st.expander("Show: "):
        plot.st_default_simulation_plot_dict(to_plot_real_dict)

    st.markdown(r"**All in one:**")
    to_plot_all = to_plot_esn_dict | to_plot_real_dict
    with st.expander("Show: "):
        plot.st_default_simulation_plot_dict(to_plot_all)
        plot.st_plot_dim_selection(to_plot_all,
                                   key=f"{key}__st_investigate_partial_w_out_influence__ds")

    st.markdown(r"**Dimension wise correlation:**")
    if st.checkbox("Dimension wise correlation: ",
                   key=f"{key}__st_investigate_partial_w_out_influence__dwc"):

        corr_data_dict = {"correlation x": [], "correlation y": [],
                          "corr value": [], "dim": []}
        for x_name, x_data in to_plot_real_dict.items():
            for y_name, y_data in to_plot_esn_dict.items():
                corr_multidim = correlate_input_and_r_gen(x_data,
                                                          y_data,
                                                          time_delay=0)

                for i_dim in range(corr_multidim.shape[0]):
                    corr_value = corr_multidim[i_dim, i_dim]
                    corr_data_dict["dim"].append(i_dim)
                    corr_data_dict["correlation x"].append(x_name)
                    corr_data_dict["correlation y"].append(y_name)
                    corr_data_dict["corr value"].append(corr_value)
        df_corr = pd.DataFrame.from_dict(corr_data_dict)

        fig = px.bar(df_corr, x="correlation x",
                     y="corr value",
                     color="correlation y",
                     barmode="group",
                     facet_row="dim")
        st.plotly_chart(fig)


@st.experimental_memo
def correlate_input_and_r_gen(inp: np.ndarray,
                              r_gen: np.ndarray,
                              time_delay: int = 0
                              ) -> np.ndarray:
    """Correlate the reservoir input with the driven r_gen states and add a time_delay.

    Correlate every r_gen dimension with every input dimension.

    Args:
        inp: The input time series used to drive the reservoir of shape (drive steps, sys_dim).
        r_gen: The r_gen states created during driving of shape (drive steps, r_gen_dim).
        time_delay: An optional time delay to apply r_gen before correlating. A positive time
                    delay correlates r_gen with past input values, a negative correlates r_gen
                    with future input values.

    Returns:
        The correlation matrix of shape (r_gen_dim, input_dim).
    """
    if time_delay == 0:
        r_gen_slice = r_gen
        inp = inp
    else:
        if time_delay > 0:
            r_gen_slice = r_gen[time_delay:, :]
            inp = inp[:-time_delay, :]
        elif time_delay < 0:
            r_gen_slice = r_gen[:time_delay, :]
            inp = inp[-time_delay:, :]
        else:
            raise ValueError

    r_gen_dim = r_gen_slice.shape[1]
    inp_dim = inp.shape[1]
    correlation = np.zeros((r_gen_dim, inp_dim))
    for i_r in range(r_gen_dim):
        for i_inp in range(inp_dim):
            correlation[i_r, i_inp] = \
                np.corrcoef(r_gen_slice[:, i_r], inp[:, i_inp])[0, 1]

    return correlation



@st.experimental_memo
def get_pca_transformed_quantities(r_gen_train: np.ndarray,
                                   r_gen_pred: np.ndarray,
                                   w_out: np.ndarray
                                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Perform a PCA transform on r_gen_states and transform r_gen_train, pred and w_out.

    Args:
        r_gen_train: The generalized train reservoir states of shape (train steps, r_gen_dim).
        r_gen_pred: The generalized pred reservoir states of shape (pred steps, r_gen_dim).
        w_out: The output matrix of shape (out_dim, r_gen_dim).

    Returns:
        A tuple with the transformed r_gen_train_pca, r_gen_pred_pca and w_out_pca.
    """
    pca = PCA()
    r_gen_train_pca = pca.fit_transform(r_gen_train)
    r_gen_pred_pca = pca.transform(r_gen_pred)
    p = pca.components_
    w_out_pca = w_out @ p.T

    return r_gen_train_pca, r_gen_pred_pca, w_out_pca
