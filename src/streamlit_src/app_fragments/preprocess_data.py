"""Python file that includes (streamlit) functions to preprocess the raw data. """

from __future__ import annotations

import streamlit as st
import numpy as np

from src.streamlit_src.app_fragments import streamlit_utilities as utils

import src.esn_src.data_preprocessing as datapre


def st_get_scale_shift_params(key: str | None = None) -> tuple[float, float] | None:
    """Streamlit element to get scale and shift parameters for preprocessing.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        Either None or two floats The first being the scale and the second being the shift along
        all dimensions.
    """
    with st.expander("Scale and shift: "):
        if st.checkbox("Scale and shift",
                       key=f"{key}__st_preprocess_simulation__normcenter_check",
                       help="Use this to scale and center the data. "):
            left, right = st.columns(2)
            with left:
                scale = st.number_input("scale", value=1.0, min_value=0.0, step=0.1, format="%f",
                                        key=f"{key}__st_preprocess_simulation__scale")
            with right:
                shift = st.number_input("shift", value=0.0, step=0.1, format="%f",
                                        key=f"{key}__st_preprocess_simulation__shift")
            scale_shift_params = scale, shift
        else:
            scale_shift_params = None
        return scale_shift_params


@st.experimental_memo(max_entries=utils.MAX_CACHE_ENTRIES)
def get_scaled_and_shifted_data(time_series: np.ndarray,
                                scale_shift_params: tuple[float, float],
                                ) -> np.ndarray | tuple[np.ndarray, tuple[np.darray, np.ndarray]]:
    """
    Scale and shift a time series.

    First center and normalize the time_series to a std of unity for each axis. Then optionally
    rescale and/or shift the time series.

    Args:
        time_series: The time series of shape (time_steps, sys_dim).
        scale_shift_params: A tuple with the first element being a float describing the std in
                            every direction of the modified time series. The second element being
                            the mean in every direction of the modified time series.

    Returns:
        The scaled and shifted time_series and, if return_scale_shift is True: A tuple containing
        the scale_vec and shift_vec.
    """
    scale, shift = scale_shift_params
    time_series, scale_shift_vector =  datapre.scale_and_shift(
        time_series,
        scale=scale,
        shift=shift,
        return_scale_shift=True)

    return time_series, scale_shift_vector


def st_get_noise_scale(key: str | None = None) -> float | None:
    """Streamlit element to get the noise scale used for preprocessing.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        A float representing the white noise scale.
    """
    with st.expander("Add white noise: "):
        if st.checkbox("Add white noise", key=f"{key}__st_preprocess_simulation__noise_check"):
            noise_scale = st.number_input("noise scale", value=0.1, min_value=0.0, step=0.01,
                                          format="%f",
                                          key=f"{key}__st_preprocess_simulation__noise")
        else:
            noise_scale = None
        return noise_scale


@st.experimental_memo(max_entries=utils.MAX_CACHE_ENTRIES)
def get_noisy_data(time_series: np.ndarray,
                   noise_scale: float = 0.1,
                   seed: int | None = None
                   ) -> np.ndarray:
    """Add gaussian noise to a time_series.

    Args:
        time_series: The input time series of shape (time_steps, sys_dim).
        noise_scale: The scale of the gaussian white noise.
        seed: The seed used to calculate the noise.

    Returns:
        The time series with added noise.
    """
    return datapre.add_noise(time_series, noise_scale=noise_scale, seed=seed)


# def st_preprocess_simulation(key: str | None = None
#                              ) -> tuple[tuple[float, float] | None, float | None]:
#     """Streamlit elements to get parameters for preprocessing the data.
#
#     To be used together with preprocess_simulation.
#
#     One can add scale and center the data and add white noise.
#     Args:
#         key: Provide a unique key if this streamlit element is used multiple times.
#
#     Returns:
#         The scale_shift_parameters and noise_scale to be input into preprocess_simulation.
#     """
#     with st.expander("Preprocess:"):
#         if st.checkbox("Normalize and center",
#                        key=f"{key}__st_preprocess_simulation__normcenter_check"):
#             left, right = st.columns(2)
#             with left:
#                 scale = st.number_input("scale", value=1.0, min_value=0.0, step=0.1, format="%f",
#                                         key=f"{key}__st_preprocess_simulation__scale")
#             with right:
#                 shift = st.number_input("shift", value=0.0, step=0.1, format="%f",
#                                         key=f"{key}__st_preprocess_simulation__shift")
#             scale_shift_params = scale, shift
#         else:
#             scale_shift_params = None
#
#         if st.checkbox("Add white noise", key=f"{key}__st_preprocess_simulation__noise_check"):
#             noise_scale = st.number_input("noise scale", value=0.1, min_value=0.0, step=0.01,
#                                           format="%f",
#                                           key=f"{key}__st_preprocess_simulation__noise")
#         else:
#             noise_scale = None
#
#     return scale_shift_params, noise_scale



# @st.experimental_memo(max_entries=utils.MAX_CACHE_ENTRIES)
# def preprocess_simulation(time_series: np.ndarray,
#                           seed: int,
#                           scale_shift_params: tuple[float, float] | None,
#                           noise_scale: float | None = None
#                           ) -> np.ndarray | tuple[np.ndarray, tuple[np.darray, np.ndarray]]:
#     """Function to preprocess the data: scale shift and add noise.
#
#     Args:
#         time_series: The input timeseries.
#         seed: The seed to use for the random noise.
#         scale_shift_params: A tuple with the first element being a float describing the std in
#                             every direction of the modified time series. The second element being
#                             the mean in every direction of the modified time series.
#                             If None don't scale and shift.
#         noise_scale: The scale of the added white noise.
#
#     Returns:
#         The modified timeseries.
#     """
#
#     mod_time_series = time_series
#     if scale_shift_params is not None:
#         scale, shift = scale_shift_params
#         mod_time_series, scale_shift_vector = get_scaled_and_shifted_data(time_series,
#                                                                           shift=shift,
#                                                                           scale=scale,
#                                                                           return_scale_shift=True)
#
#     if noise_scale is not None:
#         mod_time_series = get_noisy_data(mod_time_series,
#                                          noise_scale=noise_scale,
#                                          seed=seed)
#
#     if scale_shift_params is not None:  # If you scale and shift, also return the vectors.
#         return mod_time_series, scale_shift_vector
#     else:
#         return mod_time_series, None


@st.experimental_memo(max_entries=utils.MAX_CACHE_ENTRIES)
def inverse_transform_shift_scale(time_series: np.ndarray,
                                  scale_shift_vectors: tuple[np.ndarray, np.ndarray]
                                  ) -> np.ndarray:
    """Inverse transform a time series that was shifted and scaled.
    # TODO: not sure if needed.

    The inverse to get_scaled_and_shifted_data.

    Args:
        time_series: The shifted and scaled input timeseries.
        scale_shift_params: A tuple: (scale_vector, shift_vector).

    Returns:
        The inverse_transformed times series.
    """
    scale_vec, shift_vec = scale_shift_vectors
    return (time_series - shift_vec) / scale_vec


def st_embed_timeseries(x_dim: int, key: str | None = None) -> tuple[int, int, list[int]] | None:
    """Streamlit element to specify the embedding settings.

    To be used with get_embedded_time_series.

    Args:
        x_dim: The dimension of the time series to be embedded.
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        A tuple where the fist element is the embedding dimension (int), the second is the
        time delay (int), and the third is a list of the selected dimensions.
    """

    with st.expander("Embedding:"):

        embedding_bool = st.checkbox("Do embedding", key=f"{key}__st_embed_timeseries__embbool")

        if embedding_bool:
            cols = st.columns(2)
            with cols[0]:
                embedding_dim = int(st.number_input("Embed. dim.", value=0, min_value=0,
                                                    key=f"{key}__st_embed_timeseries__embdim"))
            with cols[1]:
                delay = int(st.number_input("Delay", value=1, min_value=1,
                                            key=f"{key}__st_embed_timeseries__delay"))
            dimension_selection = utils.st_dimension_selection_multiple(
                x_dim,
                default_select_all_bool=True,
                key=f"{key}__st_embed_timeseries")
            return embedding_dim, delay, dimension_selection
        else:
            return None  # Default values for no embedding.



@st.experimental_memo
def get_embedded_time_series(time_series: np.ndarray,
                             embedding_dimension: int,
                             delay: int,
                             dimension_selection: list[int] | None) -> np.ndarray:
    """Embed the time series.

    Args:
        time_series: The input time series of shape (timesteps, x_dim).
        embedding_dimension: The number of embedding dimensions to add.
        delay: The time delay to use.
        dimension_selection: A list of ints representing the index of the dimensions to consider.
                             If None: Take all dimensions.

    Returns:
        The embedded time series of shape (timesteps - delay, embedding_dimension * len(dimension_selection)).
    """

    return datapre.embedding(time_series,
                             embedding_dimension=embedding_dimension,
                             delay=delay,
                             dimension_selection=dimension_selection)


def st_pca_transform_time_series(x_dim: int,
                                 key: str | None = None) -> int | None:
    """Streamlit element to specify whether to perform the pca transformation or not.

    To be used with "get_pca_transformed_time_series".

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        A bool whether to perform the pca transform or not.
    """
    with st.expander("PCA transform:"):
        if st.checkbox("Do pca transform", key=f"{key}__st_pca_transform_time_series"):
            components = st.number_input("Components",
                                         value=x_dim,
                                         min_value=1,
                                         max_value=x_dim,
                                         key=f"{key}__st_pca_transform_time_series__pcs")
            return int(components)
        else:
            return None


@st.experimental_memo
def get_pca_transformed_time_series(time_series: np.ndarray,
                                    components: int) -> np.ndarray:
    """Perform a pca transform the time_series.

    Args:
        time_series: The input time series of shape (timesteps, x_dim).
        components: The number of principle components. If None: use all.

    Returns:
        The pca transformed time series of shape (timesteps, x_dim).
    """
    return datapre.pca_transform(time_series, components=components)


def st_all_preprocess(time_series: np.ndarray,
                               noise_seed: int | None = None,
                               key: str | None = None):
    """Streamlit element for all data preprocessing.

    Args:
        time_series: The input time series of shape (timesteps, x_dim).
        noise_seed: The random seed used for the noise.
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        ?
    """
    preprocess_status = []

    # PCA
    x_dim = time_series.shape[1]
    pca_out = st_pca_transform_time_series(x_dim=x_dim, key=key)
    if pca_out is not None:
        components = pca_out
        time_series = get_pca_transformed_time_series(time_series, components)
        preprocess_status.append("PCA")

    # Embedding
    x_dim = time_series.shape[1]
    embedding_out = st_embed_timeseries(x_dim, key=key)
    if embedding_out is not None:
        embedding_dims, embedding_delay, embedding_dim_selection = embedding_out
        time_series = get_embedded_time_series(
            time_series,
            embedding_dimension=embedding_dims,
            delay=embedding_delay,
            dimension_selection=embedding_dim_selection)
        preprocess_status.append("Embedding")

    # Scale and Shift:
    scale_shift_out = st_get_scale_shift_params(key=key)
    if scale_shift_out is not None:
        scale_shift_params = scale_shift_out
        time_series, scale_shift_vector = get_scaled_and_shifted_data(time_series,
                                                                      scale_shift_params)
        preprocess_status.append("Scale&Shift")

    # Add noise:
    noise_out = st_get_noise_scale(key=key)
    if noise_out is not None:
        noise_scale = noise_out
        time_series = get_noisy_data(time_series,
                                     noise_scale=noise_scale,
                                     seed=noise_seed)
        preprocess_status.append("Noise")

    if len(preprocess_status) != 0:
        st.markdown("**Used:** " + (", ").join(preprocess_status))
    else:
        st.markdown("**Used:** -")
    st.markdown(f"**Preprocessed data shape:** {time_series.shape}")

    return time_series
