"""Python file that includes (streamlit) functions to preprocess the raw data. """

from __future__ import annotations

import streamlit as st
import numpy as np

from src.streamlit_src.app_fragments import streamlit_utilities as utils

import src.esn_src.data_preprocessing as datapre


def st_select_time_steps_split_up(default_t_train_disc: int = 1000,
                                  default_t_train_sync: int = 300,
                                  default_t_train: int = 2000,
                                  default_t_pred_disc: int = 1000,
                                  default_t_pred_sync: int = 300,
                                  default_t_pred: int = 5000,
                                  key: str | None = None,
                                  ) -> tuple[int, int, int, int, int, int]:
    """Streamlit elements train discard, train sync, train, pred discard, pred sync and pred.

    Args:
        default_t_train_disc: Default train disc time steps.
        default_t_train_sync: Default train sync time steps.
        default_t_train: Defaut train time steps.
        default_t_pred_disc: Default predict disc time steps.
        default_t_pred_sync: Default predict sync time steps.
        default_t_pred: Default predict time steps.
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        The selected time steps.
    """
    with st.expander("Time steps: "):
        t_train_disc = st.number_input('t_train_disc', value=default_t_train_disc, step=1,
                                       key=f"{key}__st_select_time_steps_split_up__td")
        t_train_sync = st.number_input('t_train_sync', value=default_t_train_sync, step=1,
                                       key=f"{key}__st_select_time_steps_split_up__ts")
        t_train = st.number_input('t_train', value=default_t_train, step=1,
                                  key=f"{key}__st_select_time_steps_split_up__t")
        t_pred_disc = st.number_input('t_pred_disc', value=default_t_pred_disc, step=1,
                                      key=f"{key}__st_select_time_steps_split_up__pd")
        t_pred_sync = st.number_input('t_pred_sync', value=default_t_pred_sync, step=1,
                                      key=f"{key}__st_select_time_steps_split_up__ps")
        t_pred = st.number_input('t_pred', value=default_t_pred, step=1,
                                 key=f"{key}__st_select_time_steps_split_up__p")

        return int(t_train_disc), int(t_train_sync), int(t_train), int(t_pred_disc), \
               int(t_pred_sync), int(t_pred)


def st_select_split_up_relative(total_steps: int,
                                default_t_train_disc_rel: int = 1000,
                                default_t_train_sync_rel: int = 300,
                                default_t_train_rel: int = 2000,
                                default_t_pred_disc_rel: int = 1000,
                                default_t_pred_sync_rel: int = 300,
                                default_t_pred_rel: int = 5000,
                                key: str | None = None,
                                ) -> tuple[int, int, int, int, int, int]:
    """Streamlit elements train discard, train sync, train, pred discard, pred sync and pred.

    Args:
        default_t_train_disc_rel: Default train disc time steps.
        default_t_train_sync_rel: Default train sync time steps.
        default_t_train_rel: Defaut train time steps.
        default_t_pred_disc: Default predict disc time steps.
        default_t_pred_sync: Default predict sync time steps.
        default_t_pred: Default predict time steps.
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        The selected time steps.
    """

    total_relative = default_t_train_disc_rel + default_t_train_sync_rel + default_t_train_rel + \
                     default_t_pred_disc_rel + default_t_pred_sync_rel + default_t_pred_rel

    t_disc_rel = default_t_train_disc_rel / total_relative
    t_sync_rel = default_t_train_sync_rel / total_relative
    t_rel = default_t_train_rel / total_relative
    p_disc_rel = default_t_pred_disc_rel / total_relative
    p_sync_rel = default_t_pred_sync_rel / total_relative
    p_rel = default_t_pred_rel / total_relative

    with st.expander("Time steps: "):
        default_t_train_disc = int(t_disc_rel * total_steps)
        t_train_disc = st.number_input('t_train_disc', value=default_t_train_disc, step=1,
                                       key=f"{key}__st_select_split_up_relative__td")
        default_t_train_sync = int(t_sync_rel * total_steps)
        t_train_sync = st.number_input('t_train_sync', value=default_t_train_sync, step=1,
                                       key=f"{key}__st_select_split_up_relative__ts")
        default_t_train = int(t_rel * total_steps)
        t_train = st.number_input('t_train', value=default_t_train, step=1,
                                  key=f"{key}__st_select_split_up_relative__t")
        default_t_pred_disc = int(p_disc_rel * total_steps)
        t_pred_disc = st.number_input('t_pred_disc', value=default_t_pred_disc, step=1,
                                      key=f"{key}__st_select_split_up_relative__pd")
        default_t_pred_sync = int(p_sync_rel * total_steps)
        t_pred_sync = st.number_input('t_pred_sync', value=default_t_pred_sync, step=1,
                                      key=f"{key}__st_select_split_up_relative__ps")
        default_t_pred = int(p_rel * total_steps)
        t_pred = st.number_input('t_pred', value=default_t_pred, step=1,
                                 key=f"{key}__st_select_split_up_relative__p")

        sum = t_train_disc + t_train_sync + t_train + t_pred_disc + t_pred_sync + t_pred
        st.write(f"Time steps not used: {total_steps - sum}")

        return t_train_disc, t_train_sync, t_train, t_pred_disc, t_pred_sync, t_pred


def split_time_series_for_train_pred(time_series: np.ndarray,
                                     t_train_disc: int,
                                     t_train_sync: int,
                                     t_train: int,
                                     t_pred_disc: int,
                                     t_pred_sync: int,
                                     t_pred: int) -> tuple[np.ndarray, np.ndarray]:
    """Split the time_series for training and prediction of an esn.

    Remove t_train_disc from time_series and use t_train_sync and t_train for x_train.
    Then remove t_pred_disc from the remainder and use the following t_pred_sync and t_pred
    steps for x_pred.

    Args:
        time_series: The input timeseries of shape (time_steps, sys_dim).
        t_train_disc: The time steps to skip before x_train.
        t_train_sync: The time steps used for synchro before training.
        t_train: The time steps used for training.
        t_pred_disc: The time steps to skip before prediction.
        t_pred_sync: The time steps to use for synchro before training.
        t_pred: The time steps used for prediction.

    Returns:
        A tuple containing x_train and x_pred.
    """
    x_train = time_series[t_train_disc: t_train_disc + t_train_sync + t_train]
    start = t_train_disc + t_train_sync + t_train + t_pred_disc
    x_pred = time_series[start: start + t_pred_sync + t_pred]

    return x_train, x_pred


@st.experimental_memo(max_entries=utils.MAX_CACHE_ENTRIES)
def get_scaled_and_shifted_data(time_series: np.ndarray,
                                scale: float = 1.0,
                                shift: float = 0.0,
                                return_scale_shift: bool = False
                                ) -> np.ndarray | tuple[np.ndarray, tuple[np.darray, np.ndarray]]:
    """
    Scale and shift a time series.

    First center and normalize the time_series to a std of unity for each axis. Then optionally
    rescale and/or shift the time series.

    Args:
        time_series: The time series of shape (time_steps, sys_dim).
        scale: Scale every axis so that the std is the scale value.
        shift: Shift every axis so that the mean is the shift value.
        return_scale_shift: If True, also return the scale_vec and shift_vec.

    Returns:
        The scaled and shifted time_series and, if return_scale_shift is True: A tuple containing
        the scale_vec and shift_vec.
    """
    return datapre.scale_and_shift(time_series,
                                   scale=scale,
                                   shift=shift,
                                   return_scale_shift=return_scale_shift)


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


def st_preprocess_simulation(key: str | None = None
                             ) -> tuple[tuple[float, float] | None, float | None]:
    """Streamlit elements to get parameters for preprocessing the data.

    To be used together with preprocess_simulation.

    One can add scale and center the data and add white noise.
    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        The scale_shift_parameters and noise_scale to be input into preprocess_simulation.
    """
    with st.expander("Preprocess:"):
        if st.checkbox("Normalize and center",
                       key=f"{key}__st_preprocess_simulation__normcenter_check"):
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

        if st.checkbox("Add white noise", key=f"{key}__st_preprocess_simulation__noise_check"):
            noise_scale = st.number_input("noise scale", value=0.1, min_value=0.0, step=0.01,
                                          format="%f",
                                          key=f"{key}__st_preprocess_simulation__noise")
        else:
            noise_scale = None

    return scale_shift_params, noise_scale


@st.experimental_memo(max_entries=utils.MAX_CACHE_ENTRIES)
def preprocess_simulation(time_series: np.ndarray,
                          seed: int,
                          scale_shift_params: tuple[float, float] | None,
                          noise_scale: float | None = None
                          ) -> np.ndarray | tuple[np.ndarray, tuple[np.darray, np.ndarray]]:
    """Function to preprocess the data: scale shift and add noise.

    Args:
        time_series: The input timeseries.
        seed: The seed to use for the random noise.
        scale_shift_params: A tuple with the first element being a float describing the std in
                            every direction of the modified time series. The second element being
                            the mean in every direction of the modified time series.
                            If None don't scale and shift.
        noise_scale: The scale of the added white noise.

    Returns:
        The modified timeseries.
    """

    mod_time_series = time_series
    if scale_shift_params is not None:
        scale, shift = scale_shift_params
        mod_time_series, scale_shift_vector = get_scaled_and_shifted_data(time_series,
                                                                          shift=shift,
                                                                          scale=scale,
                                                                          return_scale_shift=True)

    if noise_scale is not None:
        mod_time_series = get_noisy_data(mod_time_series,
                                         noise_scale=noise_scale,
                                         seed=seed)

    if scale_shift_params is not None:  # If you scale and shift, also return the vectors.
        return mod_time_series, scale_shift_vector
    else:
        return mod_time_series, None


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


def st_embed_timeseries(x_dim: int, key: str | None = None) -> tuple[int, int, list[int] | None]:
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
            return 0, 1, None  # Default values for no embedding.



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


def st_pca_transform_time_series(key: str | None = None) -> bool:
    """Streamlit element to specify whether to perform the pca transformation or not.

    To be used with "get_pca_transformed_time_series".

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        A bool whether to perform the pca transform or not.
    """
    with st.expander("PCA transform:"):
        return st.checkbox("Do pca transform", key=f"{key}__st_pca_transform_time_series")


@st.experimental_memo
def get_pca_transformed_time_series(time_series: np.ndarray) -> np.ndarray:
    """Perform a pca transform the time_series.

    Args:
        time_series: The input time series of shape (timesteps, x_dim).

    Returns:
        The pca transformed time series of shape (timesteps, x_dim).
    """
    return datapre.pca_transform(time_series)
