"""Python file that includes streamlit elements that are used to build/train/predict with an esn."""

from __future__ import annotations

import copy
import inspect
from typing import Any

import numpy as np
import streamlit as st

from src.esn_src import esn, utilities
from src.streamlit_src.app_fragments import streamlit_utilities as utils

def esn_hash(obj):
    items = sorted(obj.__dict__.items(), key=lambda it: it[0])
    return hash((type(obj),) + tuple(items))


ESN_DICT = {"ESN_normal": esn.ESN_normal,
            # "ESN_pca": esn.ESN_pca,
            }

ESN_HASH_FUNC = {esn._ResCompCore: esn_hash}

W_IN_TYPES = ["random_sparse", "ordered_sparse", "random_dense_uniform", "random_dense_gaussian"]
BIAS_TYPES = ["no_bias", "random_bias", "constant_bias"]
NETWORK_TYPES = ["erdos_renyi", "scale_free", "small_world", "random_directed",
                 "random_dense_gaussian",
                 "scipy_sparse"]
ACTIVATION_FUNCTIONS = ["tanh", "sigmoid", "relu", "linear"]
R_TO_R_GEN_TYPES = ["linear_r", "linear_and_square_r", "output_bias", "bias_and_square_r",
                    "linear_and_square_r_alt",
                    "exponential_r", "bias_and_exponential_r"]

ESN_TYPING = Any


def st_select_split_up_relative(total_steps: int,
                                default_t_train_disc_rel: int = 1000,
                                default_t_train_sync_rel: int = 300,
                                default_t_train_rel: int = 2000,
                                default_t_pred_disc_rel: int = 1000,
                                default_t_pred_sync_rel: int = 300,
                                default_t_pred_rel: int = 5000,
                                key: str | None = None,
                                ) -> tuple[int, int, int, int, int, int] | None:
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
        The selected time steps or None if too many were selected.
    """

    total_relative = default_t_train_disc_rel + default_t_train_sync_rel + default_t_train_rel + \
                     default_t_pred_disc_rel + default_t_pred_sync_rel + default_t_pred_rel

    t_disc_rel = default_t_train_disc_rel / total_relative
    t_sync_rel = default_t_train_sync_rel / total_relative
    t_rel = default_t_train_rel / total_relative
    p_disc_rel = default_t_pred_disc_rel / total_relative
    p_sync_rel = default_t_pred_sync_rel / total_relative
    p_rel = default_t_pred_rel / total_relative

    with st.expander("Train-Predict split: "):
        default_t_train_disc = int(t_disc_rel * total_steps)
        t_train_disc = st.number_input('t_train_disc',
                                       value=default_t_train_disc,
                                       step=1,
                                       key=f"{key}__st_select_split_up_relative__td")
        default_t_train_sync = int(t_sync_rel * total_steps)
        t_train_sync = st.number_input('t_train_sync',
                                       value=default_t_train_sync,
                                       step=1,
                                       key=f"{key}__st_select_split_up_relative__ts")
        default_t_train = int(t_rel * total_steps)
        t_train = st.number_input('t_train',
                                  value=default_t_train,
                                  step=1,
                                  key=f"{key}__st_select_split_up_relative__t")
        default_t_pred_disc = int(p_disc_rel * total_steps)
        t_pred_disc = st.number_input('t_pred_disc',
                                      value=default_t_pred_disc,
                                      step=1,
                                      key=f"{key}__st_select_split_up_relative__pd")
        default_t_pred_sync = int(p_sync_rel * total_steps)
        t_pred_sync = st.number_input('t_pred_sync',
                                      value=default_t_pred_sync,
                                      step=1,
                                      key=f"{key}__st_select_split_up_relative__ps")
        default_t_pred = int(p_rel * total_steps)
        t_pred = st.number_input('t_pred',
                                 value=default_t_pred,
                                 step=1,
                                 key=f"{key}__st_select_split_up_relative__p")

        sum = t_train_disc + t_train_sync + t_train + t_pred_disc + t_pred_sync + t_pred
        st.write(f"Time steps not used: {total_steps - sum}")
        if sum > total_steps:
            st.error("More timesteps selected than available in processed data. ", icon="🚨")
            return None
        else:
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


@st.cache(hash_funcs=ESN_HASH_FUNC, allow_output_mutation=False,
          max_entries=utils.MAX_CACHE_ENTRIES)
def build(esn_type: str, seed: int, x_dim: int, build_args: dict[str, Any]) -> ESN_TYPING:
    """Build the esn class.

    Args:
        esn_type: One of the esn types defined in ESN_DICT.
        seed: Set the global seed. TODO: maybe dont set global seed?
        x_dim: The x_dimension of the data to be predicted.
        build_args: The build args parsed to esn_obj.build.

    Returns:
        The built esn.
    """
    if esn_type in ESN_DICT.keys():
        esn = ESN_DICT[esn_type]()
    else:
        raise Exception("This esn_type is not accounted for")

    build_args = copy.deepcopy(build_args)

    seed_args = _get_seed_args_in_build(esn)
    rng = np.random.default_rng(seed)
    seeds = rng.integers(0, 1000000, len(seed_args))
    for i_seed, seed_arg in enumerate(seed_args):
        build_args[seed_arg] = seeds[i_seed]

    build_kwargs = utilities._remove_invalid_args(esn.build, build_args)

    esn.build(x_dim, **build_kwargs)
    return esn


def _get_seed_args_in_build(esn_obj: ESN_TYPING) -> list[str, ...]:
    """Utility function to get all the seed kwargs in the esn.build function.

    Args:
        esn_obj: The esn object with the build method.

    Returns:
        List of keyword argument names of build, that have "seed" in it.
    """
    build_func = esn_obj.build
    args = inspect.signature(build_func).parameters
    return [arg_name for arg_name in args.keys() if "seed" in arg_name]


def st_select_esn_type(esn_sub_section: tuple[str, ...] | None = None,
                       key: str | None = None) -> str:
    """Streamlit elements to specify the esn type.

    Args:
        esn_sub_section: A subsection of the keys in ESN_DICT, or if None, take all of ESN_DICT.
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        The esn_type as a string.
    """
    if esn_sub_section is None:
        esn_dict = ESN_DICT
    else:
        esn_dict = {esn_name: esn_class for esn_name, esn_class in ESN_DICT.items()
                    if esn_name in esn_sub_section}
        if len(esn_dict) == 0:  # TODO: proper error
            raise Exception(f"The systems in {esn_sub_section} are not accounted for.")

    esn_type = st.selectbox('esn type', esn_dict.keys(), key=f"{key}__st_select_esn_type")
    return esn_type


def st_basic_esn_build(key: str | None = None) -> dict[str, Any]:
    """Streamlit elements to specify the basic esn settings.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        The basic esn build_args as a dictionary.
    """

    basic_build_args = {}
    basic_build_args["r_dim"] = int(st.number_input('Reservoir Dim', value=500, step=1,
                                                    key=f"{key}__st_basic_esn_build__rd"))
    basic_build_args["r_to_r_gen_opt"] = st.selectbox('r_to_r_gen_opt', R_TO_R_GEN_TYPES,
                                                      key=f"{key}__st_basic_esn_build__rrgen")
    basic_build_args["act_fct_opt"] = st.selectbox('act_fct_opt', ACTIVATION_FUNCTIONS,
                                                   key=f"{key}__st_basic_esn_build__actfct")
    basic_build_args["node_bias_opt"] = st.selectbox('node_bias_opt', BIAS_TYPES,
                                                     key=f"{key}__st_basic_esn_build__nbo")
    disabled = True if basic_build_args["node_bias_opt"] == "no_bias" else False
    basic_build_args["bias_scale"] = st.number_input('bias_scale', value=0.1, step=0.1,
                                                     disabled=disabled,
                                                     key=f"{key}__st_basic_esn_build__bs")
    basic_build_args["leak_factor"] = st.number_input('leak_factor', value=0.0, step=0.01,
                                                      min_value=0.0, max_value=1.0,
                                                      key=f"{key}__st_basic_esn_build__lf")
    basic_build_args["w_in_opt"] = st.selectbox('w_in_opt', W_IN_TYPES,
                                                key=f"{key}__st_basic_esn_build__winopt")
    basic_build_args["w_in_scale"] = st.number_input('w_in_scale', value=1.0, step=0.1,
                                                     key=f"{key}__st_basic_esn_build__winsc")
    log_reg_param = st.number_input('Log regulation parameter', value=-7., step=1., format="%f",
                                    key=f"{key}__st_basic_esn_build__reg")
    basic_build_args["reg_param"] = 10 ** (log_reg_param)

    return basic_build_args


def st_network_build_args(key: str | None = None) -> dict[str, object]:
    """Streamlit elements to specify the network settings.

    Args:
        key: Provide a unique key if this streamlit element is used multiple times.

    Returns:
        A dictionary containing the network build args.
    """
    network_build_args = {}
    network_build_args["n_rad"] = st.number_input('n_rad', value=0.1, step=0.1, format="%f",
                                                  key=f"{key}__st_network_build_args__nrad")
    network_build_args["n_avg_deg"] = st.number_input('n_avg_deg', value=5.0, step=0.1,
                                                      key=f"{key}__st_network_build_args__ndeg")
    network_build_args["n_type_opt"] = st.selectbox('n_type_opt', NETWORK_TYPES,
                                                    key=f"{key}__st_network_build_args__nopt")
    return network_build_args


@st.cache(hash_funcs=ESN_HASH_FUNC, allow_output_mutation=False,
          max_entries=utils.MAX_CACHE_ENTRIES)
def train_return_res(esn_obj: ESN_TYPING,
                     x_train: np.ndarray,
                     t_train_sync: int,
                     ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], ESN_TYPING]:
    """Train the esn_obj with a given x_train and t_train-sync and return internal reservoir states.

    TODO: check when to use this func and when to use train.

    Args:
        esn_obj: The esn_obj, that has a train method.
        x_train: The np.ndarray of shape (t_train_sync_steps + t_train_steps, sys_dim)
        t_train_sync: The number of time steps used for syncing the esn before training.

    Returns:
        Tuple with the fitted output, the real output and reservoir dictionary containing states
        for r_act_fct_inp, r_internal, r_input, r, r_gen, and the esn_obj.
    """
    x_train = x_train.copy()
    esn_obj.train(x_train,
                  sync_steps=t_train_sync,
                  save_y_train=True,
                  save_out=True,
                  save_res_inp=True,
                  save_r_internal=True,
                  save_r=True,
                  save_r_gen=True
                  )

    y_train_true = esn_obj.get_y_train()
    y_train_fit = esn_obj.get_out()

    res_state_dict = {}
    res_state_dict["r_act_fct_inp"] = esn_obj.get_act_fct_inp()
    res_state_dict["r_internal"] = esn_obj.get_r_internal()
    res_state_dict["r_input"] = esn_obj.get_res_inp()
    res_state_dict["r"] = esn_obj.get_r()
    res_state_dict["r_gen"] = esn_obj.get_r_gen()

    return y_train_fit, y_train_true, res_state_dict, esn_obj


@st.cache(hash_funcs=ESN_HASH_FUNC, allow_output_mutation=False,
          max_entries=utils.MAX_CACHE_ENTRIES)
def predict_return_res(esn_obj: ESN_TYPING, x_pred: np.ndarray, t_pred_sync: int
                       ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray], ESN_TYPING]:
    """Predict with the esn_obj with a given x_pred and x_pred_sync and return internal reservoir states.

    TODO: check when to use this func and when to use predict.

    Args:
        esn_obj: The esn_obj, that has a predict method.
        x_pred: The np.ndarray of shape (t_pred_sync_steps + t_pred_steps, sys_dim)
        t_pred_sync: The number of time steps used for syncing the esn before prediction.

    Returns:
        Tuple with the fitted output, the real output and reservoir dictionary containing states
        for r_act_fct_inp, r_internal, r_input, r, r_gen, and the esn_obj.
    """
    x_pred = x_pred.copy()
    y_pred, y_pred_true = esn_obj.predict(x_pred,
                                          sync_steps=t_pred_sync,
                                          save_res_inp=True,
                                          save_r_internal=True,
                                          save_r=True,
                                          save_r_gen=True
                                          )
    res_state_dict = {}
    res_state_dict["r_act_fct_inp"] = esn_obj.get_act_fct_inp()
    res_state_dict["r_internal"] = esn_obj.get_r_internal()
    res_state_dict["r_input"] = esn_obj.get_res_inp()
    res_state_dict["r"] = esn_obj.get_r()
    res_state_dict["r_gen"] = esn_obj.get_r_gen()

    return y_pred, y_pred_true, res_state_dict, esn_obj
