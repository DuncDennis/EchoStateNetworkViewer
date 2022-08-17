"""Python file that includes streamlit elements that are used to build/train/predict with an esn."""

from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st

from src.esn_src import esn, utilities

def esn_hash(obj):
    items = sorted(obj.__dict__.items(), key=lambda it: it[0])
    return hash((type(obj),) + tuple(items))


ESN_DICT = {"ESN_normal": esn.ESN_normal,
            "ESN_pca": esn.ESN_pca,
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


@st.cache(hash_funcs=ESN_HASH_FUNC, allow_output_mutation=True)
def build(esn_type: str, seed: int, x_dim: int, **kwargs) -> ESN_TYPING:
    """Build the esn class.

    Args:
        esn_type: One of the esn types defined in ESN_DICT.
        seed: Set the global seed. TODO: maybe dont set global seed?
        x_dim: The x_dimension of the data to be predicted.
        **kwargs: All other build args.

    Returns:
        The build esn.
    """
    print("Running build")
    if esn_type in ESN_DICT.keys():
        esn = ESN_DICT[esn_type]()
    else:
        raise Exception("This esn_type is not accounted for")

    np.random.seed(seed)

    build_kwargs = utilities._remove_invalid_args(esn.build, kwargs)
    esn.build(x_dim, **build_kwargs)
    return esn


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


def st_basic_esn_build(key: str | None = None) -> dict[str, object]:
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


@st.cache(hash_funcs=ESN_HASH_FUNC)
def train(esn_obj: ESN_TYPING, x_train: np.ndarray, t_train_sync: int,
          ) -> tuple[np.ndarray, np.ndarray]:
    """Train the esn_obj with a given x_train and t_train-sync.

    Args:
        esn_obj: The esn_obj, that has a train method.
        x_train: The np.ndarray of shape (t_train_sync_steps + t_train_steps, sys_dim)
        t_train_sync: The number of time steps used for syncing the esn before training.

    Returns:
        Tuple with the fitted output and the real output.
    """

    esn_obj.train(x_train, sync_steps=t_train_sync,
                  save_y_train=True, save_out=True)

    y_train_true = esn_obj.get_y_train()
    y_train_fit = esn_obj.get_out()

    return y_train_fit, y_train_true


@st.cache(hash_funcs=ESN_HASH_FUNC)
def train_return_res(esn_obj: ESN_TYPING, x_train: np.ndarray, t_train_sync: int,
                     ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Train the esn_obj with a given x_train and t_train-sync and return internal reservoir states.

    TODO: check when to use this func and when to use train.

    Args:
        esn_obj: The esn_obj, that has a train method.
        x_train: The np.ndarray of shape (t_train_sync_steps + t_train_steps, sys_dim)
        t_train_sync: The number of time steps used for syncing the esn before training.

    Returns:
        Tuple with the fitted output, the real output and reservoir dictionary containing states
        for r_act_fct_inp, r_internal, r_input, r, r_gen.
    """
    print("Running train")

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

    return y_train_fit, y_train_true, res_state_dict


@st.cache(hash_funcs=ESN_HASH_FUNC)
def predict(esn_obj: ESN_TYPING, x_pred: np.ndarray, t_pred_sync: int
            ) -> tuple[np.ndarray, np.ndarray]:
    """Predict with the esn_obj with a given x_pred and x_pred_sync.

    Args:
        esn_obj: The esn_obj, that has a predict method.
        x_pred: The np.ndarray of shape (t_pred_sync_steps + t_pred_steps, sys_dim)
        t_pred_sync: The number of time steps used for syncing the esn before prediction.

    Returns:
        Tuple with predicted and true data, each of shape (t_pred_steps, sys_dim)
    """

    y_pred, y_pred_true = esn_obj.predict(x_pred, sync_steps=t_pred_sync)

    return y_pred, y_pred_true


@st.cache(hash_funcs=ESN_HASH_FUNC)
def predict_return_res(esn_obj: ESN_TYPING, x_pred: np.ndarray, t_pred_sync: int
                       ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Predict with the esn_obj with a given x_pred and x_pred_sync and return internal reservoir states.

    TODO: check when to use this func and when to use predict.

    Args:
        esn_obj: The esn_obj, that has a predict method.
        x_pred: The np.ndarray of shape (t_pred_sync_steps + t_pred_steps, sys_dim)
        t_pred_sync: The number of time steps used for syncing the esn before prediction.

    Returns:
        Tuple with the fitted output, the real output and reservoir dictionary containing states
        for r_act_fct_inp, r_internal, r_input, r, r_gen.
    """

    print("Running predict")

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

    return y_pred, y_pred_true, res_state_dict
