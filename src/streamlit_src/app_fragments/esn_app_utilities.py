from __future__ import annotations

import streamlit as st


def st_main_checkboxes(key: str | None = None) -> tuple[bool, bool, bool, bool, bool]:
    """Streamlit element to create 5 esn checkbs: Raw data, Prepr data, Build, Train and Predict.

    Args:
        key: A optional key if it's used multiple times.

    Returns:
        The states of the 5 checkboxes: raw_data_bool, preproc_data_bool, build_bool,
        train_bool, predict_bool.
    """

    basic_key = f"{key}__st_main_checkboxes"

    raw_data_checkbox_label = "📼 Get raw data"
    preproc_checkbox_label = "🌀 Preprocess data"
    build_checkbox_label = "🛠️ Build"
    train_checkbox_label = "🦾 Train"
    predict_checkbox_label = "🔮 Predict"

    def on_change_raw():
        if not st.session_state[f"{basic_key}__raw"]:
            other_keys = [f"{basic_key}__{x}" for x in ("preproc", "build", "train", "pred")]
            for k in other_keys:
                st.session_state[k] = False

    def on_change_preproc():
        if not st.session_state[f"{basic_key}__preproc"]:
            other_keys = [f"{basic_key}__{x}" for x in ("build", "train", "pred")]
            for k in other_keys:
                st.session_state[k] = False

    def on_change_build():
        if not st.session_state[f"{basic_key}__build"]:
            other_keys = [f"{basic_key}__{x}" for x in ("train", "pred")]
            for k in other_keys:
                st.session_state[k] = False

    def on_change_train():
        if not st.session_state[f"{basic_key}__train"]:
            other_keys = [f"{basic_key}__{x}" for x in ("pred", )]
            for k in other_keys:
                st.session_state[k] = False

    raw_data_bool = st.checkbox(raw_data_checkbox_label,
                                key=f"{basic_key}__raw",
                                on_change=on_change_raw)

    disabled = False if raw_data_bool else True
    preproc_data_bool = st.checkbox(preproc_checkbox_label,
                                    disabled=disabled,
                                    key=f"{basic_key}__preproc",
                                    on_change=on_change_preproc)

    disabled = False if preproc_data_bool else True
    build_bool = st.checkbox(build_checkbox_label,
                             disabled=disabled,
                             key=f"{basic_key}__build",
                             on_change=on_change_build)

    disabled = False if build_bool else True
    train_bool = st.checkbox(train_checkbox_label,
                             disabled=disabled,
                             key=f"{basic_key}__train",
                             on_change=on_change_train)

    disabled = False if train_bool else True
    predict_bool = st.checkbox(predict_checkbox_label,
                               disabled=disabled,
                               key=f"{basic_key}__pred")

    return raw_data_bool, preproc_data_bool, build_bool, train_bool, predict_bool


def uncheck_all_main_checkboxes(label_to_search: str = "st_main_checkboxes") -> None:
    """Utility function to uncheck all main checkboxes created by st_main_checkboxes.

    Has to be run before the checkboxes are initialized.

    Args:
        label_to_search: A string to be searched in the st.session_state keys.

    """
    for key in st.session_state.keys():
        if label_to_search in key:
            st.session_state[key] = False



def st_train_or_predict_select(key: str | None = None) -> str:
    """A select box to distinguish between train and predict.

    Args:
        key: A optional key if it's used multiple times.

    Returns:
        Either "train" or "predict".
    """
    return st.selectbox("Train or predict", ["train", "predict"],
                        key=f"{key}__st_train_or_predict_select")
