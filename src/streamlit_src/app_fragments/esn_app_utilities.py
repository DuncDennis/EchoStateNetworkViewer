from __future__ import annotations

import streamlit as st


def st_main_checkboxes(key: str | None = None) -> tuple[bool, bool, bool, bool]:
    """Streamlit element to create 4 esn checkboxes: Simulate data, Build, Train and Predict.

    Args:
        key: A optional key if it's used multiple times.

    Returns:
        The states of the four checkboxes: simulate_bool, build_bool, train_bool, predict_bool.
    """

    basic_key = f"{key}__st_main_checkboxes"

    def on_change_sim():
        if not st.session_state[f"{basic_key}__sim"]:
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

    simulate_bool = st.checkbox("🌀Simulate data", key=f"{basic_key}__sim",
                                on_change=on_change_sim)

    disabled = False if simulate_bool else True
    build_bool = st.checkbox("🛠️ Build", disabled=disabled, key=f"{basic_key}__build",
                             on_change=on_change_build)

    disabled = False if build_bool else True
    train_bool = st.checkbox("🦾 Train", disabled=disabled, key=f"{basic_key}__train",
                             on_change=on_change_train)

    disabled = False if train_bool else True
    predict_bool = st.checkbox("🔮 Predict", disabled=disabled, key=f"{basic_key}__pred")

    return simulate_bool, build_bool, train_bool, predict_bool


def st_train_or_predict_select(key: str | None = None) -> str:
    """A select box to distinguish between train and predict.

    Args:
        key: A optional key if it's used multiple times.

    Returns:
        Either "train" or "predict".
    """
    return st.selectbox("Train or predict", ["train", "predict"],
                        key=f"{key}__st_train_or_predict_select")
