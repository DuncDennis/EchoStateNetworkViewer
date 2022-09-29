from __future__ import annotations

import streamlit as st

STATUS_TO_STATUS_LABEL_MAPPER = {"seed_bool": "🌱 Random seed",
                                 "raw_data_bool": "📼 Create raw data",
                                 "preproc_data_bool": "🌀 Preprocess data",
                                 "build_bool": "🛠️ Build RC",
                                 "tp_split_bool": "✂ Train-Predict split",
                                 "train_bool": "🦾 Train RC",
                                 "predict_bool": "🔮 Predict with RC"}

STATUS_TO_ICON_MAPPER = {"seed_bool": "🌱",
                         "raw_data_bool": "📼",
                         "preproc_data_bool": "🌀",
                         "build_bool": "🛠️",
                         "tp_split_bool": "✂",
                         "train_bool": "🦾",
                         "predict_bool": "🔮"}

STATUS_BOOL_TO_ICON_MAPPER = {True: "✅",
                              False: "❌"}

STATUS_NEEDED_TO_PROGRESS = {"seed_bool": [],
                             "raw_data_bool": [],
                             "preproc_data_bool": ["seed_bool", "raw_data_bool"],
                             "build_bool": ["seed_bool", "preproc_data_bool"],
                             "tp_split_bool": ["preproc_data_bool"],
                             "train_bool": ["tp_split_bool", "build_bool"],
                             "predict_bool": ["tp_split_bool", "train_bool"]}

def st_write_status(status_dict: dict[str, bool]) -> None:
    """Streamlit element to write the status of the RC pipeline.

    Can be used inside a container to display the status at the top.

    Args:
        status_dict: A dictionary of the form:
                     status_dict = {"seed_bool": False/True,
                                    "raw_data_bool": False/True,
                                    "preproc_data_bool": False/True,
                                    "build_bool": False/True,
                                    "tp_split_bool": False/True,
                                    "train_bool": False/True,
                                    "predict_bool": False/True}

    """

    cols = st.columns(2)
    for key, val in status_dict.items():
        phase_icon = STATUS_TO_STATUS_LABEL_MAPPER[key]
        status_icon = STATUS_BOOL_TO_ICON_MAPPER[val]
        cols[0].write(phase_icon)
        cols[1].write(status_icon)


def check_if_ready_to_progress(status_dict: dict[str, bool],
                               status_name: str
                               ) -> bool:
    """Check if the previous steps are all finished to perform the step corresponding to status_name.

    Args:
        status_dict: A dictionary of the form:
                     status_dict = {"seed_bool": False/True,
                                    "raw_data_bool": False/True,
                                    "preproc_data_bool": False/True,
                                    "build_bool": False/True,
                                    "tp_split_bool": False/True,
                                    "train_bool": False/True,
                                    "predict_bool": False/True}
        status_name: Name of the current status to check. E.g. "build_bool".

    Returns:
        True if ready to make the step corresponding to status_name, else False.
    """

    status_needed = STATUS_NEEDED_TO_PROGRESS[status_name]
    return all(status_dict[status_name] for status_name in status_needed)

def create_needed_status_string(status_dict: dict[str, bool],
                                status_name: str
                                ) -> str:
    """Function to create the string which discribes which steps are needed before progressing.

    Args:
        status_dict: A dictionary of the form:
                     status_dict = {"seed_bool": False/True,
                                    "raw_data_bool": False/True,
                                    "preproc_data_bool": False/True,
                                    "build_bool": False/True,
                                    "tp_split_bool": False/True,
                                    "train_bool": False/True,
                                    "predict_bool": False/True}
        status_name: Name of the current status to check. E.g. "build_bool".

    Returns:
        A string of the form: "Finish the step(s): [THE STEPS] to see something."
    """
    status_needed = STATUS_NEEDED_TO_PROGRESS[status_name]
    steps_need_finish = [f"[{STATUS_TO_STATUS_LABEL_MAPPER[key]}]" for key, val in
                         status_dict.items() if not val and key in status_needed]
    string_to_write = "Finish the step(s): " + ", ".join(steps_need_finish) + " to see something."
    return string_to_write


def create_needed_status_string_tab(status_name: str
                                    ) -> str:
    """Function to create the string which describes the current step that needs to be finished.

    Used in the tabs view.

    Args:
        status_name: Name of the current status to check. E.g. "build_bool".

    Returns:
        A string of the form: f'Finish [The step] to see something.'.
    """
    string_to_write = f'Finish [{STATUS_TO_STATUS_LABEL_MAPPER[status_name]}] to see something.'
    return string_to_write


# def st_main_checkboxes(key: str | None = None) -> tuple[bool, bool, bool, bool, bool]:
#     """Streamlit element to create 5 esn checkbs: Raw data, Prepr data, Build, Train and Predict.
#
#     Args:
#         key: A optional key if it's used multiple times.
#
#     Returns:
#         The states of the 5 checkboxes: raw_data_bool, preproc_data_bool, build_bool,
#         train_bool, predict_bool.
#     """
#
#     basic_key = f"{key}__st_main_checkboxes"
#
#     raw_data_checkbox_label = "📼 Get raw data"
#     preproc_checkbox_label = "🌀 Preprocess data"
#     build_checkbox_label = "🛠️ Build"
#     train_checkbox_label = "🦾 Train"
#     predict_checkbox_label = "🔮 Predict"
#
#     def on_change_raw():
#         if not st.session_state[f"{basic_key}__raw"]:
#             other_keys = [f"{basic_key}__{x}" for x in ("preproc", "build", "train", "pred")]
#             for k in other_keys:
#                 st.session_state[k] = False
#
#     def on_change_preproc():
#         if not st.session_state[f"{basic_key}__preproc"]:
#             other_keys = [f"{basic_key}__{x}" for x in ("build", "train", "pred")]
#             for k in other_keys:
#                 st.session_state[k] = False
#
#     def on_change_build():
#         if not st.session_state[f"{basic_key}__build"]:
#             other_keys = [f"{basic_key}__{x}" for x in ("train", "pred")]
#             for k in other_keys:
#                 st.session_state[k] = False
#
#     def on_change_train():
#         if not st.session_state[f"{basic_key}__train"]:
#             other_keys = [f"{basic_key}__{x}" for x in ("pred", )]
#             for k in other_keys:
#                 st.session_state[k] = False
#
#     raw_data_bool = st.checkbox(raw_data_checkbox_label,
#                                 key=f"{basic_key}__raw",
#                                 on_change=on_change_raw)
#
#     disabled = False if raw_data_bool else True
#     preproc_data_bool = st.checkbox(preproc_checkbox_label,
#                                     disabled=disabled,
#                                     key=f"{basic_key}__preproc",
#                                     on_change=on_change_preproc)
#
#     disabled = False if preproc_data_bool else True
#     build_bool = st.checkbox(build_checkbox_label,
#                              disabled=disabled,
#                              key=f"{basic_key}__build",
#                              on_change=on_change_build)
#
#     disabled = False if build_bool else True
#     train_bool = st.checkbox(train_checkbox_label,
#                              disabled=disabled,
#                              key=f"{basic_key}__train",
#                              on_change=on_change_train)
#
#     disabled = False if train_bool else True
#     predict_bool = st.checkbox(predict_checkbox_label,
#                                disabled=disabled,
#                                key=f"{basic_key}__pred")
#
#     return raw_data_bool, preproc_data_bool, build_bool, train_bool, predict_bool
#
#
# def uncheck_all_main_checkboxes(label_to_search: str = "st_main_checkboxes") -> None:
#     """Utility function to uncheck all main checkboxes created by st_main_checkboxes.
#
#     Has to be run before the checkboxes are initialized.
#
#     Args:
#         label_to_search: A string to be searched in the st.session_state keys.
#
#     """
#     for key in st.session_state.keys():
#         if label_to_search in key:
#             st.session_state[key] = False


def st_train_or_predict_select(key: str | None = None) -> str:
    """A select box to distinguish between train and predict.

    Args:
        key: A optional key if it's used multiple times.

    Returns:
        Either "train" or "predict".
    """
    return st.selectbox("Train or predict", ["train", "predict"],
                        key=f"{key}__st_train_or_predict_select")
