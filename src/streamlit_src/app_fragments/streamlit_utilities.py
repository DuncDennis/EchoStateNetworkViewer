"""Utlity streamlit fragments."""
from __future__ import annotations

from typing import Any

import numpy as np
import streamlit as st


def st_line() -> None:
    """Draw a seperator line."""
    st.markdown("""---""")


def st_selectbox_with_all(name: str, options: list[str], key: str | None = None) -> list[str]:
    """A streamlit element for a multiselect with a "select all" checkbox.

    Args:
        name: The name of the multiselect.
        options: The options in the multiselect.
        key: A optional key if it's used multiple times.

    Returns:
        The selection.
    """
    container = st.container()
    all = st.checkbox("Select all", key=f"{key}__select_all")
    if all:
        default = options
    else:
        default = options[0]

    selection = container.multiselect(name, options, default=default, key=f"{key}__multi_select")

    return selection


def st_dimension_selection(dimension: int, key: str | None = None) -> int:
    """A number input to select the dimension.

    Args:
        dimension: The dimension of the system.
        key: A possible unique key.

    Returns:
        The selected dimension.
    """

    return st.number_input("Dimension", value=0, max_value=dimension-1,
                           min_value=0, key=f"{key}__dim_selection")


def st_dimension_selection_multiple(dimension: int, key: str | None = None) -> list[int]:
    """Streamlit element to select multiple dimensions.

    Args:
        dimension: The maximal dimension that can be selected.

    Returns:
        A list of integers representing the selected dimensions.
    """

    dim_select_opts = [f"{i}" for i in range(dimension)]
    dim_selection = st_selectbox_with_all("Dimensions", dim_select_opts,
                                          key=f"{key}__dim_select_mult")
    dim_selection = [int(x) for x in dim_selection]
    return dim_selection


@st.experimental_memo
def get_random_int() -> int:
    """Get a random integer between 1 and 1000000.
    TODO: maybe handle with generators in the future.
    Is used to get a new seed.

    Returns:
        The random integer.
    """
    return np.random.randint(1, 1000000)


def st_seed(key: str | None = None) -> int:
    """Streamlit element to specify the random seed.

    Args:
        key: A optional key if it's used multiple times.

    Returns:
        The seed.
    """
    new_seed = st.button("new random seed", key=f"{key}__new_seed")
    if new_seed:
        get_random_int.clear()
        seed = get_random_int()
        st.session_state[f"{key}__st_seed__seed"] = seed

    seed = int(st.number_input("Seed", max_value=1000000, key=f"{key}__st_seed__seed"))
    return seed


def st_add_to_state(name: str, value: Any) -> None:
    """Add a variable to the session state.

    The name will be saved as f"{name}".

    Args:
        name: The name of the session state variable.
        value: The value of the variable.

    """
    st.session_state[name] = value


def st_get_session_state(name: str) -> Any:
    """Get a variable of session state by defining the prefix and name.

    Args:
        name: The name of the session state variable.

    Returns:
        The value of the variable.
    """
    if name in st.session_state:
        return st.session_state[name]
    else:
        return None


def st_reset_all_check_boxes(key: str | None = None) -> None:
    """Streamlit button to reset all session state variables that are boolean to false.

    This is used to set all checkboxes back to False.

    Args:
        key: A optional key if it's used multiple times.

    """
    help = "Sets all checkboxes in the app back to false. "
    if st.button("Untick all", help=help, key=f"{key}__st_reset_all_check_boxes"):
        true_checkboxes = {key: val for key, val in st.session_state.items() if
                           type(val) == bool and val is True}
        for k in true_checkboxes.keys():
            if k == f"{key}__st_reset_all_check_boxes":
                continue
            else:
                st.session_state[k] = False


def clear_all_cashes() -> None:
    """Function to clear all cashed values.

    TODO: Not really clearing all caches. st.cache is not cleared (but used in train and predict and build)
    """
    st.experimental_memo.clear()
    st.experimental_singleton.clear()


def st_clear_all_cashes_button(key: str | None = None) -> None:
    """Streamlit button to clear all cashed values.

    Args:
        key: A optional key if it's used multiple times.

    """
    help = "Clears all cashed values. Use if app gets too slow because of memory issues."
    if st.button("Clear cash", help=help, key=f"{key}__st_clear_all_cashes"):
        clear_all_cashes()


if __name__ == '__main__':
    pass
