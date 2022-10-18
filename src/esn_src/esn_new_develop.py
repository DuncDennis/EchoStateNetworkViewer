# -*- coding: utf-8 -*-
""" Implements the Echo State Network (ESN) used in Reservoir Computing """


# TODO: Check imports:
from __future__ import annotations

from typing import Callable, Tuple, Any

import numpy as np
from abc import ABC, abstractmethod

from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse.linalg.eigen.arpack.arpack \
    import ArpackNoConvergence as _ArpackNoConvergence
import networkx as nx
from sklearn import decomposition

from src.esn_src import utilities

class ResCompCore(ABC):
    """
    Reservoir Computing base class with the following data flow:

    x_i --[x_to_xproc_fct]--> xproc_i --[xproc_to_res_fct]--> r_i --[r_to_rgen_fct]--> rgen_i
    --[rgen_to_rproc_fct]--> rproc_i --[rproc_to_rfit_fct]--> rfit_i --[rfit_to_y_fct]--> y_i
    --[y_to_xnext]--> x_{i+1}.
    """

    def __init__(self):

        # DIMENSIONS:
        self.r_dim: int | None = None  # reservoir dimension.
        self.rgen_dim: int | None = None  # generalized reservoir state dimension.
        self.rproc_dim: int | None = None  # processed reservoir state dimensions.
        self.rfit_dim: int | None = None   # final to-fit reservoir state dimension.
        self.x_dim: int | None = None  # Input data dimension.
        self.xproc_dim: int | None = None # Preprocessed input data dimension.
        self.y_dim: int | None = None  # Output data dimension.

        # FIXED PARAMETERS:
        self.leak_factor: float | None = None  # leak factor lf (0=<lf=<1) in res update equation.
        self.node_bias: np.ndarray | None = None  # Node bias in res update equation.

        # OTHER SETTINGS:
        self._default_r: np.ndarray | None = None  # The default starting reservoir state.

        # DYNAMIC INTERNAL QUANTITIES:
        self._last_x: np.ndarray | None = None
        self._last_xproc: np.ndarray | None = None
        self._last_rinp: np.ndarray | None = None
        self._last_rinternal: np.ndarray | None = None
        self._last_r: np.ndarray | None = None
        self._last_rgen: np.ndarray | None = None
        self._last_rproc: np.ndarray | None = None
        self._last_rfit: np.ndarray | None = None
        self._last_y: np.ndarray | None = None

        # SAVED INTERNAL QUANTITIES:
        # TODO: Check how to handle.
        self._saved_res_inp = None
        self._saved_r_internal = None
        self._saved_r = None
        self._saved_r_gen = None
        self._saved_out = None
        self._saved_y_train = None

    # ABSTRACT TRANSFER FUNCTIONS:

    @abstractmethod
    def activation_fct(self, r: np.ndarray) -> np.ndarray:
        """Abstract method for the element wise activation function"""

    @abstractmethod
    def x_to_xproc_fct(self, x: np.ndarray) -> np.ndarray:
        """Abstract method to connect the raw input with the processed input."""

    @abstractmethod
    def xproc_to_res_fct(self, xproc: np.ndarray) -> np.ndarray:
        """Abstract method to connect the processed input with the reservoir."""

    @abstractmethod
    def internal_res_fct(self, r: np.ndarray) -> np.ndarray:
        """Abstract method to internally update the reservoir state."""

    @abstractmethod
    def r_to_rgen_fct(self, r: np.ndarray) -> np.ndarray:
        """Abstract method to transform r to rgen."""

    @abstractmethod
    def rgen_to_rproc_fct(self, rgen: np.ndarray) -> np.ndarray:
        """Abstract method to transform the rgen to rproc."""

    @abstractmethod
    def rproc_to_rfit_fct(self, rproc: np.ndarray,
                          x: np.ndarray | None = None) -> np.ndarray:
        """Abstract method to transform the rproc to rfit.
            Might optionally get a input connection.
        """

    @abstractmethod
    def rfit_to_y_fct(self, rfit: np.ndarray) -> np.ndarray:
        """Abstract method to transform the rfit to the reservoir output y."""

    @abstractmethod
    def y_to_xnext_fct(self, y: np.ndarray,
                       x: np.ndarray | None = None) -> np.ndarray:
        """Abstract method to connect the reservoir output with the required output.

        Usually the required output is the x(t+1), and normally y(t) = x(t+1),
        but some architectures also require the input x(t)

        There is a possibility to also use the previous input x, as in hybrid-ESN.

        Args:
            y: The reservoir output y(t) of shape (self._y_dim, ).
            x: The previous reservoir input x(t) of shape (self._x_dim, ).

        Returns:
            The output x(t+1) of shape (self._x_dim, ).
        """

    # SETTER FUNCTIONS for functions that have to be set during training.

    @abstractmethod
    def set_rfit_to_y_fct(self, rfit_array: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Is essentially the training. -> sets self.r_fit_to_y_fct.

        Args:
            rfit_array: Array to be fitted to y_train. Has shape (train_steps, self.rfit_dim).
            y_train: The desired output of shape (train_steps, self.y_dim).

        Returns:
            y_train_fit: The fitted output of shape (train_steps, self.y_dim).
        """

    @abstractmethod
    def set_x_to_xproc_fct(self, train: np.ndarray):
        """Set the input preprocess function."""

    # @abstractmethod
    # def set_input_to_res_fct(self, train: np.ndarray):
    #     """Set the input_to_res_fct.
    #     May use train data for preprocessing.
    #     """

    @abstractmethod
    def set_y_to_xnext_fct(self, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Set the y to xnext function using the train data.
        return x_train and y_train.
        """

    @abstractmethod
    def set_rgen_to_rproc_fct(self, rgen_array: np.ndarray) -> np.ndarray:
        """Process rgen_array, define rgen_to_rproc_fct and return rproc_array.
        Preprocessing of rgen_array: normally identity.
        """

    @abstractmethod
    def add_noise_to_x_train(self, x_train: np.ndarray):
        """Add noise to x_train before training.
        # TODO: check if necessary.
        """

    def set_default_r(self, default_r: np.ndarray) -> None:
        """Set the default reservoir state used to initialize."""
        self._default_r = default_r

    def reset_reservoir(self) -> None:
        """Reset the reservoir state."""
        self._last_r = self._default_r


    # INTERNAL UPDATE FUNCTIONS:
    def res_update_step(self, x: np.ndarray):
        """Update the reservoir state from r_i to r_(i+1), save the resulting reservoir state.

        Updates self._last_r. Assumes self._last_r is not None.

        Args:
            x: The input of shape (sys_dim, ).
        """

        # Save driving input:
        self._last_x = x

        # Preprocess the input.
        self._last_xproc = self.x_to_xproc_fct(self._last_x)

        # Create the reservoir-input-term.
        self._last_rinp = self.xproc_to_res_fct(self._last_xproc)

        # Create the reservoir-reservoir-term.
        self._last_rinternal = self.internal_res_fct(self._last_r)

        # Update the reservoir state:
        self._last_r = self.leak_factor * self._last_r + (1 - self.leak_factor) * \
                       self.activation_fct(self._last_rinp +
                                           self._last_rinternal +
                                           self.node_bias)

        # Return next reservoir state:
        return self._last_r

    def res_to_output_step(self):
        """Create output given self._last_r (and self._last_x).

        Returns:
            The output corresponding to self._last_r of shape (self.x_dim, ).
        """
        # last reservoir to generalized reservoir state:
        self._last_rgen = self.r_to_rgen_fct(self._last_r)

        # last generalized reservoir state to processed reservoir state:
        self._last_rproc = self.rgen_to_rproc_fct(self._last_rgen)

        # last processed reservoir state to fitting-reservoir state:
        self._last_rfit = self.rproc_to_rfit_fct(self._last_rproc, self._last_x)

        # last fitting-reservoir state to reservoir output y:
        self._last_y = self.rfit_to_y_fct(self._last_rfit)

        # last reservoir output to real output = next step in time series:
        self._last_x = self.y_to_xnext_fct(self._last_y, self._last_x)

        # Return final output, i.e. the next step in time series:
        return self._last_x

    def drive(self, x_array: np.ndarray) -> np.ndarray:
        """Drive the reservoir using the input x_array.

        Args:
            x_array: The input time series of shape (time_steps, self.x_dim).

        Returns:
            The saved reservoir states.
        """

        drive_steps = x_array.shape[0]

        r_array = np.zeros((drive_steps, self.r_dim))

        for i in range(drive_steps):
            x = x_array[i, :]
            r_array[i, :] = self.res_update_step(x)

        return r_array

    def train(self, use_for_train: np.ndarray, sync_steps: int = 0):
        """Synchronize and train the reservoir.

        Args:
            use_for_train: The data used for training of the shape
                           (sync_steps + train_steps, self.x_dim).
            sync_steps: The number of steps to synchronize the reservoir.

        Returns:
            x_train_fit and x_train.
        """

        # Split sync and train steps:
        sync = use_for_train[:sync_steps]
        train = use_for_train[sync_steps:]

        # Set the input preprocess function (which depends on train).
        self.set_x_to_xproc_fct(train)

        # Split train into x_train (input to reservoir) and y_train (output of reservoir).
        x_train, y_train = self.set_y_to_xnext_fct(train)

        # Add input noise:
        x_train = self.add_noise_to_x_train(x_train)

        # reset reservoir state.
        self.reset_reservoir()

        # Synchronize reservoir:
        self.drive(sync)  # sets the reservoir state self._last_r to be synced to the input.

        # Train synchronized ...
        # Drive reservoir to get r states:
        r_array = self.drive(x_train)

        # From r_array get rgen_array:
        rgen_array = self.r_to_rgen_fct(r_array)

        # From rgen_array get rproc_array:
        rproc_array = self.set_rgen_to_rproc_fct(rgen_array=rgen_array)

        # from rproc_array get rfit_array.
        rfit_array = self.rproc_to_rfit_fct(rproc_array, x_train)

        # Perform the fitting:
        y_train_fit = self.set_rfit_to_y_fct(rfit_array=rfit_array, y_train=y_train)

        # Get real output from reservoir output:
        xnext_train_fit = self.y_to_xnext_fct(y_train_fit)

        return xnext_train_fit, x_train

    def train_synced(self,
                     x_train: np.ndarray,
                     y_train: np.ndarray):
        """Train the synchronized reservoir on input x_train and output y_train.

        Args:
            x_train: Input array of shape (time steps, self.x_dim) to drive the reservoir.
            y_train: Desired output array of shape (time steps, self.y_dim).

        Returns:
            (y_train_fit and y_train).
        """

        # Get r_array.
        self.drive(x_train)
        r_array = None

        # From r_array get rgen_array.
        rgen_array = self.r_to_rgen_fct(r_array)  # vectorized

        # From rgen_array get rproc_array.
        rproc_array = self.set_rgen_to_rproc_fct(rgen_array=rgen_array)

        # from rproc_array get rfit_array.
        rfit_array = self.rproc_to_rfit_fct(rproc_array, x_train)  # vectorized

        # Train the output layer, return the fitted y_train_fit.
        y_train_fit = self.set_rfit_to_y_fct(rfit_array=rfit_array, y_train=y_train)

        # Res output to output:

        return y_train_fit, y_train



    def loop(self, steps: int):
        """"""

        # Create first prediction from last saved reservoir state (produced during drive).
        self._last_rgen = self.r_to_rgen_fct(self._last_r)
        self._last_rproc = self.rgen_to_rproc_fct(self._last_rgen)
        self._last_rfit = self.rproc_to_rfit_fct(self._last_rproc, self._last_x)
        self._last_y = self.rfit_to_y_fct(self._last_rfit)
        self._last_x = self.y_to_xnext_fct(self._last_y, self._last_x)

        # Loop for (steps - 1) steps:
        for i in range(1, steps):  # One step less:
            self._res_update(self._last_x)
            self._last_rproc = self.rgen_to_rproc_fct(self._last_rgen)
            self._last_rfit = self.rproc_to_rfit_fct(self._last_rproc, self._last_x)
            self._last_y = self.rfit_to_y_fct(self._last_rfit)
            self._last_x = self.y_to_xnext_fct(self._last_y, self._last_x)


    def predict(self, use_for_pred: np.ndarray, sync_steps: int = 0):
        # Reset reservoir state

        # Synchronize the reservoir:
        if sync_steps > 0:
            sync = use_for_pred[:sync_steps]
            true_data = use_for_pred[sync_steps:]
            self.drive(sync)
        else:
            true_data = use_for_pred

        steps = true_data.shape[0]
        return self.loop(steps), true_data


    def _r_gen_to_out_fct(self, r_gen: np.ndarray) -> np.ndarray:
        return self._w_out @ r_gen

    def _r_to_r_gen(self):
        self._last_r_gen = self._r_to_r_gen_fct(self._last_r, self._last_x)

    def _res_gen_to_output(self):
        self._last_y = self._r_gen_to_out_fct(self._last_r_gen)

    def _out_to_inp(self):
        return self._y_to_x_fct(self._last_x, self._last_y)

    def drive(self, input, save_res_inp=False, save_r_internal=False, save_r=False,
              save_r_gen=False, save_out=False) -> None:
        """Drive the reservoir with an input, optionally save various reservoir and output states.
        """
        steps = input.shape[0]

        if save_res_inp:
            self._saved_res_inp = np.zeros((steps, self._r_dim))
        if save_r_internal:
            self._saved_r_internal = np.zeros((steps, self._r_dim))
        if save_r:
            self._saved_r = np.zeros((steps, self._r_dim))
        if save_r_gen:
            self._saved_r_gen = np.zeros((steps, self._r_gen_dim))
        if save_out:
            self._saved_out = np.zeros((steps, self._y_dim))

        for i_x, x in enumerate(input):
            self._res_update(x)

            if save_res_inp:
                self._saved_res_inp[i_x, :] = self._last_res_inp
            if save_r_internal:
                self._saved_r_internal[i_x, :] = self._last_r_interal
            if save_r:
                self._saved_r[i_x, :] = self._last_r
            if save_r_gen or save_out:
                self._r_to_r_gen()
                if save_r_gen:
                    self._saved_r_gen[i_x, :] = self._last_r_gen
                if save_out:
                    self._res_gen_to_output()
                    self._saved_out[i_x, :] = self._last_y

    def _fit_w_out(self, y_train, r_gen_train):
        self._w_out = np.linalg.solve(
            r_gen_train.T @ r_gen_train + self._reg_param * np.eye(r_gen_train.shape[1]),
            r_gen_train.T @ y_train).T

    def train_synced(self, x_train, y_train, save_y_train=False, **kwargs) -> tuple[Any, Any]:
        """Train the synced reservoir.

        Drive the reservoir with x_train, get the r_gen states corresponding to x_train,
        and get _w_out by fitting the r_gen states to y_train.

        Args:
            x_train: The input array of shape (train_steps - 1, sys_dim).
            y_train: The output array of shape (train_steps - 1, sys_dim).
            save_y_train: If true, save the true y_train.
            **kwargs:
        """
        kwargs["save_r_gen"] = True

        save_out = False
        if "save_out" in kwargs.keys():
            if kwargs["save_out"]:  # can not save out during training before w_out is calculated
                save_out = True
                kwargs["save_out"] = False

        self.drive(x_train, **kwargs)
        r_gen_train = self._saved_r_gen
        self._fit_w_out(y_train, r_gen_train)

        # Added because now always return y_train_fit:
        y_train_fit = (self._w_out @ self._saved_r_gen.T).T

        if save_y_train:
            self._saved_y_train = y_train

        if save_out:
            self._saved_out = y_train_fit

        # Added to always return y_train and y_train_fit:
        return y_train_fit, y_train

    def loop(self, steps, save_res_inp=False, save_r_internal=False, save_r=False,
             save_r_gen=False, save_out=False):
        if save_res_inp:
            self._saved_res_inp = np.zeros((steps, self._r_dim))
        if save_r_internal:
            self._saved_r_internal = np.zeros((steps, self._r_dim))
        if save_r:
            self._saved_r = np.zeros((steps, self._r_dim))
        if save_r_gen:
            self._saved_r_gen = np.zeros((steps, self._r_gen_dim))
        if save_out:
            self._saved_out = np.zeros((steps, self._y_dim))

        x_pred = np.zeros((steps, self._x_dim))
        self._r_to_r_gen()
        self._res_gen_to_output()
        x = self._out_to_inp()
        x_pred[0, :] = x

        if save_res_inp:
            self._saved_res_inp[0, :] = self._last_res_inp
        if save_r_internal:
            self._saved_r_internal[0, :] = self._last_r_interal
        if save_r:
            self._saved_r[0, :] = self._last_r
        if save_r_gen:
            self._saved_r_gen[0, :] = self._last_r_gen
        if save_out:
            self._saved_out[0, :] = self._last_y

        for i in range(1, steps):
            self._res_update(x)
            self._r_to_r_gen()
            self._res_gen_to_output()
            x = self._out_to_inp()
            x_pred[i, :] = x
            if save_res_inp:
                self._saved_res_inp[i, :] = self._last_res_inp
            if save_r_internal:
                self._saved_r_internal[i, :] = self._last_r_interal
            if save_r:
                self._saved_r[i, :] = self._last_r
            if save_r_gen:
                self._saved_r_gen[i, :] = self._last_r_gen
            if save_out:
                self._saved_out[i, :] = self._last_y
        return x_pred

    def train(self, x_sync, x_train, y_train, reset_res_state=True, **kwargs) -> tuple[Any, Any]:
        if reset_res_state:
            self.reset_r()
        self.drive(x_sync)
        return self.train_synced(x_train, y_train, **kwargs)

    def predict(self, use_for_pred, sync_steps=0, reset_res_state=True, **kwargs):
        if reset_res_state:
            self.reset_r()

        if sync_steps > 0:
            sync = use_for_pred[:sync_steps]
            true_data = use_for_pred[sync_steps:]
            self.drive(sync)
        else:
            true_data = use_for_pred

        steps = true_data.shape[0]
        return self.loop(steps, **kwargs), true_data

    def set_r(self, r):
        self._last_r = r

    def reset_r(self):
        self.set_r(self._default_r)

    def get_act_fct_inp(self):
        return self._saved_res_inp + self._saved_r_internal + self._node_bias

    def get_res_inp(self):
        return self._saved_res_inp

    def get_r_internal(self):
        return self._saved_r_internal

    def get_r(self):
        return self._saved_r

    def get_r_gen(self):
        return self._saved_r_gen

    def get_out(self):
        return self._saved_out

    def get_y_train(self):
        return self._saved_y_train

    def get_w_out(self):
        """Return w_out matrix.

        Returns: w_out matrix of shape (y_dim, r_gen_dim).

        """
        return self._w_out

    def get_dimensions(self) -> tuple[int, int, int, int]:
        """Return the dimensions of input, reservoir, gen. reservoir and output.

        Returns:
            Tuple of the four dimensions: input x, reservoir r, gen reservoir r_gen and output y.
        """
        return self._x_dim, self._r_dim, self._r_gen_dim, self._y_dim

    def get_act_fct(self) -> None | Callable[[np.ndarray], np.ndarray]:
        return self._act_fct

    def get_res_iterator_func(self) -> Callable[[np.ndarray], np.ndarray]:
        """Return the reservoir update function, once everything is trained and specfified.

        Note: Only works if the r_to_r_gen_fct does not depend on the input x.

        Returns:
            The reservoir update function.
        """
        def res_iterator(r):
            last_out = self._r_gen_to_out_fct(self._r_to_r_gen_fct(r, None))
            r_iplus1 = self._leak_factor * r + (1 - self._leak_factor) * self._act_fct(
                self._inp_coupling_fct(last_out) +
                self._res_internal_update_fct(r) + self._node_bias)
            return r_iplus1

        return res_iterator
