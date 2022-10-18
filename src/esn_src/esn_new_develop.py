# -*- coding: utf-8 -*-
""" Implements the Echo State Network (ESN) used in Reservoir Computing """

from __future__ import annotations

import numpy as np
from abc import ABC, abstractmethod

import src.esn_src.utilities as utilities

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

    # UPDATE FUNCTIONS:
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
        xnext = self.y_to_xnext_fct(self._last_y, self._last_x)

        return xnext

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

    def train(self, use_for_train: np.ndarray,
              sync_steps: int = 0):
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
        # rgen_array = self.r_to_rgen_fct(r_array)
        rgen_array = utilities.vectorize(self.r_to_rgen_fct, (r_array, ))

        # From rgen_array get rproc_array:
        rproc_array = self.set_rgen_to_rproc_fct(rgen_array=rgen_array)

        # from rproc_array get rfit_array.
        # rfit_array = self.rproc_to_rfit_fct(rproc_array, x_train)
        rfit_array = utilities.vectorize(self.rproc_to_rfit_fct, (rproc_array, x_train))

        # Perform the fitting:
        y_train_fit = self.set_rfit_to_y_fct(rfit_array=rfit_array, y_train=y_train)

        # Get real output from reservoir output:
        # xnext_train_fit = self.y_to_xnext_fct(y_train_fit)
        xnext_train_fit = utilities.vectorize(self.y_to_xnext_fct, (y_train_fit, ))

        return xnext_train_fit, x_train

    def predict(self, use_for_pred: np.ndarray,
                sync_steps: int = 0,
                pred_steps: int | None = None,
                reset_reservoir_bool: bool = True
                ):
        """Predict with the trained reservoir, i.e. let the reservoir run in a loop.

        Args:
            use_for_pred: The data to predict of shape (true_data + sync_steps, self.x_dim).
            sync_steps:  The time steps used to sync the reservoir before prediction.
            pred_steps:  Optional argument to define the number of steps to predict.
            reset_reservoir_bool: Whether to reset the reservoir before sync or not.

        Returns:
            The predicted and true time series.
            pred_data of shape (pred_steps, self.x_dim), and true_data of shape
            true_data, self.x_dim).
        """

        # Split sync and true predict steps:
        sync = use_for_pred[:sync_steps]
        true_data = use_for_pred[sync_steps:]

        # Reset the reservoir:
        if reset_reservoir_bool:
            self.reset_reservoir()

        # sync the reservoir:
        self.drive(sync)

        # Take as many pred_steps as there are true_data steps if None:
        if pred_steps is None:
            pred_steps = true_data.shape[0]

        # Array to save pred_data to:
        pred_data = np.zeros((pred_steps, self.x_dim))

        # loop reservoir to create prediction, i.e. populate pred_data ...

        # Take last reservoir state created during drive and get first output prediction:
        xnext = self.res_to_output_step()
        pred_data[0, :] = xnext

        for i in range(1, pred_steps):
            # Take predicted output to drive next reservoir update step:
            self.res_update_step(xnext)  # -> sets self._last_r

            xnext = self.res_to_output_step()  # -> get next output from self._last_r
            pred_data[i, :] = xnext

        return pred_data, true_data

    # BUILD FUNCTIONS:
    def set_default_r(self, default_r: np.ndarray) -> None:
        """Set the default reservoir state used to initialize."""
        self._default_r = default_r

    def reset_reservoir(self) -> None:
        """Reset the reservoir state."""
        self._last_r = self._default_r

    def set_node_bias(self, ) -> None:
        """Set the node bias. """
        self.node_bias = np.ones(self.r_dim)

    def set_leak_factor(self, leak_factor: float = 0.0) -> None:
        """Set the leak factor. """
        self.leak_factor = leak_factor

    def add_noise_to_x_train(self, x_train: np.ndarray) -> np.ndarray:
        """Add noise to x_train before training.
        """
        return x_train

    def build(self):
        """Build the basic quantities of the reservoir object. """
        self.x_dim = 3
        self.xproc_dim = 3
        self.r_dim = 100
        self.rgen_dim = 100
        self.rproc_dim = 100
        self.rfit_dim = 100
        self.y_dim = 3

        self.set_default_r(np.zeros(self.r_dim))
        self.set_node_bias()
        self.set_leak_factor()

class TestMixin:
    def __init__(self):
        # All the parameters additionally defined here:
        self.w_in: np.ndarray | None = None
        self.network: np.ndarray | None = None
        self.w_out: np.ndarray | None = None
        self.reg_param: float | None = None

    def activation_fct(self, r: np.ndarray) -> np.ndarray:
        return np.tanh(r)

    def x_to_xproc_fct(self, x: np.ndarray) -> np.ndarray:
        return x

    def xproc_to_res_fct(self, xproc: np.ndarray) -> np.ndarray:
        return self.w_in @ xproc

    def internal_res_fct(self, r: np.ndarray) -> np.ndarray:
        return self.network @ r

    def r_to_rgen_fct(self, r: np.ndarray) -> np.ndarray:
        return r

    def rgen_to_rproc_fct(self, rgen: np.ndarray) -> np.ndarray:
        return rgen

    def rproc_to_rfit_fct(self, rproc: np.ndarray,
                          x: np.ndarray | None = None) -> np.ndarray:
        return rproc

    def rfit_to_y_fct(self, rfit: np.ndarray) -> np.ndarray:
        return self.w_out @ rfit

    def y_to_xnext_fct(self, y: np.ndarray,
                       x: np.ndarray | None = None) -> np.ndarray:
        return y

    def set_rfit_to_y_fct(self, rfit_array: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        self.w_out = np.linalg.solve(
            rfit_array.T @ rfit_array + self.reg_param * np.eye(rfit_array.shape[1]),
            rfit_array.T @ y_train).T
        return (self.w_out @ rfit_array.T).T

    def set_x_to_xproc_fct(self, train: np.ndarray):
        pass

    def set_y_to_xnext_fct(self, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x_train = train[: -1, :]
        y_train = train[1:, :]
        return x_train, y_train

    def set_rgen_to_rproc_fct(self, rgen_array: np.ndarray) -> np.ndarray:
        return rgen_array

    def build(self) -> None:
        self.w_in = np.random.randn(self.r_dim, self.xproc_dim)

        self.network = np.random.randn(self.r_dim, self.r_dim)

        self.reg_param: float | None = 1e-7


class ESN_test(TestMixin,
               ResCompCore,
               ):
    """Test ESN class"""

    def __init__(self):
        ResCompCore.__init__(self)
        TestMixin.__init__(self)

    def build(self):

        ResCompCore.build(self)

        TestMixin.build(self)
