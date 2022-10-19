# -*- coding: utf-8 -*-
""" Implements the Echo State Network (ESN) used in Reservoir Computing """

from __future__ import annotations

from typing import Callable

from abc import ABC, abstractmethod
import numpy as np

# For networks:
import networkx as nx
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse.linalg.eigen.arpack.arpack \
    import ArpackNoConvergence as _ArpackNoConvergence

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
        self.x_train_noise_scale: float | None = None  # The noise scale used for the train noise.
        self.x_train_noise_seed: int | None = None  # The noise seed used for the train noise.

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
            x: The previous reservoir input x(t) of shape (self.x_dim, ).

        Returns:
            The output x(t+1) of shape (self.x_dim, ).
        """

    # SETTER FUNCTIONS DURING TRAINING.
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

    # SETTER FUNCTIONS CALLED in Mixins build:

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

    def reset_reservoir(self) -> None:
        """Reset the reservoir state."""
        self._last_r = self._default_r

    # BUILD FUNCTIONS:
    def set_default_r(self, default_r: np.ndarray | None = None) -> None:
        """Set the default reservoir state used to initialize."""

        if default_r is None:
            self._default_r = np.zeros(self.r_dim)
        else:
            self._default_r = default_r

    def set_node_bias(self,
                      node_bias_opt: str = "no_bias",
                      node_bias_scale: float = 0.1,
                      node_bias_seed: None | int = None
                      ) -> None:
        """Set the node bias. """

        if node_bias_opt == "no_bias":
            self.node_bias = np.zeros(self.r_dim)
        elif node_bias_opt == "random_bias":
            if node_bias_seed is not None:
                rng = np.random.default_rng(node_bias_seed)
                self.node_bias = node_bias_scale * rng.uniform(low=-1.0,
                                                               high=1.0,
                                                               size=self.r_dim)
            else: # Use global seed:
                self.node_bias = node_bias_scale * np.random.uniform(low=-1.0,
                                                                     high=1.0,
                                                                     size=self.r_dim)
        elif node_bias_opt == "constant_bias":
            self.node_bias =  node_bias_scale * np.ones(self.r_dim)

        else:
            raise ValueError(f"node_bias_opt {node_bias_opt} not recognized! "
                             f"Must be no_bias, random_bias or constant_bias. ")

    def set_leak_factor(self, leak_factor: float = 0.0) -> None:
        """Set the leak factor. """
        self.leak_factor = leak_factor

    def set_x_train_noise(self,
                          x_train_noise_scale: float | None = None,
                          x_train_noise_seed: int | None = None
                          ) -> None:
        """Set x_train_noise options.
        # TODO: Option to add noise scale in relation to std of x_train?
        # TODO: More noise options?
        """
        self.x_train_noise_scale = x_train_noise_scale
        self.x_train_noise_seed = x_train_noise_seed


    def add_noise_to_x_train(self, x_train: np.ndarray) -> np.ndarray:
        """Add noise to x_train before training.
        # TODO: see todos in set_x_train_noise.
        """
        if self.x_train_noise_scale is not None:
            if self.x_train_noise_seed is not None:
                rng = np.random.default_rng(self.x_train_noise_seed)
                x_train += rng.standard_normal(x_train.shape) * self.x_train_noise_scale
            else: # Use global seed:
                x_train += np.random.standard_normal(x_train.shape) * self.x_train_noise_scale

        return x_train

    def subbuild(self,
                 x_dim: int,
                 r_dim: int = 300,
                 leak_factor: float = 0.0,
                 node_bias_opt: str = "random_bias",
                 node_bias_seed: int | None = None,
                 node_bias_scale: float = 0.1,
                 default_r: np.ndarray | None = None,
                 x_train_noise_scale: float | None = None,
                 x_train_noise_seed: int | None = None
                 ):
        """Build the basic quantities of the reservoir object. """

        # Set dimensions:
        self.x_dim = x_dim
        self.r_dim = r_dim

        # Set default reservoir state:
        self.set_default_r(default_r)

        # Set node bias:
        self.set_node_bias(node_bias_opt=node_bias_opt,
                           node_bias_scale=node_bias_scale,
                           node_bias_seed=node_bias_seed)

        # set leak factor:
        self.set_leak_factor(leak_factor=leak_factor)

        # Set x_train noise:
        self.set_x_train_noise(x_train_noise_scale=x_train_noise_scale,
                               x_train_noise_seed=x_train_noise_seed)

class ActFunctionMixin:
    """Set the standard activation functions. """

    def __init__(self):
        self._act_fct: Callable[[np.ndarray], np.ndarray] | None = None

    def activation_fct(self, r: np.ndarray) -> np.ndarray:
        return self._act_fct(r)

    def subbuild(self, act_fct_opt: str = "tanh"):
        """Sub-build activation function."""
        if act_fct_opt == "tanh":
            self._act_fct = np.tanh
        elif act_fct_opt == "sigmoid":
            self._act_fct = utilities.sigmoid
        elif act_fct_opt == "relu":
            self._act_fct = utilities.relu
        elif act_fct_opt == "linear":
            self._act_fct = lambda r: r
        else:
            raise ValueError(f"act_fct_opt {act_fct_opt} not recognized! "
                             f"Must be tanh, sigmoid, relu or linear. ")

class RToRgenMixin:
    """Set the standard r_to_r_gen_fcts, i.e. readout functions. """
    def __init__(self):
        self._r_to_rgen_fct: Callable[[np.ndarray], np.ndarray] | None = None

    def r_to_rgen_fct(self, r: np.ndarray) -> np.ndarray:
        return self._r_to_rgen_fct(r)

    def subbuild(self, r_to_rgen_opt: str = "linear_r"):
        """Sub-build the r_to_rgen function. """
        if r_to_rgen_opt == "linear_r":
            self._r_to_rgen_fct = lambda r: r
        elif r_to_rgen_opt == "linear_and_square_r":
            self._r_to_rgen_fct = lambda r: np.hstack((r, r ** 2))
        elif r_to_rgen_opt == "output_bias":
            self._r_to_rgen_fct = utilities.add_one
        elif r_to_rgen_opt == "bias_and_square_r":
            self._r_to_rgen_fct = lambda r: np.hstack((np.hstack((r, r ** 2)), 1))
        elif r_to_rgen_opt == "linear_and_square_r_alt":
            def linear_and_square_alt(r):
                r_gen = np.copy(r).T
                r_gen[::2] = r.T[::2] ** 2
                return r_gen.T
            self._r_to_rgen_fct = lambda r: linear_and_square_alt(r)
        else:
            raise ValueError(f"r_to_rgen_opt {r_to_rgen_opt} not recognized! "
                             f"Must be linear_r, linear_and_square_r, output_bias, "
                             f"bias_and_square_r or linear_and_square_r_alt. ")

class ReservoirNetworkMixin:
    """The standard network update function.
    # TODO: functions are just more or less copied from old rescomp code.
    """
    def __init__(self) -> None:
        self._n_edge_prob: float | None = None
        self._n_rad: float | None = None
        self._n_avg_deg: float | None = None
        self._network: np.ndarray | None = None

    def internal_res_fct(self, r: np.ndarray) -> np.ndarray:
        """Abstract method to internally update the reservoir state."""
        return self._network @ r

    def _create_network_connections(self,
                                    n_type_opt: str,
                                    network_creation_seed: int | None = None
                                    ) -> None:
        """ Generate the baseline random network to be scaled"""

        if network_creation_seed is None:
            seed = np.random
        else:
            seed = network_creation_seed

        if n_type_opt == "erdos_renyi":
            network = nx.fast_gnp_random_graph(self.r_dim,
                                               self._n_edge_prob,
                                               seed=seed)
        elif n_type_opt == "scale_free":
            network = nx.barabasi_albert_graph(self.r_dim,
                                               int(self._n_avg_deg / 2),
                                               seed=seed)
        elif n_type_opt == "small_world":
            network = nx.watts_strogatz_graph(self.r_dim,
                                              k=int(self._n_avg_deg), p=0.1,
                                              seed=seed)
        elif n_type_opt == "erdos_renyi_directed":
            network = nx.fast_gnp_random_graph(self.r_dim,
                                               self._n_edge_prob,
                                               seed=seed,
                                               directed=True)
        elif n_type_opt == "random_dense":
            network = nx.from_numpy_matrix(np.ones((self.r_dim, self.r_dim)))

        else:
            raise ValueError(f"n_type_opt {n_type_opt} not recognized! "
                             f"Must be erdos_renyi, scale_free, small_world, "
                             f"erdos_renyi_directed or random_dense.")
        self._network = nx.to_numpy_array(network)

    def _vary_network(self, network_variation_attempts: int = 10) -> None:
        """ Varies the weights of self._network, while conserving the topology.

        The non-zero elements of the adjacency matrix are uniformly randomized,
        and the matrix is scaled (self.scale_network()) to self.spectral_radius.

        Specification done via protected members

        """

        # contains tuples of non-zero elements:
        arg_binary_network = np.argwhere(self._network)

        for i in range(network_variation_attempts):
            try:
                # uniform entries from [-0.5, 0.5) at non-zero locations:
                rand_shape = self._network[self._network != 0.].shape
                self._network[
                    arg_binary_network[:, 0], arg_binary_network[:, 1]] = \
                    np.random.random(size=rand_shape) - 0.5

                self._scale_network()

            except _ArpackNoConvergence:
                continue
            break
        else:
            raise _ArpackNoConvergence

    def _scale_network(self) -> None:
        """ Scale self._network, according to desired spectral radius.

        Can cause problems due to non converging of the eigenvalue evaluation

        Specification done via protected members

        """
        self._network = sparse.csr_matrix(self._network)
        try:
            eigenvals = linalg.eigs(
                self._network, k=1, v0=np.ones(self.r_dim),
                maxiter=1e3 * self.r_dim)[0]
        except _ArpackNoConvergence:
            raise

        maximum = np.absolute(eigenvals).max()
        self._network = ((self._n_rad / maximum) * self._network)

    def subbuild(self,
                 n_type_opt: str = "erdos_renyi",
                 n_rad: float = 0.1,
                 n_avg_deg: float = 6.0,
                 network_creation_attempts: int = 10,
                 network_creation_seed: int | None = None
                 ) -> None:
        """Sub/build to create network. """
        self._n_rad = n_rad
        self._n_avg_deg = n_avg_deg
        self._n_edge_prob = self._n_avg_deg / (self.r_dim - 1)
        for i in range(network_creation_attempts):
            try:
                self._create_network_connections(n_type_opt,
                                                 network_creation_seed)
                self._vary_network()
            except _ArpackNoConvergence:
                continue
            break
        else:
            raise Exception("Network creation during ESN init failed %d times"
                            % network_creation_attempts)

class ReservoirOutputFitMixin:
    """Standard Ridge Regression (RR) output fit from reservoir to output. """
    def __init__(self) -> None:
        self.reg_param: float | None = None  # the regularization parameter for RR.
        self.w_out: np.ndarray | None = None
        self._r_fit_to_y_fct: Callable[[np.ndarray], np.ndarray] | None = None
        self._ridge_regression_opt: str | None = None

    def rfit_to_y_fct(self, rfit: np.ndarray) -> np.ndarray:
        return self._r_fit_to_y_fct(rfit)

    def set_rfit_to_y_fct(self, rfit_array: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Is essentially the training. -> sets self.r_fit_to_y_fct.

        Args:
            rfit_array: Array to be fitted to y_train. Has shape (train_steps, self.rfit_dim).
            y_train: The desired output of shape (train_steps, self.y_dim).

        Returns:
            y_train_fit: The fitted output of shape (train_steps, self.y_dim).
        """
        if self._ridge_regression_opt == "no_bias":
            self._r_fit_to_y_fct = lambda rfit: self.w_out @ rfit
            self.w_out = np.linalg.solve(
                rfit_array.T @ rfit_array + self.reg_param * np.eye(rfit_array.shape[1]),
                rfit_array.T @ y_train).T

        elif self._ridge_regression_opt == "bias":
            self._r_fit_to_y_fct = lambda rfit: self.w_out @ utilities.add_one(rfit)
            rfit_array = utilities.vectorize(utilities.add_one, (rfit_array, ))
            eye = np.eye(rfit_array.shape[1])
            eye[-1, -1] = 0  # No regularization for bias term.
            self.w_out = np.linalg.solve(
                rfit_array.T @ rfit_array + self.reg_param * eye,
                rfit_array.T @ y_train).T

        return (self.w_out @ rfit_array.T).T

    def subbuild(self,
                 reg_param: float = 1e-8,
                 ridge_regression_opt: str = "no_bias") -> None:
        """Sub-build for Reservoir to output fitting.

        Args:
            reg_param: The regularization parameter for Ridge regression.
            ridge_regression_opt: A string either "no_bias" or "bias".
        """

        self.reg_param = reg_param

        if ridge_regression_opt not in ["no_bias", "bias"]:
            raise ValueError(f"ridge_regression_opt {ridge_regression_opt} not recognized! "
                             f"Must be no_bias or bias.")
        else:
            self._ridge_regression_opt = ridge_regression_opt

class SimpleRgenToRprocMixin:
    """Very simple rgen_to_rproc_fct: No processing of rgen states."""
    def __init__(self):
        pass

    def set_rgen_to_rproc_fct(self, rgen_array: np.ndarray) -> np.ndarray:
        """No processing of rgen states."""
        return rgen_array

    def rgen_to_rproc_fct(self, rgen: np.ndarray) -> np.ndarray:
        """No processing of rgen states."""
        return rgen

class SimpleRprocToRfitMixin:
    """Very simple rproc_to_rfit_fct: No modification of rproc."""
    def __init__(self):
        pass

    def rproc_to_rfit_fct(self, rproc: np.ndarray,
                          x: np.ndarray | None = None) -> np.ndarray:
        """No modification of rproc"""
        return rproc

class XToXProcSimple:
    """Simple x_to_xproc_fct: No processing of the input. """
    def __init__(self):
        pass

    def x_to_xproc_fct(self, x: np.ndarray) -> np.ndarray:
        """No processing"""
        return x

    def set_x_to_xproc_fct(self, train: np.ndarray):
        """No processing"""

    def subbuild(self):
        """Subbuild for no-processing x_to_xproc_fct.
        Just set xproc_dim to x_dim. """
        self.xproc_dim = self.x_dim

class SimpleInputMatrixMixin:
    """Simple Input matrix. """
    def __init__(self):
        self.w_in: np.ndarray | None = None

    def xproc_to_res_fct(self, xproc: np.ndarray) -> np.ndarray:
        return self.w_in @ xproc

    def subbuild(self,
                 w_in_opt: str = "random_sparse",
                 w_in_scale: float = 1.0,
                 w_in_seed: int | None = None) -> None:
        """Sub-build for simple input matrix w_in build.

        Args:
            w_in_opt: A string, either "random_sparse", "ordered_sparse", "random_dense_uniform",
                      or "random_dense_gaussian".
            w_in_scale: The input scale.
            w_in_seed: The seed for the random creation.

        """
        if w_in_seed is None:
            rng = np.random
        else:
            rng = np.random.default_rng(w_in_seed)

        self.w_in = np.zeros((self.r_dim, self.xproc_dim))

        if w_in_opt == "random_sparse":

            for i in range(self.r_dim):
                random_x_coord = rng.choice(np.arange(self.xproc_dim))
                self.w_in[i, random_x_coord] = rng.uniform(
                    low=-w_in_scale,
                    high=w_in_scale)

        elif w_in_opt == "ordered_sparse":
            dim_wise = np.array([int(self.r_dim / self.xproc_dim)] * self.xproc_dim)
            dim_wise[:self.r_dim % self.xproc_dim] += 1
            s = 0
            dim_wise_2 = dim_wise[:]
            for i in range(len(dim_wise_2)):
                s += dim_wise_2[i]
                dim_wise[i] = s
            dim_wise = np.append(dim_wise, 0)
            for d in range(self.xproc_dim):
                for i in range(dim_wise[d - 1], dim_wise[d]):
                    self.w_in[i, d] = rng.uniform(
                        low=-w_in_scale,
                        high=w_in_scale)

        elif w_in_opt == "random_dense_uniform":
            self.w_in = rng.uniform(low=-w_in_scale,
                                    high=w_in_scale,
                                    size=(self.r_dim, self.xproc_dim))

        elif w_in_opt == "random_dense_gaussian":
            self.w_in = w_in_scale * rng.randn(self.r_dim, self.xproc_dim)

class SimpleYtoXnextMixin:
    """The simplest resoutput Y to next input x function y_to_xnext_fct: Identity function. """
    def __init__(self):
        pass

    def y_to_xnext_fct(self, y: np.ndarray,
                       x: np.ndarray | None = None) -> np.ndarray:
        """Just the identity. """
        return y

    def set_y_to_xnext_fct(self, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Just the identity. Also do the connected x_train and y_train split. """
        x_train = train[: -1, :]
        y_train = train[1:, :]
        return x_train, y_train

    def subbuild(self):
        """Sub-build the Simple Y to next X Mixin. """
        pass


class ESN(
    ActFunctionMixin,
    ReservoirNetworkMixin,
    ReservoirOutputFitMixin,
    RToRgenMixin,
    SimpleRgenToRprocMixin,
    SimpleRprocToRfitMixin,
    XToXProcSimple,
    SimpleInputMatrixMixin,
    SimpleYtoXnextMixin,
    ResCompCore):

    def __init__(self):
        ResCompCore.__init__(self)
        ActFunctionMixin.__init__(self)
        ReservoirNetworkMixin.__init__(self)
        ReservoirOutputFitMixin.__init__(self)
        SimpleRgenToRprocMixin.__init__(self)
        SimpleRprocToRfitMixin.__init__(self)
        XToXProcSimple.__init__(self)
        SimpleInputMatrixMixin.__init__(self)
        SimpleYtoXnextMixin.__init__(self)
        RToRgenMixin.__init__(self)

    def build(self,

              # BASIC:
              x_dim: int,
              r_dim: int = 500,
              leak_factor: int = 0.0,
              node_bias_opt: str = "random_bias",
              node_bias_seed: int | None = None,
              node_bias_scale: float = 0.1,
              default_r: np.ndarray | None = None,
              x_train_noise_scale: float | None = None,
              x_train_noise_seed: int | None = None,

              #ACT FCT:
              act_fct_opt: str = "tanh",

              # Reservoir Network:
              n_type_opt: str = "erdos_renyi",
              n_rad: float = 0.1,
              n_avg_deg: float = 6.0,
              network_creation_attempts: int = 10,
              network_creation_seed: int | None = None,

              # R to Rgen:
              r_to_rgen_opt: str = "linear_r",

              # Output fit / Training:
              reg_param: float = 1e-8,
              ridge_regression_opt: str = "no_bias",

              # Input Matrix:
              w_in_opt: str = "random_sparse",
              w_in_scale: float = 1.0,
              w_in_seed: int | None = None
              ):

        # Basic Rescomp build:
        ResCompCore.subbuild(
            self,
            x_dim=x_dim,
            r_dim=r_dim,
            leak_factor=leak_factor,
            node_bias_opt=node_bias_opt,
            node_bias_seed=node_bias_seed,
            node_bias_scale=node_bias_scale,
            default_r=default_r,
            x_train_noise_scale=x_train_noise_scale,
            x_train_noise_seed=x_train_noise_seed,
        )

        # Activation Function build:
        ActFunctionMixin.subbuild(
            self,
            act_fct_opt=act_fct_opt)

        # Reservoir Network build:
        ReservoirNetworkMixin.subbuild(
            self,
            n_type_opt=n_type_opt,
            n_rad=n_rad,
            n_avg_deg=n_avg_deg,
            network_creation_attempts=network_creation_attempts,
            network_creation_seed=network_creation_seed,
        )

        # R to Rgen build:
        RToRgenMixin.subbuild(
            self, r_to_rgen_opt=r_to_rgen_opt)

        # Res output fit:
        ReservoirOutputFitMixin.subbuild(
            self,
            reg_param=reg_param,
            ridge_regression_opt=ridge_regression_opt)

        # Simple X to XProc:
        XToXProcSimple.subbuild(self)

        # Input Matrix build:
        SimpleInputMatrixMixin.subbuild(
            self,
            w_in_opt=w_in_opt,
            w_in_scale=w_in_scale,
            w_in_seed=w_in_seed
            )


        # SimpleYtoXnextMixin.subbuild(self) # after xproc dim.
        # SimpleRgenToRprocMixin no subbuild needed.
        # SimpleRprocToRfitMixin no subbuild needed.
