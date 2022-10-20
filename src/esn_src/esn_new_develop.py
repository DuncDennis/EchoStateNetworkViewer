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
# For Input/Output/Rgen scaler:
from sklearn.preprocessing import StandardScaler
# FOR Rgen pca:
from sklearn.decomposition import PCA

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

    @abstractmethod
    def set_rproc_to_rfit_fct(self, rproc_array: np.ndarray, x_train: np.ndarray) -> np.ndarray:
        """Abstract method to set the rproc_to_rfit_fct.
        Called during training"""

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

    def drive(self, x_array: np.ndarray,
              more_out_bool: bool = False
              ) -> np.ndarray | tuple[np.ndarray, dict[str, np.ndarray]]:
        """Drive the reservoir using the input x_array.

        Args:
            x_array: The input time series of shape (time_steps, self.x_dim).
            more_out_bool: Return more internal states if true.

        Returns:
            The saved reservoir states.
        """

        drive_steps = x_array.shape[0]

        r_array = np.zeros((drive_steps, self.r_dim))

        if more_out_bool:
            more_out = {"xproc": np.zeros((drive_steps, self.xproc_dim)),
                        "rinp": np.zeros((drive_steps, self.r_dim)),
                        "rinternal": np.zeros((drive_steps, self.r_dim))}

        for i in range(drive_steps):
            x = x_array[i, :]
            r_array[i, :] = self.res_update_step(x)

            if more_out_bool:
                more_out["xproc"][i, :] = self._last_xproc
                more_out["rinp"][i, :] = self._last_rinp
                more_out["rinternal"][i, :] = self._last_rinternal

        if more_out_bool:
            return r_array, more_out

        return r_array

    def train(self,
              use_for_train: np.ndarray,
              sync_steps: int = 0,
              more_out_bool: bool = False
              ) -> tuple[np.ndarray, np.ndarray] | \
                   tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        """Synchronize and train the reservoir.
        Args:
            use_for_train: The data used for training of the shape
                           (sync_steps + train_steps, self.x_dim).
            sync_steps: The number of steps to synchronize the reservoir.
            more_out_bool: Return more internal states if true.

        Returns:
            x_train_fit, x_train, (and more_out optionally).
            more_out = {"xproc": xproc_array,
                        "rinp": rinp_array,
                        "rinternal": rinternal_array,
                        "r": r_array,
                        "rgen": rgen_array,
                        "rproc": rproc_array,
                        "rfit_array": rfit_array,
                        "y": y_train_fit}.
        """

        # Split sync and train steps:
        sync = use_for_train[:sync_steps]
        train = use_for_train[sync_steps:]

        # Split train into x_train (input to reservoir) and y_train (output of reservoir).
        x_train, y_train = self.set_y_to_xnext_fct(train)

        # Set the input preprocess function (which depends on train).
        self.set_x_to_xproc_fct(x_train)

        # Add input noise:
        x_train_w_noise = self.add_noise_to_x_train(x_train)

        # reset reservoir state.
        self.reset_reservoir()

        # Synchronize reservoir:
        self.drive(sync)  # sets the reservoir state self._last_r to be synced to the input.

        # Train synchronized ...
        # Drive reservoir to get r states:
        drive_out = self.drive(x_train_w_noise, more_out_bool=more_out_bool)
        if more_out_bool:
            r_array, more_out_drive = drive_out
        else:
            r_array = drive_out

        # From r_array get rgen_array:
        # rgen_array = self.r_to_rgen_fct(r_array)
        rgen_array = utilities.vectorize(self.r_to_rgen_fct, (r_array, ))

        # From rgen_array get rproc_array:
        rproc_array = self.set_rgen_to_rproc_fct(rgen_array=rgen_array)

        # from rproc_array get rfit_array.
        rfit_array = self.set_rproc_to_rfit_fct(rproc_array=rproc_array, x_train=x_train)

        # Perform the fitting:
        y_train_fit = self.set_rfit_to_y_fct(rfit_array=rfit_array, y_train=y_train)

        # Get real output from reservoir output:
        xnext_train_fit = utilities.vectorize(self.y_to_xnext_fct, (y_train_fit, ))

        # Get real output from desired resrvoir output.
        xnext_train = utilities.vectorize(self.y_to_xnext_fct, (y_train, ))

        if more_out_bool:
            more_out = {
                "r": r_array,
                "rgen": rgen_array,
                "rproc": rproc_array,
                "rfit": rfit_array,
                "y": y_train_fit}

            more_out = more_out_drive | more_out
            return xnext_train_fit, xnext_train, more_out

        else:
            return xnext_train_fit, xnext_train

    def predict(self, use_for_pred: np.ndarray,
                sync_steps: int = 0,
                pred_steps: int | None = None,
                reset_reservoir_bool: bool = True,
                more_out_bool: bool = False
                ):
        """Predict with the trained reservoir, i.e. let the reservoir run in a loop.

        Args:
            use_for_pred: The data to predict of shape (true_data + sync_steps, self.x_dim).
            sync_steps:  The time steps used to sync the reservoir before prediction.
            pred_steps:  Optional argument to define the number of steps to predict.
            reset_reservoir_bool: Whether to reset the reservoir before sync or not.
            more_out_bool: Return more internal states if true.

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

        # Arrays to save more_out data:
        if more_out_bool:
            more_out = {"xproc": np.zeros((pred_steps, self.xproc_dim)),
                        "rinp": np.zeros((pred_steps, self.r_dim)),
                        "rinternal": np.zeros((pred_steps, self.r_dim)),
                        "r": np.zeros((pred_steps, self.r_dim)),
                        "rgen": np.zeros((pred_steps, self.rgen_dim)),
                        "rproc": np.zeros((pred_steps, self.rproc_dim)),
                        "rfit": np.zeros((pred_steps, self.rfit_dim)),
                        "y": np.zeros((pred_steps, self.y_dim))}

        for i in range(pred_steps):
            xnext = self.res_to_output_step()  # -> get next output from self._last_r
            pred_data[i, :] = xnext
            if more_out_bool:
                more_out["xproc"][i, :] = self._last_xproc
                more_out["rinp"][i, :] = self._last_rinp
                more_out["rinternal"][i, :] = self._last_rinternal
                more_out["r"][i, :] = self._last_r
                more_out["rgen"][i, :] = self._last_rgen
                more_out["rproc"][i, :] = self._last_rproc
                more_out["rfit"][i, :] = self._last_rfit
                more_out["y"][i, :] = self._last_y

            # Take predicted output to drive next reservoir update step:
            self.res_update_step(xnext)  # -> sets self._last_r

        if more_out_bool:
            return pred_data, true_data, more_out
        else:
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
            else:
                rng = np.random
            self.node_bias = node_bias_scale * rng.uniform(low=-1.0,
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

        self.rgen_dim = self._r_to_rgen_fct(np.ones(self.r_dim)).size

class NetworkMixin:
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

        if n_type_opt == "erdos_renyi":
            network = nx.fast_gnp_random_graph(self.r_dim,
                                               self._n_edge_prob,
                                               seed=np.random)
        elif n_type_opt == "scale_free":
            network = nx.barabasi_albert_graph(self.r_dim,
                                               int(self._n_avg_deg / 2),
                                               seed=np.random)
        elif n_type_opt == "small_world":
            network = nx.watts_strogatz_graph(self.r_dim,
                                              k=int(self._n_avg_deg), p=0.1,
                                              seed=np.random)
        elif n_type_opt == "erdos_renyi_directed":
            network = nx.fast_gnp_random_graph(self.r_dim,
                                               self._n_edge_prob,
                                               seed=np.random,
                                               directed=True)
        elif n_type_opt == "random_dense":
            network = nx.from_numpy_matrix(np.ones((self.r_dim, self.r_dim)))

        else:
            raise ValueError(f"n_type_opt {n_type_opt} not recognized! "
                             f"Must be erdos_renyi, scale_free, small_world, "
                             f"erdos_renyi_directed or random_dense.")
        self._network = nx.to_numpy_array(network)

    def _vary_network(self,
                      network_variation_attempts: int = 10) -> None:
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

        with utilities.temp_seed(network_creation_seed):
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

class OutputFitMixin:
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

class NoRgenToRprocMixin:
    """Very simple rgen_to_rproc_fct: No processing of rgen states."""
    def __init__(self):
        pass

    def set_rgen_to_rproc_fct(self, rgen_array: np.ndarray) -> np.ndarray:
        """No processing of rgen states."""
        return rgen_array

    def rgen_to_rproc_fct(self, rgen: np.ndarray) -> np.ndarray:
        """No processing of rgen states."""
        return rgen

    def subbuild(self):
        """subbuild NoRgenToRproc: """
        self.rproc_dim = self.rgen_dim

class RgenToRprocMixin:
    """Possibility to process Rgen to Rproc. """
    def __init__(self):
        # Scale(!) and center Rgen states:
        self._scale_rgen_bool: bool | None = None  # Whether to rescale rgen states
        self.rgen_standard_scaler: None | StandardScaler = None

        # PCA transform (scaled and centered) Rgen states:
        self._perform_pca_bool: bool | None = None
        self._n_pca_components: int | None = None
        self._pca_matrix: np.ndarray | None = None
        self._pca_rgen_mean: np.ndarray | None = None
        self.pca_object: None | PCA = None

        # Function:
        self._rgen_to_rproc_fct: Callable[[np.ndarray], np.ndarray] | None = None

    def set_rgen_to_rproc_fct(self, rgen_array: np.ndarray) -> np.ndarray:
        """No processing of rgen states."""
        if self._scale_rgen_bool:
            self.rgen_standard_scaler = StandardScaler()
            rgen_array = self.rgen_standard_scaler.fit_transform(rgen_array)


        if self._perform_pca_bool:
            self.pca_object = PCA(n_components=self._n_pca_components)
            self._pca_rgen_mean = np.mean(rgen_array, axis=0)
            rgen_array = self.pca_object.fit_transform(rgen_array)
            self._pca_matrix = self.pca_object.components_

        if self._scale_rgen_bool:
           _rgen_to_rproc_fct_scaler = \
                lambda rgen: self.rgen_standard_scaler.transform(rgen[np.newaxis, :])[0, :]

        if self._perform_pca_bool:
            _rgen_to_rproc_fct_pca = \
                lambda rgen: self._pca_matrix @ (rgen - self._pca_rgen_mean)

        if self._scale_rgen_bool and self._perform_pca_bool:  # Scale+Center and then PCA
            self._rgen_to_rproc_fct = \
                lambda rgen: _rgen_to_rproc_fct_pca(_rgen_to_rproc_fct_scaler(rgen))

        elif self._scale_rgen_bool and not(self._perform_pca_bool): # Only Scale+Center
            self._rgen_to_rproc_fct = lambda rgen: _rgen_to_rproc_fct_scaler(rgen)

        elif not(self._scale_rgen_bool) and self._perform_pca_bool:  # Only PCA
            self._rgen_to_rproc_fct = lambda rgen: _rgen_to_rproc_fct_pca(rgen)

        else: # No processing of rgen
            self._rgen_to_rproc_fct = lambda rgen: rgen

        return rgen_array

    def rgen_to_rproc_fct(self, rgen: np.ndarray) -> np.ndarray:
        """Some processing of rgen states."""
        return self._rgen_to_rproc_fct(rgen)

    def subbuild(self,
                 scale_rgen_bool: bool = False,
                 perform_pca_bool: bool = False,
                 pca_components: int | None = None
                 ):

        self._scale_rgen_bool = scale_rgen_bool
        self._perform_pca_bool = perform_pca_bool
        self._n_pca_components = pca_components

        if self._perform_pca_bool:
            if self._n_pca_components is None:
                self.rproc_dim = self.rgen_dim

            else:
                self.rproc_dim = self._n_pca_components
        else:
            self.rproc_dim = self.rgen_dim


class NoRprocToRfitMixin:
    """Very simple rproc_to_rfit_fct: No modification of rproc."""
    def __init__(self):
        pass

    def rproc_to_rfit_fct(self, rproc: np.ndarray,
                          x: np.ndarray | None = None) -> np.ndarray:
        """No modification of rproc"""
        return rproc

    def set_rproc_to_rfit_fct(self, rproc_array: np.ndarray, x_train: np.ndarray) -> np.ndarray:
        """No modification of rproc"""
        return rproc_array

    def subbuild(self):
        """Sub-buid Rproc to Rfit. """
        self.rfit_dim = self.rproc_dim

class NoXToXProcMixin:
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


class ScalerXToXProcMixin:
    """Scale and center input x with sklearns StandardScaler, (i.e. no mean and unit-variance). """
    def __init__(self):

        self._scale_input_bool: bool | None = None
        self.standard_scaler: None | StandardScaler = None
        self._x_to_xproc_fct: Callable[[np.ndarray], np.ndarray] | None = None

    def x_to_xproc_fct(self, x: np.ndarray) -> np.ndarray:
        """Scale and shift"""
        return self._x_to_xproc_fct(x)

    def set_x_to_xproc_fct(self, train: np.ndarray):
        """Scale and shift"""

        if self._scale_input_bool:
            self.standard_scaler = StandardScaler()
            self.standard_scaler.fit(train)
            self._x_to_xproc_fct = lambda x: \
                self.standard_scaler.transform(x[np.newaxis, :])[0, :]

        else:
            self._x_to_xproc_fct = lambda x: x

    def subbuild(self, scale_input_bool: bool = False):
        """Subbuild for standard scaling x_to_xproc_fct."""
        self.xproc_dim = self.x_dim
        self._scale_input_bool = scale_input_bool


class InputMatrixMixin:
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

class NoYToXnextMixin:
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
        self.y_dim = self.x_dim

class ScalerYToXnextMixin:
    """Scale fit also on scaled output. """
    def __init__(self):
        self._scale_output_bool: bool | None = None
        self._y_to_xnext_fct: Callable[[np.ndarray], np.ndarray] | None = None


    def y_to_xnext_fct(self, y: np.ndarray,
                       x: np.ndarray | None = None) -> np.ndarray:
        """Either identity or scaler. """
        return self._y_to_xnext_fct(y, x)

    def set_y_to_xnext_fct(self, train: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Either add the output standard scaler or not. """

        x_train = train[: -1, :]
        y_train = train[1:, :]

        if self._scale_output_bool:
            # Check if there is already a standard scaler defined from XToXProcMixin.
            if not hasattr(self, "standard_scaler"):
                self.standard_scaler = StandardScaler()
                self.standard_scaler.fit(train)
            else:
                if self.standard_scaler is None:
                    self.standard_scaler = StandardScaler()
                    self.standard_scaler.fit(train)

            # Transform the output to be fitted:
            y_train = self.standard_scaler.transform(y_train)

            # Define the y_to_xnext_fct:
            self._y_to_xnext_fct = lambda y, x: \
                self.standard_scaler.inverse_transform(y[np.newaxis, :])[0, :]
        else:
            # Or just the identity:
            self._y_to_xnext_fct = lambda y, x: y

        return x_train, y_train

    def subbuild(self, scale_output_bool: bool = False):
        """Sub-build the Simple Y to next X Mixin. """
        self._scale_output_bool = scale_output_bool
        self.y_dim = self.x_dim

# Special Mixins:
class HybridMixin:
    """Special Mixin to support Hybrid RC, combining ESN with knowledge-based model:

    Related to:
    Pathak, J., Wikner, A., Fussell, R., Chandra, S., Hunt, B., Girvan, M., & Ott, E.
    (2018). Hybrid Forecasting of Chaotic Processes: Using Machine Learning in Conjunction with a
    Knowledge-Based Model. Chaos, 28(4). https://doi.org/10.1063/1.5028373
    """

    def __init__(self):
        # Hybrid Model:
        # input
        self.input_model: Callable[[np.ndarray], np.ndarray] | None = None
        self._scale_input_model_bool: bool | None = None
        self.standard_scaler_input_model: None | StandardScaler = None

        # output
        self.output_model: Callable[[np.ndarray], np.ndarray] | None = None
        self._scale_output_model_bool: Callable[[np.ndarray], np.ndarray] | None = None
        self.standard_scaler_output_model: None | StandardScaler = None

        # both
        self._out_model_is_inp_bool: bool | None = None
        self._last_input_model_result: np.ndarray | None = None

        # Scaler XToXProc:
        self._scale_input_bool: bool | None = None
        self.standard_scaler: None | StandardScaler = None
        self._x_to_xproc_fct: Callable[[np.ndarray], np.ndarray] | None = None

        # Rproc to Rfit:
        self._rproc_to_rfit_fct: Callable[[np.ndarray], np.ndarray] | None = None


    def x_to_xproc_fct(self, x: np.ndarray) -> np.ndarray:
        """Scale and shift + input model."""
        return self._x_to_xproc_fct(x)

    def set_x_to_xproc_fct(self, train: np.ndarray):
        """Scale and shift + input model."""

        if self.input_model is not None:
            def input_model(x):
                # to be able to save the model evaluation.
                self._last_input_model_result = self.input_model(x)
                return self._last_input_model_result

            if self._scale_input_model_bool:
                # create the input model scaler, i.e. scale the results from the input_model.
                self.standard_scaler_input_model = StandardScaler()
                input_model_array = utilities.vectorize(self.input_model, (train, ))
                self.standard_scaler_input_model.fit(input_model_array)
                _inp_model_scaler = \
                    lambda x: self.standard_scaler_input_model.transform(x[np.newaxis, :])[0, :]
            else: # no input model scaling:
                _inp_model_scaler = lambda x: x

        if self._scale_input_bool:
            self.standard_scaler = StandardScaler()
            self.standard_scaler.fit(train)

            _inp_scaler = \
                lambda x: self.standard_scaler.transform(x[np.newaxis, :])[0, :]
        else:  # no input scaling:
            _inp_scaler = lambda x: x


        if self.input_model is not None:
            self._x_to_xproc_fct = \
                lambda x: np.hstack((_inp_scaler(x), _inp_model_scaler(input_model(x))))

        else:
            self._x_to_xproc_fct = lambda x: _inp_scaler(x)

    def rproc_to_rfit_fct(self,
                          rproc: np.ndarray,
                          x: np.ndarray | None = None) -> np.ndarray:
        """Output model:
        """
        if self.output_model is None:
            return rproc

        else:
            if self._out_model_is_inp_bool:
                model_result = self._last_input_model_result
            else:
                model_result = self.output_model(x)

            if self._scale_output_model_bool:
                model_result = \
                    self.standard_scaler_output_model.transform(model_result[np.newaxis, :])[0, :]

        return np.hstack((rproc, model_result))

    def set_rproc_to_rfit_fct(self,
                              rproc_array: np.ndarray,
                              x_train: np.ndarray) -> np.ndarray:
        """Set rproc_to_rfit fct. """
        if self.output_model is not None:
            output_model_array = utilities.vectorize(self.output_model, (x_train,))
            if self._scale_output_model_bool:
                self.standard_scaler_output_model = StandardScaler()
                # Set the scaler and scale the output_model_array
                output_model_array = \
                    self.standard_scaler_output_model.fit_transform(output_model_array)

            return np.concatenate((rproc_array, output_model_array), axis=1)

        else:
            return rproc_array

    def subbuild(self,
                 scale_input_bool: bool = False,
                 scale_input_model_bool: bool = False,
                 input_model: Callable[[np.ndarray], np.ndarray] | None = None,
                 scale_output_model_bool: bool = False,
                 output_model: Callable[[np.ndarray], np.ndarray] | None = None,
                 out_model_is_inp_bool: bool = False,
                 ):

        """Hybrid ESN sub-build. """
        self._scale_input_bool = scale_input_bool
        self._scale_input_model_bool = scale_input_model_bool
        self._scale_output_model_bool = scale_output_model_bool

        self._out_model_is_inp_bool = out_model_is_inp_bool

        self.input_model = input_model
        if self._out_model_is_inp_bool:
            self.output_model = input_model
        else:
            self.output_model = output_model

        if self.input_model is None:
            self.xproc_dim = self.x_dim
        else:
            self.xproc_dim = self.x_dim + self.input_model(np.ones(self.x_dim)).size

        if self.output_model is None:
            self.rfit_dim = self.rproc_dim
        else:
            self.rfit_dim = self.rproc_dim + self.output_model(np.ones(self.x_dim)).size


class ESNSimple(
    ActFunctionMixin,
    NetworkMixin,
    OutputFitMixin,
    RToRgenMixin,
    NoRgenToRprocMixin,
    NoRprocToRfitMixin,
    NoXToXProcMixin,
    InputMatrixMixin,
    NoYToXnextMixin,
    ResCompCore):

    def __init__(self):
        ResCompCore.__init__(self)
        ActFunctionMixin.__init__(self)
        NetworkMixin.__init__(self)
        OutputFitMixin.__init__(self)
        NoRgenToRprocMixin.__init__(self)
        NoRprocToRfitMixin.__init__(self)
        NoXToXProcMixin.__init__(self)
        InputMatrixMixin.__init__(self)
        NoYToXnextMixin.__init__(self)
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

        # Simple X to XProc:
        NoXToXProcMixin.subbuild(self)

        # R to Rgen build:
        RToRgenMixin.subbuild(
            self, r_to_rgen_opt=r_to_rgen_opt)

        # Rgen to Rproc:
        NoRgenToRprocMixin.subbuild(self)

        # Rproc to Rfit:
        NoRprocToRfitMixin.subbuild(self)

        # Y to Xnext:
        NoYToXnextMixin.subbuild(self)

        # Input Matrix build:
        InputMatrixMixin.subbuild(
            self,
            w_in_opt=w_in_opt,
            w_in_scale=w_in_scale,
            w_in_seed=w_in_seed
            )

        # Reservoir Network build:
        NetworkMixin.subbuild(
            self,
            n_type_opt=n_type_opt,
            n_rad=n_rad,
            n_avg_deg=n_avg_deg,
            network_creation_attempts=network_creation_attempts,
            network_creation_seed=network_creation_seed,
        )

        # Activation Function build:
        ActFunctionMixin.subbuild(
            self,
            act_fct_opt=act_fct_opt)

        # Res output fit:
        OutputFitMixin.subbuild(
            self,
            reg_param=reg_param,
            ridge_regression_opt=ridge_regression_opt)

class ESN(
    ActFunctionMixin,
    NetworkMixin,
    OutputFitMixin,
    RToRgenMixin,
    RgenToRprocMixin,
    NoRprocToRfitMixin,
    ScalerXToXProcMixin,
    InputMatrixMixin,
    ScalerYToXnextMixin,
    ResCompCore):

    def __init__(self):
        ResCompCore.__init__(self)
        ActFunctionMixin.__init__(self)
        NetworkMixin.__init__(self)
        OutputFitMixin.__init__(self)
        RgenToRprocMixin.__init__(self)
        NoRprocToRfitMixin.__init__(self)
        ScalerXToXProcMixin.__init__(self)
        InputMatrixMixin.__init__(self)
        ScalerYToXnextMixin.__init__(self)
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

              # X to Xproc Standard Scaler:
              scale_input_bool: bool = False,

              # Input Matrix:
              w_in_opt: str = "random_sparse",
              w_in_scale: float = 1.0,
              w_in_seed: int | None = None,

              # Rgen to Rproc:
              scale_rgen_bool: bool = False,
              perform_pca_bool: bool = False,
              pca_components: int | None = None,

              # Y to Xnext Standard Scaler:
              scale_output_bool: bool = False,
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

        # Scaler X to XProc:
        ScalerXToXProcMixin.subbuild(
            self,
            scale_input_bool=scale_input_bool)

        # R to Rgen build:
        RToRgenMixin.subbuild(
            self, r_to_rgen_opt=r_to_rgen_opt)

        # Rgen to Rproc:
        RgenToRprocMixin.subbuild(
            self,
            scale_rgen_bool=scale_rgen_bool,
            perform_pca_bool=perform_pca_bool,
            pca_components=pca_components,
        )

        # Rproc to Rfit:
        NoRprocToRfitMixin.subbuild(self)

        # Y to Xnext:
        ScalerYToXnextMixin.subbuild(
            self,
            scale_output_bool=scale_output_bool)

        # Input Matrix build:
        InputMatrixMixin.subbuild(
            self,
            w_in_opt=w_in_opt,
            w_in_scale=w_in_scale,
            w_in_seed=w_in_seed
            )

        # Reservoir Network build:
        NetworkMixin.subbuild(
            self,
            n_type_opt=n_type_opt,
            n_rad=n_rad,
            n_avg_deg=n_avg_deg,
            network_creation_attempts=network_creation_attempts,
            network_creation_seed=network_creation_seed,
        )

        # Activation Function build:
        ActFunctionMixin.subbuild(
            self,
            act_fct_opt=act_fct_opt)

        # Res output fit:
        OutputFitMixin.subbuild(
            self,
            reg_param=reg_param,
            ridge_regression_opt=ridge_regression_opt)

class ESNHybrid(
    ActFunctionMixin,
    NetworkMixin,
    OutputFitMixin,
    RToRgenMixin,
    RgenToRprocMixin,
    HybridMixin,
    InputMatrixMixin,
    ScalerYToXnextMixin,
    ResCompCore):

    def __init__(self):
        ResCompCore.__init__(self)
        ActFunctionMixin.__init__(self)
        NetworkMixin.__init__(self)
        OutputFitMixin.__init__(self)
        RgenToRprocMixin.__init__(self)
        HybridMixin.__init__(self)
        InputMatrixMixin.__init__(self)
        ScalerYToXnextMixin.__init__(self)
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

              # Hybrid RC:
              scale_input_bool: bool = False,
              scale_input_model_bool: bool = False,
              input_model: Callable[[np.ndarray], np.ndarray] | None = None,
              scale_output_model_bool: bool = False,
              output_model: Callable[[np.ndarray], np.ndarray] | None = None,
              out_model_is_inp_bool: bool = False,

              # Input Matrix:
              w_in_opt: str = "random_sparse",
              w_in_scale: float = 1.0,
              w_in_seed: int | None = None,

              # Rgen to Rproc:
              scale_rgen_bool: bool = False,
              perform_pca_bool: bool = False,
              pca_components: int | None = None,

              # Y to Xnext Standard Scaler:
              scale_output_bool: bool = False,
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

        # R to Rgen build:
        RToRgenMixin.subbuild(
            self, r_to_rgen_opt=r_to_rgen_opt)

        # Rgen to Rproc:
        RgenToRprocMixin.subbuild(
            self,
            scale_rgen_bool=scale_rgen_bool,
            perform_pca_bool=perform_pca_bool,
            pca_components=pca_components,
        )

        # Hybrid Mixin:
        HybridMixin.subbuild(self,
            scale_input_bool=scale_input_bool,
            scale_input_model_bool=scale_input_model_bool,
            input_model=input_model,
            scale_output_model_bool=scale_output_model_bool,
            output_model=output_model,
            out_model_is_inp_bool=out_model_is_inp_bool
        )

        # Y to Xnext:
        ScalerYToXnextMixin.subbuild(
            self,
            scale_output_bool=scale_output_bool)

        # Input Matrix build:
        InputMatrixMixin.subbuild(
            self,
            w_in_opt=w_in_opt,
            w_in_scale=w_in_scale,
            w_in_seed=w_in_seed
            )

        # Reservoir Network build:
        NetworkMixin.subbuild(
            self,
            n_type_opt=n_type_opt,
            n_rad=n_rad,
            n_avg_deg=n_avg_deg,
            network_creation_attempts=network_creation_attempts,
            network_creation_seed=network_creation_seed,
        )

        # Activation Function build:
        ActFunctionMixin.subbuild(
            self,
            act_fct_opt=act_fct_opt)

        # Res output fit:
        OutputFitMixin.subbuild(
            self,
            reg_param=reg_param,
            ridge_regression_opt=ridge_regression_opt)

ESN_DICT = {"ESN": ESN,
            "ESNHybrid": ESNHybrid}
