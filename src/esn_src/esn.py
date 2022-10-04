# -*- coding: utf-8 -*-
""" Implements the Echo State Network (ESN) used in Reservoir Computing """
from __future__ import annotations

from typing import Callable

import numpy as np
from scipy import sparse
from scipy.sparse import linalg
from scipy.sparse.linalg.eigen.arpack.arpack \
    import ArpackNoConvergence as _ArpackNoConvergence
import networkx as nx
from sklearn import decomposition

from src.esn_src import utilities

class _ResCompCore():
    """
    Reservoir Computing base class.
    """

    def __init__(self):

        super(_ResCompCore, self).__init__()

        self._r_dim = None
        self._r_gen_dim = None
        self._x_dim = None
        self._y_dim = None

        self._w_out = None

        self._y_to_x_fct = None  # the function to map the output to the input (e.g. when training on difference)
        self._act_fct = None
        self._res_internal_update_fct = None
        self._inp_coupling_fct = None
        self._r_to_r_gen_fct = None

        self._leak_factor = None
        self._node_bias = None

        self._last_x = None
        self._last_res_inp = None
        self._last_r_interal = None
        self._last_r = None
        self._last_r_gen = None
        self._last_y = None

        self._saved_res_inp = None
        self._saved_r_internal = None
        self._saved_r = None
        self._saved_r_gen = None
        self._saved_out = None
        self._saved_y_train = None

        self._reg_param = None

        self._default_r = None

    def _r_gen_to_out_fct(self, r_gen: np.ndarray) -> np.ndarray:
        return self._w_out @ r_gen

    def _res_update(self, x: np.ndarray) -> None:
        """Update the reservoir from r_i to r_(i+1), save the resulting reservoir state.

        Args:
            x: The input of shape (sys_dim, ).
        """

        self._last_x = x
        self._last_res_inp = self._inp_coupling_fct(self._last_x)
        self._last_r_interal = self._res_internal_update_fct(self._last_r)

        self._last_r = self._leak_factor * self._last_r + (1 - self._leak_factor) * self._act_fct(
            self._last_res_inp +
            self._last_r_interal + self._node_bias)

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

    def train_synced(self, x_train, y_train, save_y_train=False, **kwargs) -> None:
        """Train the synced reservoir.

        Drive the reservoir with x_train, get the r_gen states corresponding to x_train,
        and get _w_out by fitting the r_gen states to y_train.

        Args:
            x_train: The input array of shape (train_steps - 1, sys_dim).
            y_train: The output array of shape (train_steps - 1, sys_dim).
            save_y_train: If true, save the fitted y_train.
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

        if save_y_train:
            self._saved_y_train = y_train

        if save_out:
            self._saved_out = (self._w_out @ self._saved_r_gen.T).T

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

    def train(self, x_sync, x_train, y_train, reset_res_state=True, **kwargs):
        if reset_res_state:
            self.reset_r()
        self.drive(x_sync)
        self.train_synced(x_train, y_train, **kwargs)

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


class _add_basic_defaults():
    """
    add activation function options, node bias options, leak factor, default_res_state,
    reg_param

    SETS:
        - self._act_fct: r -> r
        - self._node_bias: np.ndarray (shape: r)
        - self._leak_factor: float between 0 and 1
        - self._reg_param: positive float
    """

    def __init__(self):

        self._act_fct_opt = None
        self._act_fct_flag_synonyms = utilities._SynonymDict()
        self._act_fct_flag_synonyms.add_synonyms(0, ["tanh", "tanh_simple", "simple"])
        self._act_fct_flag_synonyms.add_synonyms(1, ["sigmoid"])
        self._act_fct_flag_synonyms.add_synonyms(2, ["relu"])
        self._act_fct_flag_synonyms.add_synonyms(3, ["identity", "linear"])

        self._node_bias_opt = None
        self._node_bias_flag_synonyms = utilities._SynonymDict()
        self._node_bias_flag_synonyms.add_synonyms(0, ["no_bias"])
        self._node_bias_flag_synonyms.add_synonyms(1, ["random_bias"])
        self._node_bias_flag_synonyms.add_synonyms(2, ["constant_bias"])
        self._bias_scale = None

    def set_activation_function(self, act_fct_opt="tanh"):
        if type(act_fct_opt) == str:
            self._act_fct_opt = act_fct_opt
            act_fct_flag = self._act_fct_flag_synonyms.get_flag(act_fct_opt)
            if act_fct_flag == 0:
                self._act_fct = np.tanh
            elif act_fct_flag == 1:
                self._act_fct = utilities.sigmoid
            elif act_fct_flag == 2:
                self._act_fct = utilities.relu
            elif act_fct_flag == 3:
                self._act_fct = lambda x: x
        else:
            self._act_fct_opt = "CUSTOM"
            self._act_fct = act_fct_opt

    def set_node_bias(self, node_bias_opt="no_bias", bias_scale=1.0):
        if type(node_bias_opt) == str:
            self._node_bias_opt = node_bias_opt
            node_bias_flag = self._node_bias_flag_synonyms.get_flag(node_bias_opt)
            if node_bias_flag == 0:
                self._node_bias = 0
            elif node_bias_flag == 1:
                self._bias_scale = bias_scale
                self._node_bias = self._bias_scale * np.random.uniform(low=-1.0, high=1.0, size=self._r_dim)
            elif node_bias_flag == 2:
                self._bias_scale = bias_scale
                self._node_bias = self._bias_scale * np.ones(self._r_dim)
        else:
            self._node_bias_opt = "CUSTOM"
            self._node_bias = node_bias_opt

    def set_leak_factor(self, leak_factor=0.0):
        self._leak_factor = leak_factor

    def set_default_res_state(self, default_res_state=None):
        if default_res_state is None:
            self._default_r = np.zeros(self._r_dim)
        else:
            self._default_r = default_res_state

    def set_reg_param(self, reg_param=1e-8):
        self._reg_param = reg_param


class _add_network_update_fct():
    """
    add network as internal res update function

    SETS:
        - self._res_internal_update_fct r -> r
    """
    def __init__(self):
        self._network = None
        self._res_internal_update_fct = lambda r: self._network @ r

        self._n_type_opt = None
        self._n_rad = None
        self._n_avg_deg = None
        self._n_edge_prob = None
        self._n_type_flag_synonyms = utilities._SynonymDict()
        self._n_type_flag_synonyms.add_synonyms(0, ["random", "erdos_renyi"])
        self._n_type_flag_synonyms.add_synonyms(1, ["scale_free", "barabasi_albert"])
        self._n_type_flag_synonyms.add_synonyms(2, ["small_world", "watts_strogatz"])
        self._n_type_flag_synonyms.add_synonyms(3, ["random_directed", "erdos_renyi_directed"])
        self._n_type_flag_synonyms.add_synonyms(4, ["random_dense"])
        self._n_type_flag_synonyms.add_synonyms(5, ["scipy_sparse"])

    def _create_scipy_sparse(self):
        # see https://github.com/pvlachas/RNN-RC-Chaos/blob/a403e0e843cf9dde11833f0206f94f91169a4661/Methods/Models/esn/esn.py#L100
        density = self._n_avg_deg/self._r_dim

        # sparse scipy matrix. It might have non-zero diagonal elements, the random values are uniformly distributed between 0 and 1.
        self._network = sparse.random(self._r_dim, self._r_dim, density=density).toarray()
        self._network = 2*self._network - 1

        self._scale_network()


    def create_network(self, n_rad=0.1, n_avg_deg=6.0,
                       n_type_opt="erdos_renyi", network_creation_attempts=10):
        if type(n_type_opt) == str:
            self._n_type_opt = n_type_opt
            self._n_rad = n_rad
            self._n_avg_deg = n_avg_deg
            self._n_edge_prob = self._n_avg_deg / (self._r_dim - 1)
            self._n_type_opt = n_type_opt
            n_type_flag = self._n_type_flag_synonyms.get_flag(n_type_opt)
            if n_type_flag == 5:  # scipy sparse
                self._create_scipy_sparse()
            else:
                for i in range(network_creation_attempts):
                    try:
                        self._create_network_connections(n_type_flag)
                        self._vary_network()
                    except _ArpackNoConvergence:
                        continue
                    break
                else:
                    raise Exception("Network creation during ESN init failed %d times"
                                    % network_creation_attempts)
        else:
            self._n_type_opt = "CUSTOM"
            self._network = n_type_opt

    def _create_network_connections(self, n_type_flag):
        """ Generate the baseline random network to be scaled

        Specification done via protected members
        """

        if n_type_flag == 0:
            network = nx.fast_gnp_random_graph(self._r_dim, self._n_edge_prob,
                                               seed=np.random)
        elif n_type_flag == 1:
            network = nx.barabasi_albert_graph(self._r_dim,
                                               int(self._n_avg_deg / 2),
                                               seed=np.random)
        elif n_type_flag == 2:
            network = nx.watts_strogatz_graph(self._r_dim,
                                              k=int(self._n_avg_deg), p=0.1,
                                              seed=np.random)
        elif n_type_flag == 3:
            network = nx.fast_gnp_random_graph(self._r_dim, self._n_edge_prob,
                                               seed=np.random, directed=True)
        elif n_type_flag == 4:
            # network = nx.from_numpy_matrix(np.random.randn(self._r_dim, self._r_dim))
            network = nx.from_numpy_matrix(np.ones((self._r_dim, self._r_dim)))
        else:
            raise Exception("the network type %s is not implemented" %
                            str(self._n_type_opt))
        self._network = nx.to_numpy_array(network)

    def _vary_network(self, network_variation_attempts=10):
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
                # self.logger.error(
                #     'Network Variaion failed! -> Try agin!')
                continue
            break
        else:
            # TODO: Better logging of exceptions
            # self.logger.error("Network variation failed %d times"
            #                   % network_variation_attempts)
            raise _ArpackNoConvergence

    def _scale_network(self):
        """ Scale self._network, according to desired spectral radius.

        Can cause problems due to non converging of the eigenvalue evaluation

        Specification done via protected members

        """
        self._network = sparse.csr_matrix(self._network)
        try:
            eigenvals = linalg.eigs(
                self._network, k=1, v0=np.ones(self._r_dim),
                maxiter=1e3 * self._r_dim)[0]
        except _ArpackNoConvergence:
            # self.logger.error('Eigenvalue calculation in scale_network failed!')
            raise

        maximum = np.absolute(eigenvals).max()
        self._network = ((self._n_rad / maximum) * self._network)

    def return_network(self):
        return self._network.toarray()

    def return_avg_deg(self):
        return


class _add_no_res_update_fct():
    """
    Use a Dynamical System as res_internal_update_fct
    SETS:
        - self._res_internal_update_fct r -> r
    """

    def __init__(self):
        self._res_internal_update_fct = lambda r: np.zeros(r.shape)


class _add_basic_r_to_rgen():
    """
    add basic r to r_gen behaviour
    SETS:
    - self._r_to_r_gen_fct: r -> r_gen (any dimension)
    """

    def __init__(self):

        self._r_to_r_gen_opt = None
        self._r_to_r_gen_synonyms = utilities._SynonymDict()
        self._r_to_r_gen_synonyms.add_synonyms(0, ["linear_r", "simple", "linear"])
        self._r_to_r_gen_synonyms.add_synonyms(1, "linear_and_square_r")
        self._r_to_r_gen_synonyms.add_synonyms(2, ["output_bias", "bias"])
        self._r_to_r_gen_synonyms.add_synonyms(3, ["bias_and_square_r"])
        self._r_to_r_gen_synonyms.add_synonyms(4, ["linear_and_square_r_alt"])
        self._r_to_r_gen_synonyms.add_synonyms(5, ["exponential_r"])
        self._r_to_r_gen_synonyms.add_synonyms(6, ["bias_and_exponential_r"])

    def set_r_to_r_gen_fct(self, r_to_r_gen_opt="linear"):
        if type(r_to_r_gen_opt) == str:
            self._r_to_r_gen_opt = r_to_r_gen_opt
            r_to_r_gen_flag = self._r_to_r_gen_synonyms.get_flag(r_to_r_gen_opt)
            if r_to_r_gen_flag == 0:
                self._r_to_r_gen_fct = lambda r, x: r
            elif r_to_r_gen_flag == 1:
                self._r_to_r_gen_fct = lambda r, x: np.hstack((r, r ** 2))
            elif r_to_r_gen_flag == 2:
                self._r_to_r_gen_fct = lambda r, x: np.hstack((r, 1))
            elif r_to_r_gen_flag == 3:
                self._r_to_r_gen_fct = lambda r, x: np.hstack((np.hstack((r, r ** 2)), 1))
            elif r_to_r_gen_flag == 4:
                def temp(r, x):
                    r_gen = np.copy(r).T
                    r_gen[::2] = r.T[::2] ** 2
                    return r_gen.T

                self._r_to_r_gen_fct = temp
            elif r_to_r_gen_flag == 5:
                self._r_to_r_gen_fct = lambda r, x: np.hstack((r, np.exp(r)))
            elif r_to_r_gen_flag == 6:
                self._r_to_r_gen_fct = lambda r, x: np.hstack((np.hstack((r, np.exp(r))), 1))
        else:
            self._r_to_r_gen_opt = "CUSTOM"
            self._r_to_r_gen_fct = r_to_r_gen_opt

        self.set_r_to_r_gen_dim()

    def set_r_to_r_gen_dim(self):
        self._r_gen_dim = self._r_to_r_gen_fct(np.zeros(self._r_dim), None).shape[0]


class _add_preprocess_r_to_rgen():
    """
    1. Scaler and/or center r_train.
        - specify if you want to scale
        - specify if you want to center
    2. Add a pca transformation.
        - specify the number of components
        - specify if you want to center
    """

    def __init__(self):

        self._r_to_r_gen_opt = None
        self._r_to_r_gen_synonyms = utilities._SynonymDict()
        self._r_to_r_gen_synonyms.add_synonyms(0, ["linear_r", "simple", "linear"])
        self._r_to_r_gen_synonyms.add_synonyms(1, "linear_and_square_r")
        self._r_to_r_gen_synonyms.add_synonyms(2, ["output_bias", "bias"])
        self._r_to_r_gen_synonyms.add_synonyms(3, ["bias_and_square_r"])
        self._r_to_r_gen_synonyms.add_synonyms(4, ["linear_and_square_r_alt"])
        self._r_to_r_gen_synonyms.add_synonyms(5, ["exponential_r"])
        self._r_to_r_gen_synonyms.add_synonyms(6, ["bias_and_exponential_r"])

        self._r_train_std = None
        self._r_train_mean = None

        self._pca = None
        self._pca_matrix = None
        self._pca_input_mean = None

    def set_r_train_preprocessing(self,
                                  center_r_train: bool = False,
                                  rescale_r_train: bool = False,
                                  perform_pca: bool = False,
                                  pca_components: int | None = None,
                                  center_pca_input: bool = True,
                                  ):
        self.center_r_train = center_r_train
        self.rescale_r_train = rescale_r_train
        self.perform_pca = perform_pca
        self.pca_components = pca_components
        self.center_pca_input = center_pca_input

    def set_r_to_r_gen_fct(self,
                           r_train: np.ndarray,
                           r_to_r_gen_opt="linear"
                           ):
        if type(r_to_r_gen_opt) == str:
            self._r_to_r_gen_opt = r_to_r_gen_opt
            r_to_r_gen_flag = self._r_to_r_gen_synonyms.get_flag(r_to_r_gen_opt)
            if r_to_r_gen_flag == 0:
                basic_rrgen_fct = lambda r, x: r
            elif r_to_r_gen_flag == 1:
                basic_rrgen_fct = lambda r, x: np.hstack((r, r ** 2))
            elif r_to_r_gen_flag == 2:
                basic_rrgen_fct = lambda r, x: np.hstack((r, 1))
            elif r_to_r_gen_flag == 3:
                basic_rrgen_fct = lambda r, x: np.hstack((np.hstack((r, r ** 2)), 1))
            elif r_to_r_gen_flag == 4:
                def temp(r, x):
                    r_gen = np.copy(r).T
                    r_gen[::2] = r.T[::2] ** 2
                    return r_gen.T

                basic_rrgen_fct = temp
            elif r_to_r_gen_flag == 5:
                basic_rrgen_fct = lambda r, x: np.hstack((r, np.exp(r)))
            elif r_to_r_gen_flag == 6:
                basic_rrgen_fct = lambda r, x: np.hstack((np.hstack((r, np.exp(r))), 1))
        else:
            self._r_to_r_gen_opt = "CUSTOM"
            basic_rrgen_fct = r_to_r_gen_opt

        if self.center_r_train:
            self._r_train_mean = np.mean(r_train, axis=0)
            r_train = r_train - self._r_train_mean
        else:
            self._r_train_mean = 0
        if self.rescale_r_train:
            self._r_train_scale = np.std(r_train, axis=0)
            r_train = r_train/self._r_train_scale
        else:
            self._r_train_scale = 1
        if self.perform_pca:
            self._pca = decomposition.PCA(n_components=self.pca_components)
            self._pca.fit(r_train)
            self._pca_matrix = self._pca.components_
            if self.center_pca_input:
                self._pca_input_mean = np.mean(r_train, axis=0)
            else:
                self._pca_input_mean = 0

            preprocess_fct = lambda r: self._pca_matrix @ (
                        ((r - self._r_train_mean) / self._r_train_scale) - self._pca_input_mean)
        else:
            preprocess_fct = lambda r: (r - self._r_train_mean) / self._r_train_scale

        self._r_to_r_gen_fct = lambda r, x: basic_rrgen_fct(preprocess_fct(r), x)

        self._r_gen_dim = self._r_to_r_gen_fct(np.zeros(self._r_dim), np.zeros(self._x_dim)).shape[
            0]


class _add_model_r_to_rgen():
    """
    add r to r_gen behaviour with hybrid-output model
    SETS:
    - self._r_to_r_gen_fct: r -> r_gen (any dimension)
    """

    def __init__(self):

        self._r_to_r_gen_opt = None
        self._r_to_r_gen_synonyms = utilities._SynonymDict()
        self._r_to_r_gen_synonyms.add_synonyms(0, ["linear_r", "simple", "linear"])
        self._r_to_r_gen_synonyms.add_synonyms(1, "linear_and_square_r")
        self._r_to_r_gen_synonyms.add_synonyms(2, ["output_bias", "bias"])
        self._r_to_r_gen_synonyms.add_synonyms(3, ["bias_and_square_r"])
        self._r_to_r_gen_synonyms.add_synonyms(4, ["linear_and_square_r_alt"])
        self._r_to_r_gen_synonyms.add_synonyms(5, ["exponential_r"])
        self._r_to_r_gen_synonyms.add_synonyms(6, ["bias_and_exponential_r"])

        self.output_model = None
        self.scale_shift_vector_output = None

    def set_output_model(self,
                         output_model: Callable[[np.ndarray], np.ndarray] | None = None,
                         scale_shift_vector_output: tuple[np.ndarray, np.ndarray] | None = None,
                         ):
        """Set the output model, and add an optional scale_shift_vector.
        """
        if output_model is not None:
            if scale_shift_vector_output is not None:
                self.scale_shift_vector_output = scale_shift_vector_output
                scale_vec, shift_vec = self.scale_shift_vector_output
                self.output_model = lambda x: output_model((x - shift_vec)/scale_vec) * scale_vec + shift_vec
            else:
                self.output_model = output_model
        else:
            pass

    def set_r_to_r_gen_fct(self, r_to_r_gen_opt="linear"):
        if type(r_to_r_gen_opt) == str:
            self._r_to_r_gen_opt = r_to_r_gen_opt
            r_to_r_gen_flag = self._r_to_r_gen_synonyms.get_flag(r_to_r_gen_opt)
            if r_to_r_gen_flag == 0:
                _r_to_r_gen_fct_no_model = lambda r, x: r
            elif r_to_r_gen_flag == 1:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, r ** 2))
            elif r_to_r_gen_flag == 2:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, 1))
            elif r_to_r_gen_flag == 3:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((np.hstack((r, r ** 2)), 1))
            elif r_to_r_gen_flag == 4:
                def temp(r, x):
                    r_gen = np.copy(r).T
                    r_gen[::2] = r.T[::2] ** 2
                    return r_gen.T

                _r_to_r_gen_fct_no_model = temp

            elif r_to_r_gen_flag == 5:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, np.exp(r)))
            elif r_to_r_gen_flag == 6:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((np.hstack((r, np.exp(r))), 1))

        else:
            self._r_to_r_gen_opt = "CUSTOM"
            _r_to_r_gen_fct_no_model = r_to_r_gen_opt

        if self.output_model is not None:
            self._r_to_r_gen_fct = lambda r, x: np.hstack(
                (_r_to_r_gen_fct_no_model(r, x), self.output_model(x)))
        else:
            self._r_to_r_gen_fct = lambda r, x: _r_to_r_gen_fct_no_model(r, x)

        self._r_gen_dim = self._r_to_r_gen_fct(np.zeros(self._r_dim), np.zeros(self._x_dim)).shape[
            0]


class _add_pca_r_to_rgen():
    """
    add r to r_gen behvaiour, where r_gen = standard_RGEN_SETTINGS(pca.transform(r))
    SETS:
    - self._r_to_r_gen_fct: r -> r_gen (any dimension)
    """

    def __init__(self):

        self._r_to_r_gen_opt = None
        self._r_to_r_gen_synonyms = utilities._SynonymDict()
        self._r_to_r_gen_synonyms.add_synonyms(0, ["linear_r", "simple", "linear"])
        self._r_to_r_gen_synonyms.add_synonyms(1, "linear_and_square_r")
        self._r_to_r_gen_synonyms.add_synonyms(2, ["output_bias", "bias"])
        self._r_to_r_gen_synonyms.add_synonyms(3, ["bias_and_square_r"])
        self._r_to_r_gen_synonyms.add_synonyms(4, ["linear_and_square_r_alt"])
        self._r_to_r_gen_synonyms.add_synonyms(5, ["exponential_r"])
        self._r_to_r_gen_synonyms.add_synonyms(6, ["bias_and_exponential_r"])

        self._pca = None
        self._pca_components = None
        self._pca_comps_to_skip = None
        self._norm_with_expl_var = None  # see "whitening" in sklearn.decomposition.PCA.
        self._centering_pre_fit = None

        self._input_data_mean = None
        self._matrix = None

    def set_pca_components(self, pca_components,
                           pca_comps_to_skip=0,
                           norm_with_expl_var=False,
                           centering_pre_trans=True):
        self._pca_components = pca_components
        self._pca_comps_to_skip = pca_comps_to_skip
        self._norm_with_expl_var = norm_with_expl_var
        self._centering_pre_trans = centering_pre_trans

    def fit_pca(self, r_train):
        self._pca = decomposition.PCA(n_components=self._pca_components)
        self._pca.fit(r_train)

        if self._centering_pre_trans:
            self._input_data_mean = np.mean(r_train, axis=0)

        self._matrix = self._pca.components_.T

        if self._norm_with_expl_var:
            self._matrix = self._matrix / np.sqrt(self._pca.explained_variance_)

    def set_r_to_r_gen_fct(self, r_to_r_gen_opt="linear"):
        if type(r_to_r_gen_opt) == str:
            self._r_to_r_gen_opt = r_to_r_gen_opt
            r_to_r_gen_flag = self._r_to_r_gen_synonyms.get_flag(r_to_r_gen_opt)
            if r_to_r_gen_flag == 0:
                _r_to_r_gen_fct_no_model = lambda r, x: r
            elif r_to_r_gen_flag == 1:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, r ** 2))
            elif r_to_r_gen_flag == 2:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, 1))
            elif r_to_r_gen_flag == 3:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((np.hstack((r, r ** 2)), 1))
            elif r_to_r_gen_flag == 4:
                def temp(r, x):
                    r_gen = np.copy(r).T
                    r_gen[::2] = r.T[::2] ** 2
                    return r_gen.T

                _r_to_r_gen_fct_no_model = temp
            elif r_to_r_gen_flag == 5:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, np.exp(r)))
            elif r_to_r_gen_flag == 6:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((np.hstack((r, np.exp(r))), 1))
        else:
            self._r_to_r_gen_opt = "CUSTOM"
            _r_to_r_gen_fct_no_model = r_to_r_gen_opt

        # NOTE: the [np.newaxis, :] essentially transforms the input r to r.T (and adds an axis). Thats why the matrix
        # is multiplied on the right side.

        if self._centering_pre_trans:
            self._r_to_r_gen_fct = lambda r, x: _r_to_r_gen_fct_no_model(
                ((r - self._input_data_mean)[np.newaxis, :] @ self._matrix)[0,
                self._pca_comps_to_skip:],
                x)
        else:
            self._r_to_r_gen_fct = lambda r, x: _r_to_r_gen_fct_no_model(
                (r[np.newaxis, :] @ self._matrix)[0, self._pca_comps_to_skip:],
                x)

        # if self._norm_with_expl_var and self._centering_pre_trans:
        #     self._r_to_r_gen_fct = lambda r, x: _r_to_r_gen_fct_no_model(
        #         (((r - self._input_data_mean)[np.newaxis, :]) @ (self._pca.components_.T) / np.sqrt(self._pca.explained_variance_))[0, self._pca_comps_to_skip:],
        #         x)
        #
        # elif not self._norm_with_expl_var and self._centering_pre_trans:
        #     self._r_to_r_gen_fct = lambda r, x: _r_to_r_gen_fct_no_model(
        #         (((r - self._input_data_mean)[np.newaxis, :]) @ (
        #             self._pca.components_).T)[0, self._pca_comps_to_skip:],
        #         x)
        #
        # elif not self._norm_with_expl_var and not self._centering_pre_trans:
        #     self._r_to_r_gen_fct = lambda r, x: _r_to_r_gen_fct_no_model(
        #         ((r[np.newaxis, :]) @ (
        #             self._pca.components_).T)[0, self._pca_comps_to_skip:],
        #         x)
        #
        # if self._norm_with_expl_var and not self._centering_pre_trans:
        #     self._r_to_r_gen_fct = lambda r, x: _r_to_r_gen_fct_no_model(
        #         ((r[np.newaxis, :] / np.sqrt(self._pca.explained_variance_)) @ (
        #             self._pca.components_).T)[0, self._pca_comps_to_skip:],
        #         x)

        # if self._norm_with_expl_var:
        #     self._r_to_r_gen_fct = lambda r, x: (
        #         _r_to_r_gen_fct_no_model((self._pca.transform(r[np.newaxis, :])/np.sqrt(self._pca.explained_variance_))[0, self._pca_comps_to_skip:], x))
        # else:
        #     self._r_to_r_gen_fct = lambda r, x: (_r_to_r_gen_fct_no_model(self._pca.transform(r[np.newaxis, :])[0, self._pca_comps_to_skip:], x))

        self._r_gen_dim = self._r_to_r_gen_fct(np.zeros(self._r_dim), np.zeros(self._x_dim)).shape[
            0]


class _add_centered_r_to_rgen():
    """
    add r to r_gen behvaiour, where r_gen = standard_RGEN_SETTINGS(r - mean_train_r)
    SETS:
    - self._r_to_r_gen_fct: r -> r_gen (any dimension)
    """

    def __init__(self):

        self._r_to_r_gen_opt = None
        self._r_to_r_gen_synonyms = utilities._SynonymDict()
        self._r_to_r_gen_synonyms.add_synonyms(0, ["linear_r", "simple", "linear"])
        self._r_to_r_gen_synonyms.add_synonyms(1, "linear_and_square_r")
        self._r_to_r_gen_synonyms.add_synonyms(2, ["output_bias", "bias"])
        self._r_to_r_gen_synonyms.add_synonyms(3, ["bias_and_square_r"])
        self._r_to_r_gen_synonyms.add_synonyms(4, ["linear_and_square_r_alt"])
        self._r_to_r_gen_synonyms.add_synonyms(5, ["exponential_r"])
        self._r_to_r_gen_synonyms.add_synonyms(6, ["bias_and_exponential_r"])

        self._input_data_mean = None

    def calc_mean(self, r_train):
        self._input_data_mean = np.mean(r_train, axis=0)

    def set_r_to_r_gen_fct(self, r_to_r_gen_opt="linear"):
        if type(r_to_r_gen_opt) == str:
            self._r_to_r_gen_opt = r_to_r_gen_opt
            r_to_r_gen_flag = self._r_to_r_gen_synonyms.get_flag(r_to_r_gen_opt)
            if r_to_r_gen_flag == 0:
                _r_to_r_gen_fct_no_model = lambda r, x: r
            elif r_to_r_gen_flag == 1:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, r ** 2))
            elif r_to_r_gen_flag == 2:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, 1))
            elif r_to_r_gen_flag == 3:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((np.hstack((r, r ** 2)), 1))
            elif r_to_r_gen_flag == 4:
                def temp(r, x):
                    r_gen = np.copy(r).T
                    r_gen[::2] = r.T[::2] ** 2
                    return r_gen.T

                _r_to_r_gen_fct_no_model = temp
            elif r_to_r_gen_flag == 5:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, np.exp(r)))
            elif r_to_r_gen_flag == 6:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((np.hstack((r, np.exp(r))), 1))
        else:
            self._r_to_r_gen_opt = "CUSTOM"
            _r_to_r_gen_fct_no_model = r_to_r_gen_opt

        self._r_to_r_gen_fct = lambda r, x: _r_to_r_gen_fct_no_model(
                r - self._input_data_mean,
                x)

        self._r_gen_dim = self._r_to_r_gen_fct(np.zeros(self._r_dim), np.zeros(self._x_dim)).shape[0]


class _add_model_and_pca_r_to_rgen():
    """
    TBD
    """
    def __init__(self):

        self._r_to_r_gen_opt = None
        self._r_to_r_gen_synonyms = utilities._SynonymDict()
        self._r_to_r_gen_synonyms.add_synonyms(0, ["linear_r", "simple", "linear"])
        self._r_to_r_gen_synonyms.add_synonyms(1, "linear_and_square_r")
        self._r_to_r_gen_synonyms.add_synonyms(2, ["output_bias", "bias"])
        self._r_to_r_gen_synonyms.add_synonyms(3, ["bias_and_square_r"])
        self._r_to_r_gen_synonyms.add_synonyms(4, ["linear_and_square_r_alt"])
        self._r_to_r_gen_synonyms.add_synonyms(5, ["exponential_r"])
        self._r_to_r_gen_synonyms.add_synonyms(6, ["bias_and_exponential_r"])

        self.output_model = None

        self._pca = None
        self._pca_components = None
        self._pca_comps_to_skip = None
        self._norm_with_expl_var = None  # see "whitening" in sklearn.decomposition.PCA.
        self._centering_pre_fit = None

        self._input_data_mean = None
        self._matrix = None

    def set_output_model(self, output_model):
        self.output_model = output_model

    def set_pca_components(self, pca_components, pca_comps_to_skip=0, norm_with_expl_var=False, centering_pre_trans=True):
        self._pca_components = pca_components
        self._pca_comps_to_skip = pca_comps_to_skip
        self._norm_with_expl_var = norm_with_expl_var
        self._centering_pre_trans = centering_pre_trans

    def fit_pca(self, r_train):
        self._pca = decomposition.PCA(n_components=self._pca_components)
        self._pca.fit(r_train)

        if self._centering_pre_trans:
            self._input_data_mean = np.mean(r_train, axis=0)

        self._matrix = self._pca.components_.T

        if self._norm_with_expl_var:
            self._matrix = self._matrix / np.sqrt(self._pca.explained_variance_)

    def set_r_to_r_gen_fct(self, r_to_r_gen_opt="linear"):
        if type(r_to_r_gen_opt) == str:
            self._r_to_r_gen_opt = r_to_r_gen_opt
            r_to_r_gen_flag = self._r_to_r_gen_synonyms.get_flag(r_to_r_gen_opt)
            if r_to_r_gen_flag == 0:
                _r_to_r_gen_fct_no_model_and_pca = lambda r, x: r
            elif r_to_r_gen_flag == 1:
                _r_to_r_gen_fct_no_model_and_pca = lambda r, x: np.hstack((r, r ** 2))
            elif r_to_r_gen_flag == 2:
                _r_to_r_gen_fct_no_model_and_pca = lambda r, x: np.hstack((r, 1))
            elif r_to_r_gen_flag == 3:
                _r_to_r_gen_fct_no_model_and_pca = lambda r, x: np.hstack((np.hstack((r, r ** 2)), 1))
            elif r_to_r_gen_flag == 4:
                def temp(r, x):
                    r_gen = np.copy(r).T
                    r_gen[::2] = r.T[::2] ** 2
                    return r_gen.T

                _r_to_r_gen_fct_no_model_and_pca = temp
            elif r_to_r_gen_flag == 5:
                _r_to_r_gen_fct_no_model_and_pca = lambda r, x: np.hstack((r, np.exp(r)))
            elif r_to_r_gen_flag == 6:
                _r_to_r_gen_fct_no_model_and_pca = lambda r, x: np.hstack((np.hstack((r, np.exp(r))), 1))
        else:
            self._r_to_r_gen_opt = "CUSTOM"
            _r_to_r_gen_fct_no_model_and_pca = r_to_r_gen_opt

        # NOTE: the [np.newaxis, :] essentially transforms the input r to r.T (and adds an axis). Thats why the matrix
        # is multiplied on the right side.

        if self._centering_pre_trans:
            _r_to_r_gen_fct_no_model = lambda r, x: _r_to_r_gen_fct_no_model_and_pca(
                ((r - self._input_data_mean)[np.newaxis, :] @ self._matrix)[0, self._pca_comps_to_skip:],
                x)
        else:
            _r_to_r_gen_fct_no_model = lambda r, x: _r_to_r_gen_fct_no_model_and_pca(
                (r[np.newaxis, :] @ self._matrix)[0, self._pca_comps_to_skip:],
                x)

        self._r_to_r_gen_fct = lambda r, x: np.hstack((_r_to_r_gen_fct_no_model(r, x), self.output_model(x)))

        self._r_gen_dim = self._r_to_r_gen_fct(np.zeros(self._r_dim), np.zeros(self._x_dim)).shape[0]
        print("r_gen_dim def: ", self._r_gen_dim)


class _add_w_in():
    """
    add basic w_in behavior

    Only creates a self._w_in. What you do with it is still open.
    """
    def __init__(self):

        self._w_in_opt = None
        self._w_in_scale = None
        self._w_in_flag_synonyms = utilities._SynonymDict()
        self._w_in_flag_synonyms.add_synonyms(0, ["random_sparse"])
        self._w_in_flag_synonyms.add_synonyms(1, ["ordered_sparse"])
        self._w_in_flag_synonyms.add_synonyms(2, ["random_dense_uniform"])
        self._w_in_flag_synonyms.add_synonyms(3, ["random_dense_gaussian"])

    def create_w_in(self, w_in_opt, w_in_scale=1.0):
        # self.logger.debug("Create w_in")

        if type(w_in_opt) == str:
            self._w_in_scale = w_in_scale
            self._w_in_opt = w_in_opt
            w_in_flag = self._w_in_flag_synonyms.get_flag(w_in_opt)

            if w_in_flag == 0:
                self._w_in = np.zeros((self._r_dim, self._x_dim))
                for i in range(self._r_dim):
                    random_x_coord = np.random.choice(np.arange(self._x_dim))
                    self._w_in[i, random_x_coord] = np.random.uniform(
                        low=-self._w_in_scale,
                        high=self._w_in_scale)

            elif w_in_flag == 1:
                self._w_in = np.zeros((self._r_dim, self._x_dim))
                dim_wise = np.array([int(self._r_dim / self._x_dim)] * self._x_dim)
                dim_wise[:self._r_dim % self._x_dim] += 1
                s = 0
                dim_wise_2 = dim_wise[:]
                for i in range(len(dim_wise_2)):
                    s += dim_wise_2[i]
                    dim_wise[i] = s
                dim_wise = np.append(dim_wise, 0)
                for d in range(self._x_dim):
                    for i in range(dim_wise[d - 1], dim_wise[d]):
                        self._w_in[i, d] = np.random.uniform(
                            low=-self._w_in_scale,
                            high=self._w_in_scale)

            elif w_in_flag == 2:
                self._w_in = np.random.uniform(low=-self._w_in_scale,
                                               high=self._w_in_scale,
                                               size=(self._r_dim, self._x_dim))

            elif w_in_flag == 3:
                self._w_in = self._w_in_scale * np.random.randn(self._r_dim, self._x_dim)

        else:
            self._w_in_opt = "CUSTOM"
            self._w_in = w_in_opt


class _add_standard_input_coupling():
    """
    add normal input coupling via
    SETS:
        - self._inp_coupling_fct
    """

    def __init__(self):
        self._inp_coupling_fct = lambda x: self._w_in @ x


class _add_model_input_coupling():
    """
    add model input coupling. Input -> W_in * (Input, model(Input)
    used for Input Hybrid Reservoir Computing
    SETS:
        - self._inp_coupling_fct
    """

    def __init__(self):

        self._w_in_opt = None
        self._w_in_scale = None
        self._w_in_flag_synonyms = utilities._SynonymDict()
        self._w_in_flag_synonyms.add_synonyms(0, ["random_sparse"])
        self._w_in_flag_synonyms.add_synonyms(1, ["ordered_sparse"])
        self._w_in_flag_synonyms.add_synonyms(2, ["random_dense_uniform"])
        self._w_in_flag_synonyms.add_synonyms(3, ["random_dense_gaussian"])

        self.input_model = None
        self.input_model_to_res_factor = None
        self.scale_shift_vector_input = None

    def set_input_model(self,
                        input_model: Callable[[np.ndarray], np.ndarray] | None = None,
                        input_model_to_res_factor: float = 0.5,
                        scale_shift_vector_input: tuple[np.ndarray, np.ndarray] | None = None):
        if input_model is not None:
            if self.scale_shift_vector_input is not None:
                self.scale_shift_vector_input = scale_shift_vector_input
                scale_vec, shift_vec = self.scale_shift_vector_input
                self.input_model = lambda x: input_model((x - shift_vec) / scale_vec) * scale_vec + shift_vec
            else:
                self.input_model = input_model
            self.input_model_to_res_factor = input_model_to_res_factor

            self._inp_coupling_fct = lambda x: self._w_in @ np.hstack((x, self.input_model(x)))

        else:
            self._inp_coupling_fct = lambda x: self._w_in @ x

    def create_w_in(self, w_in_opt, w_in_scale=1.0):
        # self.logger.debug("Create w_in")

        if type(w_in_opt) == str:
            self._w_in_scale = w_in_scale
            self._w_in_opt = w_in_opt
            w_in_flag = self._w_in_flag_synonyms.get_flag(w_in_opt)

            if self.input_model is not None:
                x_dim_inp_model = self.input_model(np.ones(self._x_dim)).size
                x_dim_gen = x_dim_inp_model + self._x_dim
            else:
                x_dim_gen = self._x_dim

            if w_in_flag == 0:
                self._w_in = np.zeros((self._r_dim, x_dim_gen))

                if self.input_model is not None:

                    nr_res_nodes_connected_to_model = int(self.input_model_to_res_factor * self._r_dim)
                    nr_res_nodes_connected_to_raw = self._r_dim - nr_res_nodes_connected_to_model

                    nodes_connected_to_raw = np.random.choice(np.arange(self._r_dim),
                                                              size=nr_res_nodes_connected_to_raw,
                                                              replace=False)
                    nodes_connected_to_raw = np.sort(nodes_connected_to_raw)

                    for index in nodes_connected_to_raw:
                        random_x_coord = np.random.choice(np.arange(self._x_dim))
                        self._w_in[index, random_x_coord] = np.random.uniform(
                            low=-self._w_in_scale,
                            high=self._w_in_scale)
                    nodes_connected_to_model = np.delete(np.arange(self._r_dim),
                                                         nodes_connected_to_raw)
                    for index in nodes_connected_to_model:
                        random_x_coord = np.random.choice(np.arange(x_dim_inp_model))
                        self._w_in[index, random_x_coord + self._x_dim] = np.random.uniform(
                            low=-self._w_in_scale,
                            high=self._w_in_scale)
                else:
                    for i in range(self._r_dim):
                        random_x_coord = np.random.choice(np.arange(x_dim_gen))
                        self._w_in[i, random_x_coord] = np.random.uniform(
                            low=-self._w_in_scale,
                            high=self._w_in_scale)

            elif w_in_flag == 1:
                raise Exception("Not implemented")
            elif w_in_flag == 2:
                self._w_in = np.random.uniform(low=-self._w_in_scale,
                                               high=self._w_in_scale,
                                               size=(self._r_dim, x_dim_gen))
            elif w_in_flag == 3:
                self._w_in = self._w_in_scale * np.random.randn(self._r_dim, x_dim_gen)
        else:
            self._w_in_opt = "CUSTOM"
            self._w_in = w_in_opt


class _add_same_model_input_output():
    """
    Add model input coupling. Input -> W_in * (Input, model(Input)
    Add model output coupling. Only calculate the model(Input) once!
    Used for Input Hybrid Reservoir Computing

    """

    def __init__(self):

        self._w_in_opt = None
        self._w_in_scale = None
        self._w_in_flag_synonyms = utilities._SynonymDict()
        self._w_in_flag_synonyms.add_synonyms(0, ["random_sparse"])
        self._w_in_flag_synonyms.add_synonyms(1, ["ordered_sparse"])
        self._w_in_flag_synonyms.add_synonyms(2, ["random_dense_uniform"])
        self._w_in_flag_synonyms.add_synonyms(3, ["random_dense_gaussian"])

        self.input_model = None
        self.model_to_network_factor = None

        self.last_input_model_result = None

        self._r_to_r_gen_opt = None
        self._r_to_r_gen_synonyms = utilities._SynonymDict()
        self._r_to_r_gen_synonyms.add_synonyms(0, ["linear_r", "simple", "linear"])
        self._r_to_r_gen_synonyms.add_synonyms(1, "linear_and_square_r")
        self._r_to_r_gen_synonyms.add_synonyms(2, ["output_bias", "bias"])
        self._r_to_r_gen_synonyms.add_synonyms(3, ["bias_and_square_r"])
        self._r_to_r_gen_synonyms.add_synonyms(4, ["linear_and_square_r_alt"])
        self._r_to_r_gen_synonyms.add_synonyms(5, ["exponential_r"])
        self._r_to_r_gen_synonyms.add_synonyms(6, ["bias_and_exponential_r"])

    def save_input_model_result(self, func, x):
        x_new = func(x)
        self._last_input_model_results = x_new
        return x_new

    def set_input_model(self, input_model, model_to_network_factor=0.5):
        self.input_model = input_model
        self.model_to_network_factor = model_to_network_factor

        self._inp_coupling_fct = lambda x: self._w_in @ np.hstack(
            (x, self.save_input_model_result(self.input_model, x)))

    def create_w_in(self, w_in_opt, w_in_scale=1.0):
        # self.logger.debug("Create w_in")

        if type(w_in_opt) == str:
            self._w_in_scale = w_in_scale
            self._w_in_opt = w_in_opt
            w_in_flag = self._w_in_flag_synonyms.get_flag(w_in_opt)
            x_dim_inp_model = self.input_model(np.ones(self._x_dim)).size
            x_dim_gen = x_dim_inp_model + self._x_dim

            # print("x_dim_gen: ", x_dim_gen)

            if w_in_flag == 0:

                self._w_in = np.zeros((self._r_dim, x_dim_gen))

                nr_res_nodes_connected_to_model = int(self.model_to_network_factor * self._r_dim)
                nr_res_nodes_connected_to_raw = self._r_dim - nr_res_nodes_connected_to_model

                nodes_connected_to_raw = np.random.choice(np.arange(self._r_dim),
                                                          size=nr_res_nodes_connected_to_raw,
                                                          replace=False)
                nodes_connected_to_raw = np.sort(nodes_connected_to_raw)

                for index in nodes_connected_to_raw:
                    random_x_coord = np.random.choice(np.arange(self._x_dim))
                    self._w_in[index, random_x_coord] = np.random.uniform(
                        low=-self._w_in_scale,
                        high=self._w_in_scale)
                nodes_connected_to_model = np.delete(np.arange(self._r_dim),
                                                     nodes_connected_to_raw)
                for index in nodes_connected_to_model:
                    random_x_coord = np.random.choice(np.arange(x_dim_inp_model))
                    self._w_in[index, random_x_coord + self._x_dim] = np.random.uniform(
                        low=-self._w_in_scale,
                        high=self._w_in_scale)
            elif w_in_flag == 1:
                raise Exception("Not implemented")
            elif w_in_flag == 2:
                self._w_in = np.random.uniform(low=-self._w_in_scale,
                                               high=self._w_in_scale,
                                               size=(self._r_dim, x_dim_gen))
            elif w_in_flag == 3:
                self._w_in = self._w_in_scale * np.random.randn(self._r_dim, x_dim_gen)
        else:
            self._w_in_opt = "CUSTOM"
            self._w_in = w_in_opt

    def set_r_to_r_gen_fct(self, r_to_r_gen_opt="linear"):
        if type(r_to_r_gen_opt) == str:
            self._r_to_r_gen_opt = r_to_r_gen_opt
            r_to_r_gen_flag = self._r_to_r_gen_synonyms.get_flag(r_to_r_gen_opt)
            if r_to_r_gen_flag == 0:
                _r_to_r_gen_fct_no_model = lambda r, x: r
            elif r_to_r_gen_flag == 1:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, r ** 2))
            elif r_to_r_gen_flag == 2:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, 1))
            elif r_to_r_gen_flag == 3:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((np.hstack((r, r ** 2)), 1))
            elif r_to_r_gen_flag == 4:
                def temp(r, x):
                    r_gen = np.copy(r).T
                    r_gen[::2] = r.T[::2] ** 2
                    return r_gen.T
                _r_to_r_gen_fct_no_model = temp

            elif r_to_r_gen_flag == 5:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((r, np.exp(r)))
            elif r_to_r_gen_flag == 6:
                _r_to_r_gen_fct_no_model = lambda r, x: np.hstack((np.hstack((r, np.exp(r))), 1))

        else:
            self._r_to_r_gen_opt = "CUSTOM"
            _r_to_r_gen_fct_no_model = r_to_r_gen_opt

        self._r_to_r_gen_fct = lambda r, x: np.hstack(
            (_r_to_r_gen_fct_no_model(r, x), self._last_input_model_results))

        r_to_r_gen_init = lambda r, x: np.hstack(
            (_r_to_r_gen_fct_no_model(r, x), self.input_model(x)))

        self._r_gen_dim = r_to_r_gen_init(np.zeros(self._r_dim), np.zeros(self._x_dim)).shape[0]



class _add_standard_y_to_x():
    """
    Function from output to input for loop is identity
    """

    def __init__(self):
        self._y_to_x_fct = lambda x, y: y


class _add_y_diff_to_x():
    """
    Function from output to input where output is difference
    """

    def __init__(self):
        self._dt_difference = None
        self._y_to_x_fct = lambda x, y: x + y * self._dt_difference

    def set_dt_difference(self, dt_difference=0.1):
        self._dt_difference = dt_difference


class _add_preprocess_r_to_rgen():
    """
    1. Scaler and/or center r_train.
        - specify if you want to scale
        - specify if you want to center
    2. Add a pca transformation.
        - specify the number of components
        - specify if you want to center
    """

    def __init__(self):

        self._r_to_r_gen_opt = None
        self._r_to_r_gen_synonyms = utilities._SynonymDict()
        self._r_to_r_gen_synonyms.add_synonyms(0, ["linear_r", "simple", "linear"])
        self._r_to_r_gen_synonyms.add_synonyms(1, "linear_and_square_r")
        self._r_to_r_gen_synonyms.add_synonyms(2, ["output_bias", "bias"])
        self._r_to_r_gen_synonyms.add_synonyms(3, ["bias_and_square_r"])
        self._r_to_r_gen_synonyms.add_synonyms(4, ["linear_and_square_r_alt"])
        self._r_to_r_gen_synonyms.add_synonyms(5, ["exponential_r"])
        self._r_to_r_gen_synonyms.add_synonyms(6, ["bias_and_exponential_r"])

        self._r_train_std = None
        self._r_train_mean = None

        self._pca = None
        self._pca_matrix = None
        self._pca_input_mean = None

    def set_r_train_preprocessing(self,
                                  center_r_train: bool = False,
                                  rescale_r_train: bool = False,
                                  perform_pca: bool = False,
                                  pca_components: int | None = None,
                                  center_pca_input: bool = True,
                                  ):
        self.center_r_train = center_r_train
        self.rescale_r_train = rescale_r_train
        self.perform_pca = perform_pca
        self.pca_components = pca_components
        self.center_pca_input = center_pca_input

    def set_r_to_r_gen_fct(self,
                           r_train: np.ndarray,
                           r_to_r_gen_opt="linear"
                           ):
        if type(r_to_r_gen_opt) == str:
            self._r_to_r_gen_opt = r_to_r_gen_opt
            r_to_r_gen_flag = self._r_to_r_gen_synonyms.get_flag(r_to_r_gen_opt)
            if r_to_r_gen_flag == 0:
                basic_rrgen_fct = lambda r, x: r
            elif r_to_r_gen_flag == 1:
                basic_rrgen_fct = lambda r, x: np.hstack((r, r ** 2))
            elif r_to_r_gen_flag == 2:
                basic_rrgen_fct = lambda r, x: np.hstack((r, 1))
            elif r_to_r_gen_flag == 3:
                basic_rrgen_fct = lambda r, x: np.hstack((np.hstack((r, r ** 2)), 1))
            elif r_to_r_gen_flag == 4:
                def temp(r, x):
                    r_gen = np.copy(r).T
                    r_gen[::2] = r.T[::2] ** 2
                    return r_gen.T

                basic_rrgen_fct = temp
            elif r_to_r_gen_flag == 5:
                basic_rrgen_fct = lambda r, x: np.hstack((r, np.exp(r)))
            elif r_to_r_gen_flag == 6:
                basic_rrgen_fct = lambda r, x: np.hstack((np.hstack((r, np.exp(r))), 1))
        else:
            self._r_to_r_gen_opt = "CUSTOM"
            basic_rrgen_fct = r_to_r_gen_opt

        if self.center_r_train:
            self._r_train_mean = np.mean(r_train, axis=0)
            r_train = r_train - self._r_train_mean
        else:
            self._r_train_mean = 0
        if self.rescale_r_train:
            self._r_train_scale = np.std(r_train, axis=0)
            r_train = r_train/self._r_train_scale
        else:
            self._r_train_scale = 1
        if self.perform_pca:
            self._pca = decomposition.PCA(n_components=self.pca_components)
            self._pca.fit(r_train)
            self._pca_matrix = self._pca.components_
            if self.center_pca_input:
                self._pca_input_mean = np.mean(r_train, axis=0)
            else:
                self._pca_input_mean = 0

            preprocess_fct = lambda r: self._pca_matrix @ (
                        ((r - self._r_train_mean) / self._r_train_scale) - self._pca_input_mean)
        else:
            preprocess_fct = lambda r: (r - self._r_train_mean) / self._r_train_scale

        self._r_to_r_gen_fct = lambda r, x: basic_rrgen_fct(preprocess_fct(r), x)

        self._r_gen_dim = self._r_to_r_gen_fct(np.zeros(self._r_dim), np.zeros(self._x_dim)).shape[
            0]


class ESN_normal(_ResCompCore,
                 _add_basic_defaults,
                 _add_network_update_fct,
                 _add_basic_r_to_rgen,
                 _add_w_in,
                 _add_standard_input_coupling,
                 _add_standard_y_to_x):
    """
    Pretty standard ESN class
    """

    def __init__(self):
        _ResCompCore.__init__(self)
        _add_basic_defaults.__init__(self)
        _add_network_update_fct.__init__(self)
        _add_basic_r_to_rgen.__init__(self)
        _add_w_in.__init__(self)
        _add_standard_input_coupling.__init__(self)
        _add_standard_y_to_x.__init__(self)

        self._input_noise_scale = None
        self._input_noise_seed = None

    def train(self, use_for_train, sync_steps=0, reset_res_state=True, **kwargs):
        sync = use_for_train[:sync_steps]
        train = use_for_train[sync_steps:]

        x_train = train[:-1]
        y_train = train[1:]

        # add input noise:
        if self._input_noise_scale is not None:
            inp_rng = np.random.default_rng(self._input_noise_seed)
            x_train += inp_rng.standard_normal(x_train.shape) * self._input_noise_scale

        super(ESN_normal, self).train(sync, x_train, y_train, reset_res_state=reset_res_state,
                                      **kwargs)

    def build(self,
              x_dim,
              r_dim=500,
              n_rad=0.1,
              n_avg_deg=6.0,
              n_type_opt="erdos_renyi",
              network_creation_attempts=10,
              r_to_r_gen_opt="linear",
              act_fct_opt="tanh",
              node_bias_opt="no_bias",
              bias_scale=1.0,
              leak_factor=0.0,
              w_in_opt="random_sparse",
              w_in_scale=1.0,
              default_res_state=None,
              reg_param=1e-8,
              network_seed=None,
              bias_seed=None,
              w_in_seed=None,
              input_noise_scale: float | None = None,
              input_noise_seed: int | None = None
              ):

        # self.logger.debug("Building ESN Archtecture")

        self._input_noise_scale = input_noise_scale
        self._input_noise_seed = input_noise_seed

        self._x_dim = x_dim
        self._y_dim = x_dim
        self._r_dim = r_dim

        if network_seed is not None:
            with utilities.temp_seed(network_seed):
                self.create_network(n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt,
                                    network_creation_attempts=network_creation_attempts)
        else:
            self.create_network(n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt,
                                network_creation_attempts=network_creation_attempts)
        self.set_r_to_r_gen_fct(r_to_r_gen_opt=r_to_r_gen_opt)
        self.set_activation_function(act_fct_opt=act_fct_opt)

        if bias_seed is not None:
            with utilities.temp_seed(bias_seed):
                self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)
        else:
            self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)

        self.set_leak_factor(leak_factor=leak_factor)

        if w_in_seed is not None:
            with utilities.temp_seed(w_in_seed):
                self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)
        else:
            self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)

        self.set_default_res_state(default_res_state=default_res_state)
        self.set_reg_param(reg_param=reg_param)


class ESN_no_res(_ResCompCore, _add_basic_defaults, _add_no_res_update_fct, _add_basic_r_to_rgen,
                 _add_w_in, _add_standard_input_coupling, _add_standard_y_to_x):
    """
    Pretty standard ESN class
    """
    def __init__(self):
        _ResCompCore.__init__(self)
        _add_basic_defaults.__init__(self)
        _add_no_res_update_fct.__init__(self)
        _add_basic_r_to_rgen.__init__(self)
        _add_w_in.__init__(self)
        _add_standard_input_coupling.__init__(self)
        _add_standard_y_to_x.__init__(self)

    def train(self, use_for_train, sync_steps=0, reset_res_state=True, **kwargs):
        sync = use_for_train[:sync_steps]
        train = use_for_train[sync_steps:]

        x_train = train[:-1]
        y_train = train[1:]
        super(ESN_no_res, self).train(sync, x_train, y_train, reset_res_state=reset_res_state, **kwargs)

    def build(self, x_dim, r_dim=500,
              r_to_r_gen_opt="linear", act_fct_opt="tanh", node_bias_opt="no_bias", bias_scale=1.0, leak_factor=0.0,
              w_in_opt="random_sparse", w_in_scale=1.0, default_res_state=None, reg_param=1e-8,
              bias_seed=None, w_in_seed=None):

        # self.logger.debug("Building ESN Archtecture")

        self._x_dim = x_dim
        self._y_dim = x_dim
        self._r_dim = r_dim

        self.set_r_to_r_gen_fct(r_to_r_gen_opt=r_to_r_gen_opt)
        self.set_activation_function(act_fct_opt=act_fct_opt)

        if bias_seed is not None:
            with utilities.temp_seed(bias_seed):
                self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)
        else:
            self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)

        self.set_leak_factor(leak_factor=leak_factor)

        if w_in_seed is not None:
            with utilities.temp_seed(w_in_seed):
                self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)
        else:
            self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)

        self.set_default_res_state(default_res_state=default_res_state)
        self.set_reg_param(reg_param=reg_param)


class ESN_pca(_ResCompCore, _add_basic_defaults, _add_network_update_fct, _add_pca_r_to_rgen,
                 _add_w_in, _add_standard_input_coupling, _add_standard_y_to_x):
    """
    Train and r_to_r_gen definition is intertwined so we cant just predefine r_to_r_gen -> its all in train
    """
    def __init__(self):
        _ResCompCore.__init__(self)
        _add_basic_defaults.__init__(self)
        _add_network_update_fct.__init__(self)
        _add_pca_r_to_rgen.__init__(self)
        _add_w_in.__init__(self)
        _add_standard_input_coupling.__init__(self)
        _add_standard_y_to_x.__init__(self)

    def train(self, use_for_train, sync_steps=0, reset_res_state=True, save_y_train=False, **kwargs):
        sync = use_for_train[:sync_steps]
        train = use_for_train[sync_steps:]

        x_train = train[:-1]
        y_train = train[1:]

        if save_y_train:
            self._saved_y_train = y_train

        # r to r_gen with pca
        if reset_res_state:
            self.reset_r()

        self.drive(sync)

        kwargs["save_r"] = True
        kwargs["save_r_gen"] = False

        save_out = False
        if "save_out" in kwargs.keys():
            if kwargs["save_out"]:  # can not save out during training before w_out is calculated
                save_out = True
                kwargs["save_out"] = False

        self.drive(x_train, **kwargs)
        r_train = self.get_r()

        self.fit_pca(r_train)
        self.set_r_to_r_gen_fct(r_to_r_gen_opt=self._r_to_r_gen_opt)

        train_steps = r_train.shape[0]
        r_gen = np.zeros((train_steps, self._r_gen_dim))
        for i in range(train_steps):
            r_gen[i, :] = self._r_to_r_gen_fct(r_train[i, :], None)

        # DEBUGGING:

        # print("A.T @ A: ", self._matrix.T @ self._matrix)
        # mean_r = np.mean(r_train, axis=0)
        # A = self._pca.components_
        # is_unitary = np.allclose(np.eye(len(A)), A.dot(A.T.conj()))
        # print("IS UNITARY: ", is_unitary)
        # # pca_manual = np.dot(r_train - mean_r, A.T)
        # pca_manual = (r_train - mean_r) @ (A.T)
        #
        # mean_r @ (A.T)
        #
        # print("PCA MANUAL - PCA REAL: ", pca_manual - r_gen)
        #
        # print("R TRAIN MEAN PRE: ", np.mean(r_train, axis=0))
        # print("R GEN TRAIN MEAN: ", np.mean(r_gen, axis=0))
        # END DEBUG

        self._saved_r_gen = r_gen
        self._fit_w_out(y_train, self._saved_r_gen)

        if save_out:
            self._saved_out = (self._w_out @ self._saved_r_gen.T).T

    def build(self, x_dim, r_dim=500, n_rad=0.1, n_avg_deg=6.0, n_type_opt="erdos_renyi", network_creation_attempts=10,
              pca_components=None, pca_comps_to_skip=0, norm_with_expl_var=False, centering_pre_trans=True,
              r_to_r_gen_opt="linear", act_fct_opt="tanh", node_bias_opt="no_bias", bias_scale=1.0, leak_factor=0.0,
              w_in_opt="random_sparse", w_in_scale=1.0, default_res_state=None, reg_param=1e-8, network_seed=None,
              bias_seed=None, w_in_seed=None):

        # self.logger.debug("Building ESN Archtecture")

        if pca_components is None:
            pca_components = r_dim

        self.set_pca_components(pca_components, pca_comps_to_skip=pca_comps_to_skip,
                                norm_with_expl_var=norm_with_expl_var, centering_pre_trans=centering_pre_trans)

        self._r_to_r_gen_opt = r_to_r_gen_opt

        self._x_dim = x_dim
        self._y_dim = x_dim
        self._r_dim = r_dim

        if network_seed is not None:
            with utilities.temp_seed(network_seed):
                self.create_network(n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt,
                                    network_creation_attempts=network_creation_attempts)
        else:
            self.create_network(n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt,
                                network_creation_attempts=network_creation_attempts)
        self.set_activation_function(act_fct_opt=act_fct_opt)

        if bias_seed is not None:
            with utilities.temp_seed(bias_seed):
                self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)
        else:
            self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)

        self.set_leak_factor(leak_factor=leak_factor)

        if w_in_seed is not None:
            with utilities.temp_seed(w_in_seed):
                self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)
        else:
            self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)

        self.set_default_res_state(default_res_state=default_res_state)
        self.set_reg_param(reg_param=reg_param)


class ESN_r_process(_ResCompCore,
                    _add_basic_defaults,
                    _add_network_update_fct,
                    _add_preprocess_r_to_rgen,
                    _add_w_in,
                    _add_standard_input_coupling,
                    _add_standard_y_to_x):
    """
    Able to:
    - Add noise to x_train
    - center r_train
    - rescale r_train
    - perform a pca
    """

    def __init__(self):
        _ResCompCore.__init__(self)
        _add_basic_defaults.__init__(self)
        _add_network_update_fct.__init__(self)
        _add_preprocess_r_to_rgen.__init__(self)
        _add_w_in.__init__(self)
        _add_standard_input_coupling.__init__(self)
        _add_standard_y_to_x.__init__(self)

        self._input_noise_scale = None
        self._input_noise_seed = None

    def train(self,
              use_for_train,
              sync_steps=0,
              reset_res_state=True,
              save_y_train=False,
              **kwargs):
        sync = use_for_train[:sync_steps]
        train = use_for_train[sync_steps:]

        x_train = train[:-1]
        y_train = train[1:]

        # add input noise:
        if self._input_noise_scale is not None:
            inp_rng = np.random.default_rng(self._input_noise_seed)
            x_train += inp_rng.standard_normal(x_train.shape) * self._input_noise_scale

        if save_y_train:
            self._saved_y_train = y_train

        # r to r_gen with pca
        if reset_res_state:
            self.reset_r()

        self.drive(sync)

        kwargs["save_r"] = True
        kwargs["save_r_gen"] = False

        save_out = False
        if "save_out" in kwargs.keys():
            if kwargs["save_out"]:  # can not save out during training before w_out is calculated
                save_out = True
                kwargs["save_out"] = False

        self.drive(x_train, **kwargs)
        r_train = self.get_r()

        self.set_r_to_r_gen_fct(r_train,
                                r_to_r_gen_opt=self._r_to_r_gen_opt)

        train_steps = r_train.shape[0]
        r_gen = np.zeros((train_steps, self._r_gen_dim))
        for i in range(train_steps):
            r_gen[i, :] = self._r_to_r_gen_fct(r_train[i, :], None)

        self._saved_r_gen = r_gen
        self._fit_w_out(y_train, self._saved_r_gen)

        if save_out:
            self._saved_out = (self._w_out @ self._saved_r_gen.T).T

    def build(self,
              x_dim,
              r_dim=500,
              n_rad=0.1,
              n_avg_deg=6.0,
              n_type_opt="erdos_renyi",
              network_creation_attempts=10,
              center_r_train: bool = False,
              rescale_r_train: bool = False,
              perform_pca: bool = False,
              pca_components: int | None = None,
              center_pca_input: bool = False,
              r_to_r_gen_opt="linear",
              act_fct_opt="tanh",
              node_bias_opt="no_bias",
              bias_scale=1.0,
              leak_factor=0.0,
              w_in_opt="random_sparse",
              w_in_scale=1.0,
              default_res_state=None,
              reg_param=1e-8,
              network_seed=None,
              bias_seed=None,
              w_in_seed=None,
              input_noise_scale: float | None = None,
              input_noise_seed: int | None = None):

        self._input_noise_scale = input_noise_scale
        self._input_noise_seed = input_noise_seed

        self.set_r_train_preprocessing(center_r_train,
                                       rescale_r_train,
                                       perform_pca,
                                       pca_components,
                                       center_pca_input)

        self._r_to_r_gen_opt = r_to_r_gen_opt

        self._x_dim = x_dim
        self._y_dim = x_dim
        self._r_dim = r_dim

        if network_seed is not None:
            with utilities.temp_seed(network_seed):
                self.create_network(n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt,
                                    network_creation_attempts=network_creation_attempts)
        else:
            self.create_network(n_rad=n_rad, n_avg_deg=n_avg_deg, n_type_opt=n_type_opt,
                                network_creation_attempts=network_creation_attempts)
        self.set_activation_function(act_fct_opt=act_fct_opt)

        if bias_seed is not None:
            with utilities.temp_seed(bias_seed):
                self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)
        else:
            self.set_node_bias(node_bias_opt=node_bias_opt, bias_scale=bias_scale)

        self.set_leak_factor(leak_factor=leak_factor)

        if w_in_seed is not None:
            with utilities.temp_seed(w_in_seed):
                self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)
        else:
            self.create_w_in(w_in_opt=w_in_opt, w_in_scale=w_in_scale)

        self.set_default_res_state(default_res_state=default_res_state)
        self.set_reg_param(reg_param=reg_param)
