""" Hybrid esn file, combining ESN with knowledge-based prediction.

Related to:

Pathak, J., Wikner, A., Fussell, R., Chandra, S., Hunt, B., Girvan, M., & Ott, E.
(2018). Hybrid Forecasting of Chaotic Processes: Using Machine Learning in Conjunction with a
Knowledge-Based Model. Chaos, 28(4). https://doi.org/10.1063/1.5028373
"""

from __future__ import annotations
from typing import Callable, Tuple, Any

import numpy as np
from sklearn import decomposition

from src.esn_src import utilities
import src.esn_src.esn as esn

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


class ESN_hybrid(esn._ResCompCore,
                 esn._add_basic_defaults,
                 esn._add_network_update_fct,
                 _add_model_r_to_rgen,
                 _add_model_input_coupling,
                 esn._add_standard_y_to_x):
    """
    Hybrid esn class as in Pathak 2018.
    """

    def __init__(self):
        esn._ResCompCore.__init__(self)
        esn._add_basic_defaults.__init__(self)
        esn._add_network_update_fct.__init__(self)
        _add_model_r_to_rgen.__init__(self)
        _add_model_input_coupling.__init__(self)
        esn._add_standard_y_to_x.__init__(self)

        self._input_noise_scale = None
        self._input_noise_seed = None

    def train(self, use_for_train, sync_steps=0, reset_res_state=True, **kwargs) -> tuple[
        Any, Any]:
        sync = use_for_train[:sync_steps]
        train = use_for_train[sync_steps:]

        x_train = train[:-1]
        y_train = train[1:]

        # add input noise:
        if self._input_noise_scale is not None:
            inp_rng = np.random.default_rng(self._input_noise_seed)
            x_train += inp_rng.standard_normal(x_train.shape) * self._input_noise_scale

        return super(ESN_hybrid, self).train(sync, x_train, y_train,
                                             reset_res_state=reset_res_state,
                                             **kwargs)

    def build(self,
              x_dim: int,
              r_dim: int = 500,
              n_rad: float = 0.1,
              n_avg_deg: float = 6.0,
              n_type_opt="erdos_renyi",
              network_creation_attempts: int = 10,
              r_to_r_gen_opt="linear",
              act_fct_opt="tanh",
              node_bias_opt="no_bias",
              bias_scale=1.0,
              leak_factor=0.0,
              w_in_opt="random_sparse",
              w_in_scale=1.0,
              input_model: Callable[[np.ndarray], np.ndarray] | None = None,
              input_model_to_res_factor: float = 0.5,
              scale_shift_vector_input: tuple[np.ndarray, np.ndarray] | None = None,
              output_model: Callable[[np.ndarray], np.ndarray] | None = None,
              scale_shift_vector_output: tuple[np.ndarray, np.ndarray] | None = None,
              default_res_state=None,
              reg_param=1e-8,
              network_seed=None,
              bias_seed=None,
              w_in_seed=None,
              input_noise_scale: float | None = None,
              input_noise_seed: int | None = None
              ):

        self._input_noise_scale = input_noise_scale
        self._input_noise_seed = input_noise_seed

        self._x_dim = x_dim
        self._y_dim = x_dim
        self._r_dim = r_dim

        self.set_input_model(input_model=input_model,
                             input_model_to_res_factor=input_model_to_res_factor,
                             scale_shift_vector_input=scale_shift_vector_input)

        self.set_output_model(output_model=output_model,
                              scale_shift_vector_output=scale_shift_vector_output)

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
