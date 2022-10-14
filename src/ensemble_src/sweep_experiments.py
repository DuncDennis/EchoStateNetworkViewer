"""File to create parameter-sweep and ensemble based experiments for RC predictions. """
from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd

import src.esn_src.utilities as utilities
import src.esn_src.measures as measures

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return (np.linalg.norm(y_true - y_pred, axis=1)**2).mean()

def valid_time(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    error_series = measures.error_over_time(y_pred, y_true,
                                            normalization="root_of_avg_of_spacedist_squared")
    return measures.valid_time_index(error_series, error_threshold=0.4)

DEFAULT_TRAIN_METRICS = {
    "MSE": mse
}

DEFAULT_PREDICT_METRICS = {
    "MSE": mse,
    "VT": valid_time
}


class PredModelEnsemble:
    """Class to train, validate and test a prediction model.

    - Has a build function, where all the parameters are defined.
    - Has a train function, where the model is trained.
    - Has a predict function, to run the validation.
    - Has a test function, for the final testing of the data.
    """

    def __init__(self) -> None:
        self.model_objects: list | None = None
        self.master_seed: int | None = None
        self.seeds: list[int] | None = None
        self.rng: np.random.Generator | None = None
        self.n_ens = None

        self.train_metrics: None | dict[str, Callable] = None
        self.predict_metrics: None | dict[str, Callable] = None

        self.train_metrics_results: None | dict[str, np.ndarray] = None
        self.predict_metrics_results: None | dict[str, np.ndarray] = None

    def build_models(self,
                     model_class,
                     n_ens: int = 1,
                     seed: int = 0,
                     **build_args) -> None:
        """Function to build an ensemble of models each with a different seed.

        Saves all built models to the list: self.model_objects.

        Args:
            model_object: The model object, as for example ESN_normal()
            n_ens: The ensemble size.
            seed: The random seed.
            **build_args: All the arguments parsed to model_object.build(**build_args)

        """
        self.n_ens = n_ens
        self.master_seed = seed
        self.rng = np.random.default_rng(self.master_seed)
        self.seeds = self.rng.integers(0, 10000000, size=self.n_ens)

        self.model_objects = []
        for seed in self.seeds:
            with utilities.temp_seed(seed):
                model_object = model_class()
                model_object.build(**build_args)
                self.model_objects.append(model_object)

    def train_models(self,
                     train_data: np.ndarray,
                     sync_steps: int = 0,
                     train_metrics: None | dict[str, Callable] = None,
                     **train_args
                     ) -> None:
        """Train all the models in the self.model_objects list.

        Calls for each model_object in self.model_objects list the model_object.train(train_data,
        sync_steps=sync_steps, **train_args) function, which returns train_fit, and train_true.

        For each model_object and each train_metric given in the train_metrics dictionary, a
        value is saved in self.train_metric_results.

        Args:
            train_data: The training data of shape: (train_steps + sync_steps, sys_dim).
            sync_steps: The steps used for synchronization.
            train_metrics: A dictionary of functions which are the train metrics.
            **train_args: Optional train args parsed to model_object.train

        """

        if train_metrics is None:
            self.train_metrics = DEFAULT_TRAIN_METRICS
        self.train_metrics_results = {}
        for model in self.model_objects:
            train_fit, train_true = model.train(train_data,
                                                sync_steps=sync_steps,
                                                **train_args)
            for metric_name, metric in self.train_metrics.items():
                if metric_name in self.train_metrics_results:
                    self.train_metrics_results[metric_name].append(metric(train_true, train_fit))
                else:
                    self.train_metrics_results[metric_name] = [metric(train_true, train_fit)]

    def predict_models(self,
                       predict_data: np.ndarray,
                       sync_steps: int = 0,
                       predict_metrics: None | dict[str, Callable] = None,
                       **pred_args
                       ) -> None:
        """Predict with all the models in the self.model_objects list.

        Calls for each model_object in self.model_objects list the
        model_object.predict(predict_data, sync_steps=sync_steps, **pred_args) function,
        which returns pred and pred_true.

        For each model_object and each predict_metric given in the predict_metrics dictionary, a
        value is saved in self.predict_metrics_results.

        Args:
            predict_data: The prediction/test data of shape: (train_steps + sync_steps, sys_dim).
            sync_steps: The steps used for synchronization.
            predict_metrics: A dictionary of functions which are the prediction metrics.
            **pred_args: Optional predict args parsed to model_object.predict
        """

        if predict_metrics is None:
            self.predict_metrics = DEFAULT_PREDICT_METRICS
        self.predict_metrics_results = {}
        for model in self.model_objects:
            pred, pred_true = model.predict(predict_data,
                                                sync_steps=sync_steps,
                                                **pred_args)
            for metric_name, metric in self.predict_metrics.items():
                if metric_name in self.predict_metrics_results:
                    self.predict_metrics_results[metric_name].append(metric(pred_true, pred))
                else:
                    self.predict_metrics_results[metric_name] = [metric(pred_true, pred)]

    def return_pandas(self) -> pd.DataFrame:
        """Return the train and predict metrics as a pandas dataframe.

        Returns:
            Pandas Dataframe with the columns: "TRAIN {train metric}" and "PREDICT
            {predict metric}".
        """
        train_df = pd.DataFrame.from_dict(self.train_metrics_results)
        train_df.rename(mapper=lambda x: f"TRAIN {x}", inplace=True, axis=1)
        predict_df = pd.DataFrame.from_dict(self.predict_metrics_results)
        predict_df.rename(mapper=lambda x: f"PREDICT {x}", inplace=True, axis=1)
        return pd.concat([train_df, predict_df], axis=1)


class StatisticalModelTester():
    """
    A Class to statistically test a prediction model.
    """

    def __init__(self):
        self.error_function = lambda y_pred, y_test: measures.error_over_time(y_pred, y_test, distance_measure="L2",
                                                                              normalization="root_of_avg_of_spacedist_squared")
        self.error_threshhold = None

        self.model_creation_function = lambda: None

        self._output_flag_synonyms = utilities._SynonymDict()
        self._output_flag_synonyms.add_synonyms(0, ["full"])
        self._output_flag_synonyms.add_synonyms(1, ["valid_times"])
        self._output_flag_synonyms.add_synonyms(2, ["valid_times_median_quartile"])
        self._output_flag_synonyms.add_synonyms(3, ["error"])
        self._output_flag = None

        self.results = None  # trajectories

    def set_error_function(self, error_function):
        self.error_function = error_function

    def set_model_creation_function(self, model_creation_function):
        '''
        :param model_creation_function: A function
        :return:
        '''
        self.model_creation_function = model_creation_function

    def set_model_prediction_function(self, model_prediction_function):
        '''
        :param model_prediction_function:
        :return:
        '''
        self.model_prediction_function = model_prediction_function

    def do_ens_experiment(self, nr_model_realizations, x_pred_list, output_flag="full", save_example_trajectory=False,
                          time_it=False, **kwargs):
        print("      Starting ensemble experiment...")
        print("      output_flag: ", output_flag)

        if time_it:
            t = time.time()

        self._output_flag = self._output_flag_synonyms.get_flag(output_flag)
        nr_of_time_intervals = len(x_pred_list)

        if self._output_flag in (1, 2):
            self.error_threshhold = kwargs["error_threshhold"]
            valid_times = np.zeros((nr_model_realizations, nr_of_time_intervals))

        for i in range(nr_model_realizations):
            print(f"Realization: {i + 1}/{nr_model_realizations} ...")
            model = self.model_creation_function()
            for j, x_pred in enumerate(x_pred_list):
                y_pred, y_test = self.model_prediction_function(x_pred, model)
                if self._output_flag == 0:
                    if i == 0 and j == 0:
                        predict_steps, dim = y_pred.shape
                        results = np.zeros((nr_model_realizations, nr_of_time_intervals, 2, predict_steps, dim))
                    results[i, j, 0, :, :] = y_pred
                    results[i, j, 1, :, :] = y_test
                elif self._output_flag in (1, 2):
                    valid_times[i, j] = measures.valid_time_index(self.error_function(y_pred, y_test),
                                                                  self.error_threshhold)
                elif self._output_flag == 3:
                    if i == 0 and j == 0:
                        errors = np.zeros((nr_model_realizations, nr_of_time_intervals, predict_steps))
                    errors[i, j, :] = self.error_function(y_pred, y_test)

        to_return = []

        if self._output_flag == 0:
            to_return.append(results)
        elif self._output_flag == 1:
            to_return.append(valid_times)
        elif self._output_flag == 2:
            median = np.median(valid_times)
            first_quartile = np.quantile(valid_times, 0.25)
            third_quartile = np.quantile(valid_times, 0.75)
            to_return.append(np.array([median, first_quartile, third_quartile]))
        elif self._output_flag == 3:
            to_return.append(errors)

        if time_it:
            elapsed_time = time.time() - t
            to_return.append(elapsed_time)
        if save_example_trajectory:
            example_trajectory = (y_pred, y_test)
            to_return.append(example_trajectory)
        return to_return

    def do_ens_experiment_internal(self, nr_model_realizations, x_pred_list, **kwargs):
        nr_of_time_intervals = len(x_pred_list)

        for i in range(nr_model_realizations):
            print(f"Realization: {i + 1}/{nr_model_realizations} ...")
            model = self.model_creation_function(**kwargs)
            for j, x_pred in enumerate(x_pred_list):
                y_pred, y_test = self.model_prediction_function(x_pred, model)
                if i == 0 and j == 0:
                    predict_steps, dim = y_pred.shape
                    results = np.zeros((nr_model_realizations, nr_of_time_intervals, 2, predict_steps, dim))
                results[i, j, 0, :, :] = y_pred
                results[i, j, 1, :, :] = y_test

        self.results = results

    def get_error(self, results=None, mean=False):
        if results is None:
            results = self.results

        n_ens = results.shape[0]
        n_interval = results.shape[1]
        n_pred_steps = results.shape[3]

        error = np.zeros((n_ens, n_interval, n_pred_steps))
        for i_ens in range(n_ens):
            for i_interval in range(n_interval):
                y_pred = results[i_ens, i_interval, 0, :, :]
                y_test = results[i_ens, i_interval, 1, :, :]

                error[i_ens, i_interval, :] = self.error_function(y_pred, y_test)
        if mean:
            error_mean = np.mean(error, axis=(0, 1))
            return error_mean
        return error

    def get_valid_times(self, error=None, results=None, mean=False, error_threshhold=None):
        if error is None:
            if results is None:
                results = self.results
            error = self.get_error(results)

        if error_threshhold is None:
            error_threshhold = self.error_threshhold

        n_ens = error.shape[0]
        n_interval = error.shape[1]

        valid_times = np.zeros((n_ens, n_interval))
        for i_ens in range(n_ens):
            for i_interval in range(n_interval):
                valid_times[i_ens, i_interval] = measures.valid_time_index(error[i_ens, i_interval, :],
                                                                           error_threshhold)

        if mean:
            valid_times_mean = np.mean(valid_times)
            return valid_times_mean
        return valid_times


class StatisticalModelTesterSweep(StatisticalModelTester):
    """

    """
    def __init__(self):
        super().__init__()

        self.input_parameters = None
        self.results_sweep = []
        self.error_sweep = []
        self.valid_times_sweep = []

        self.nr_model_realizations = None
        self.nr_of_time_intervals = None

    def _dict_of_vals_to_dict_of_list(self, inp):
        return {key: ((val,) if not type(val) in (list, tuple) else tuple(val)) for key, val in inp.items()}

    def _unpack_parameters(self, **parameters):
        list_of_params = []
        parameters_w_list = self._dict_of_vals_to_dict_of_list(parameters)

        keys = parameters_w_list.keys()
        values = parameters_w_list.values()

        for x in list(itertools.product(*values)):
            d = dict(zip(keys, x))
            list_of_params.append(d)

        return list_of_params

    def do_ens_experiment_sweep(self, nr_model_realizations, x_pred_list, results_type="trajectory", error_threshhold=0.4,
                                **parameters):
        # saves the whole trajectories or valid times (hopefully less memory consuming)
        self.input_parameters = parameters
        self.nr_model_realizations = nr_model_realizations
        self.nr_of_time_intervals = x_pred_list.shape[0]
        list_of_params = self._unpack_parameters(**parameters)
        for i, params in enumerate(list_of_params):
            print(f"Sweep: {i + 1}/{len(list_of_params)}")
            print(params)
            self.do_ens_experiment_internal(nr_model_realizations, x_pred_list, **params)

            if results_type == "trajectory":
                self.results_sweep.append((params, self.results.copy()))
            elif results_type == "validtimes":
                self.error_threshhold = error_threshhold
                vt = self.get_valid_times(results=self.results, error_threshhold=error_threshhold)
                self.valid_times_sweep.append((params, vt))

    def get_results_sweep(self):
        return self.results_sweep

    def get_error_sweep(self):
        # works if self.results_sweep is already populated
        error_sweep = []
        for params, results in self.results_sweep:
            error_sweep.append((params, self.get_error(results)))

        self.error_sweep = error_sweep
        return error_sweep

    def get_valid_times_sweep(self, error_threshhold=None):
        # works if self.results_sweep is already populated
        if error_threshhold is None:
            error_threshhold = self.error_threshhold

        else:
            self.error_threshhold = error_threshhold

        valid_times_sweep = []
        for params, results in self.results_sweep:
            vt = self.get_valid_times(results=results, error_threshhold=error_threshhold)
            valid_times_sweep.append((params, vt))

        self.valid_times_sweep = valid_times_sweep
        return valid_times_sweep

    def save_sweep_results(self, name="default_name", path=None, results_type="trajectory"):
        if path is None:
            repo_path = pathlib.Path(__file__).parent.resolve().parents[0]
            path = pathlib.Path.joinpath(repo_path, "results")
            print(path)

        if results_type == "trajectory":
            if len(self.results_sweep) == 0:
                raise Exception("no trajectory results yet")
            else:
                data = self.results_sweep
        elif results_type == "error":
            if len(self.error_sweep) == 0:
                raise Exception("no error results yet")
            else:
                data = self.error_sweep
        elif results_type == "validtimes":
            if len(self.valid_times_sweep) == 0:
                raise Exception("no valid_times results yet")
            else:
                data = self.valid_times_sweep

        # check if there is a file with that name already:
        if f"{name}.hdf5" in os.listdir(path):
            i = 0
            name_temp = name
            while f"{name_temp}.hdf5" in os.listdir(path):
                i += 1
                name_temp = f"{name}{i}"

            name = f"{name}{i}"
        print(name)
        file_path = pathlib.Path.joinpath(path, f"{name}.hdf5")
        print(file_path)
        with h5py.File(file_path, "w") as f:
            runs_group = f.create_group("runs")
            i = 1
            for params, results in data:
                dset = runs_group.create_dataset(f"trajectory_{i}", data=results)
                for key, val in params.items():
                    try:
                        dset.attrs[key] = val
                    except Exception as e:
                        print(e)
                        dset.attrs[key] = str(val)
                i += 1

            # sweep_info_group = f.create_group("sweep_info")
            # for key, val in self._dict_of_vals_to_dict_of_list(self.input_parameters).items():
            #     sweep_info_group.create_dataset(key, data=val)

    def get_valid_times_df(self, **kwargs):
        if len(self.valid_times_sweep) == 0 or "error_threshhold" in kwargs.keys():
            self.get_valid_times_sweep(**kwargs)

        df_vt = None

        for params, valid_times in self.valid_times_sweep:
            vt_mean = np.mean(valid_times)
            vt_std = np.std(valid_times)
            vt_median = np.median(valid_times)
            input = {key: (val, ) for key, val in params.items()}
            input["valid_times_mean"] = (vt_mean, )
            input["valid_times_median"] = (vt_median, )
            input["valid_times_std"] = (vt_std, )
            input["error_threshhold"] = (self.error_threshhold, )
            input["nr_model_realizations"] = (self.nr_model_realizations, )
            input["nr_of_time_intervals"] = (self.nr_of_time_intervals, )

            if df_vt is None:
                df_vt = pd.DataFrame.from_dict(input)

            else:
                df_vt = pd.concat([df_vt, pd.DataFrame.from_dict(input)])

        return df_vt

    def plot_error(self, ax=None, figsize=(15, 8)):
        if ax is None:
            plt.figure(figsize=figsize)
            ax = plt.gca()

        if len(self.error_sweep) == 0:
            self.get_error_sweep()

        for params, error in self.error_sweep:
            error_mean = np.mean(error, axis=(0, 1))
            ax.plot(error_mean, label=f"{params}")
        ax.legend()
