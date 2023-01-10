"""File to create parameter-sweep and ensemble based experiments for RC predictions. """

from __future__ import annotations

import itertools
import os
import pathlib
from datetime import date
from typing import Callable, Any

import numpy as np
import pandas as pd

import src.esn_src.data_preprocessing as datapre
import src.esn_src.utilities as utilities
import src.esn_src.measures as measures

def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """The mean squared error between y_true and y_pred.

    Args:
        y_true: The true time series of shape (time steps, sysdim).
        y_pred: The predicted time series of shape (time steps, sysdim).

    Returns:
        A float representing the MSE.
    """
    return (np.linalg.norm(y_true - y_pred, axis=1)**2).mean()

def dummy_metric(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Dummy metric which returns just a 1.
    # TODO: Should not be needed.
    Args:
        y_true: The true time series of shape (time steps, sysdim).
        y_pred: The predicted time series of shape (time steps, sysdim).

    Returns:
        1.
    """
    return 1

def valid_time(y_true: np.ndarray,
               y_pred: np.ndarray,
               dt: float = 1.0,
               lle: float = 1.0,
               error_threshold: float = 0.4,
               error_norm: str = "root_of_avg_of_spacedist_squared") -> float:
    """Calculate the valid time in lyapunov times for a prediction, given y_pred and y_true.

    If you only want the index: use dt = lle = 1.0.
    If you want the time: use dt = dt, lle = 1.0.
    If you want the vt in units of lyapunov times use: dt=dt, lle=lle.

    Args:
        y_true: The true time series of shape (time steps, sysdim).
        y_pred: The predicted time series of shape (time steps, sysdim).
        dt: Time step dt.
        lle: The lyapunov exponent.
        error_threshold: Threshold for max error.
        error_norm: The norm to use for error.
    Returns:
        The valid time in units of lyapunov time (if dt and lle are properly set).
    """
    error_series = measures.error_over_time(y_pred, y_true,
                                            normalization=error_norm)
    return measures.valid_time_index(error_series,
                                     error_threshold=error_threshold) * dt * lle

def valid_time_centered_error(y_true: np.ndarray,
                              y_pred: np.ndarray,
                              dt: float = 1.0,
                              lle: float = 1.0,
                              error_threshold: float = 0.4,
                              ):
    """
    The same as valid time, but uses a specific error_norm which additionaly centeres the
    dominator in the error norm, compared to "root_of_avg_of_spacedist_squared".
    """
    return valid_time(y_true,
                      y_pred,
                      dt=dt,
                      lle=lle,
                      error_threshold=error_threshold,
                      error_norm="root_of_avg_of_centered_spacedist_squared")

DEFAULT_TRAIN_METRICS = {
    "MSE": mse
}

DEFAULT_PREDICT_METRICS = {
    "MSE": mse,
    "VT": valid_time,
    # "VT_centered": valid_time_centered_error,
}

METRIC_PREFIX = "M "
# TIME_PREFIX = "TCPU "
PART_PREFIX = "i "
PARAMETER_PREFIX = "P "

class PredModelValidator:
    """Class to validate a time-series prediction model (as ESN).

    Validate means: a built model trains and predicts on multiple data sections and the metrics
    are saved.

    """
    def __init__(self, built_model: Any) -> None:
        """Initialize.

        Add the built model as the argument.

        Args:
            built_model: A "built_model", which must have the methods:
                - built_model.train(train_data: np.ndarray, sync_steps: int, **opt_train_args)
                  -> returns: (train_fit, train_true). Where train_fit and train_true both have
                  the shape (train_steps - sync_steps - 1, data dimension).
                - built_model.predict(predict_data: np.ndarray, sync_steps: int, **opt_pred_args).
                  -> returns: (prediction, pred_true). Where prediction and pred_true both have
                  the shape (pred_steps - sync_steps - 1, data dimension).
        """
        self.built_model = built_model

        self.train_metrics: None | dict[str, Callable] = None
        self.validate_metrics: None | dict[str, Callable] = None
        self.test_metrics: None | dict[str, Callable] = None

        self.train_metrics_results: None | dict[str, list] = None
        self.validate_metrics_results: None | dict[str, list[list]] = None
        self.test_metrics_results: None | dict[str, list[list]] = None

        # TODO: Check if to add time?
        # self.train_cpu_time: None | list = None
        # self.validate_cpu_time: None | list = None
        # self.test_cpu_time: None | list = None

        self.n_train_sects: int | None = None  # sections
        self.n_test_sects: int | None = None  # sections

        self.metrics_df: None | pd.DataFrame = None

    def train_validate_test(
        self,
        train_data_list: list[np.ndarray],
        validate_data_list_of_lists: list[list[np.ndarray]],
        train_sync_steps: int = 0,
        validate_sync_steps: int = 0,
        opt_train_args: None | dict[str, Any] = None,
        opt_pred_args: None | dict[str, Any] = None,
        train_metrics: None | dict[str, Callable] = None,
        validate_metrics: None | dict[str, Callable] = None,
        test_data_list: np.ndarray | list[np.ndarray] | None = None,
        test_sync_steps: int = 0,
        test_metrics: None | dict[str, Callable] = None,
        opt_train_metrics_args: None | dict[str, dict[str, Any]] | None = None,
        opt_validate_metrics_args: None | dict[str, dict[str, Any]] | None = None,
        opt_test_metrics_args: None | dict[str, dict[str, Any]] | None = None,
        ) -> pd.DataFrame:
        """Train validate and test a built model.

        The test set is optional.

        Args:
            train_data_list: A list of train sections: [train_data1, train_data2, ...].
                each train_data entry has the shape (train steps, data dimension).
            validate_data_list_of_lists: A list of lists of validation data (which has to be
                predicted. The outer list corresponds to the training sections, i.e.
                [pred_sub_list_1, pred_sub_list_2, ...], and each pred_sub_list_i is
                pred_sub_list_i = [pred_data_sub_i_1, pred_data_sub_i_2, ...].
            train_sync_steps: The steps used to sync before training starts.
            validate_sync_steps: The steps used to sync before validation before prediction.
            opt_train_args: Optional arguments parsed to train function as a dictionary.
            opt_pred_args: Optional arguments parsed to predict function as a dictionary.
            train_metrics: A dictionary of train metrics as: {"Metric1": metric_func_1, ...}
                each metric_func takes the arguments (y_true, y_pred) and returns a float.
            validate_metrics: The same as train_metrics but for the validation.
                It is different, since for ESN at least, one uses different Metrics for training
                and prediction.
            test_data_list: Optional list of testing data. For each training section, all testing
                data sections are tested and measures with the test_metrics.
            test_sync_steps: The steps used to sync before testing (before prediction).
            test_metrics: The same as validate_metrics but for the testing. Also time-series
            predict metrics.

        Returns:
            Pandas DataFrame with the results. Dataframe has the columns:
            - train sect (an integer for the train section used).
            - val sect (an integer for the validation section used for that train section).
            - If testing data is given: test sect (an integer for the test section.)
                There is an entry for every combination of train sect, val sect and testing sect.
            - The train metrics starting with "TRAIN <metric xyz>", ..
            - The validation metrics starting with "VALIDATE <metric xyz>", ..
            - The test metrics starting with "TEST <metric xyz>", ..
        """

        # Check if nr of train and validate sections match:
        if len(train_data_list) != len(validate_data_list_of_lists):
            raise ValueError("train_data_list and validate_data_list_of_lists must "
                             "have the same length. ")

        # Save the number of training and testing sections.
        self.n_train_sects = len(train_data_list)
        if test_data_list is not None:
            self.n_test_sects = len(test_data_list)

        # Set the metrics for train, validate and test if they are None:
        if train_metrics is None:
            self.train_metrics = DEFAULT_TRAIN_METRICS
        else: # TODO: maybe check if metrics are valid?
            self.train_metrics = train_metrics
        if validate_metrics is None:
            self.validate_metrics = DEFAULT_PREDICT_METRICS
        else:
            self.validate_metrics = validate_metrics
        if test_metrics is None:
            self.test_metrics = DEFAULT_PREDICT_METRICS
        else:
            self.test_metrics = test_metrics

        # set optional train and pred args if they are None:
        if opt_train_args is None:
            opt_train_args = {}
        if opt_pred_args is None:
            opt_pred_args = {}

        # set optional train, validate and test metric args if they are None:
        if opt_train_metrics_args is None:
            opt_train_metrics_args = {}
        if opt_validate_metrics_args is None:
            opt_validate_metrics_args = {}
        if opt_test_metrics_args is None:
            opt_test_metrics_args = {}

        # Initialize the metric results:
        self.train_metrics_results = {}
        self.validate_metrics_results = {}
        self.test_metrics_results = {}

        # Iterate through the train sections:
        for i_train_section, train_data in enumerate(train_data_list):

            # TRAIN ON TRAIN SECTION.
            # TODO: maybe add time?
            # start = time.process_time()
            train_fit, train_true = self.built_model.train(train_data,
                                                           sync_steps=train_sync_steps,
                                                           **opt_train_args)
            # end = time.process_time()


            for metric_name, metric in self.train_metrics.items():
                if metric_name in opt_train_metrics_args:
                    opt_args = opt_train_metrics_args[metric_name]
                else:
                    opt_args = {}
                metric_result = metric(train_true, train_fit, **opt_args)
                if metric_name in self.train_metrics_results:
                    self.train_metrics_results[metric_name].append(metric_result)
                else:
                    self.train_metrics_results[metric_name] = [metric_result]

            # GET ALL VALIDATION SECTIONS FOR THAT SPECIFIC TRAIN SECTION
            validate_list_for_train_section = validate_data_list_of_lists[i_train_section]

            # PREDICT TRAINED MODEL ON ALL VALIDATION SECTIONS FOR THAT SPECIFIC TRAIN SECTION
            for validate_data in validate_list_for_train_section:
                pred, pred_true = self.built_model.predict(validate_data,
                                                     sync_steps=validate_sync_steps,
                                                     **opt_pred_args)

                for metric_name, metric in self.validate_metrics.items():
                    if metric_name in opt_validate_metrics_args:
                        opt_args = opt_validate_metrics_args[metric_name]
                    else:
                        opt_args = {}
                    metric_result = metric(pred_true, pred, **opt_args)
                    if metric_name in self.validate_metrics_results:
                        if len(self.validate_metrics_results[metric_name]) == i_train_section + 1:
                            self.validate_metrics_results[metric_name][i_train_section].append(metric_result)
                        else:
                            self.validate_metrics_results[metric_name].append([metric_result])
                    else:
                        self.validate_metrics_results[metric_name] = [[metric_result]]

            # PREDICT THE TRAINED MODEL ON ALL TEST SECTIONS FOR THAT SPECIFIC TRAIN SECTION
            if test_data_list is not None:
                for test_data in test_data_list:
                    pred, pred_true = self.built_model.predict(test_data,
                                                    sync_steps=test_sync_steps,
                                                    **opt_pred_args)

                    for metric_name, metric in self.test_metrics.items():
                        if metric_name in opt_test_metrics_args:
                            opt_args = opt_test_metrics_args[metric_name]
                        else:
                            opt_args = {}
                        metric_result = metric(pred_true, pred, **opt_args)
                        if metric_name in self.test_metrics_results:
                            if len(self.test_metrics_results[metric_name]) == i_train_section + 1:
                                self.test_metrics_results[metric_name][i_train_section].append(metric_result)
                            else:
                                self.test_metrics_results[metric_name].append([metric_result])
                        else:
                            self.test_metrics_results[metric_name] = [[metric_result]]

        # Calculate pandas DF out of metric dicts:
        self.metric_dfs = self.metrics_to_pandas()

        # Return pandas metrics DF
        return self.metric_dfs


    def metrics_to_pandas(self) -> pd.DataFrame:
        """Transform the metrics dictionaries to a pandas dataframe.
        """

        train_metric_prefix = f"{METRIC_PREFIX}TRAIN "
        validate_metric_prefix = f"{METRIC_PREFIX}VALIDATE "
        test_metric_prefix = f"{METRIC_PREFIX}TEST "

        train_section_name = f"{PART_PREFIX}train sect"
        validate_section_name = f"{PART_PREFIX}val sect"
        test_section_name = f"{PART_PREFIX}test sect"

        train_df = pd.DataFrame.from_dict(self.train_metrics_results)
        train_df.rename(mapper=lambda x: f"{train_metric_prefix}{x}",
                        inplace=True,
                        axis=1)
        train_df.insert(0, column=train_section_name, value=train_df.index)

        validate_df = None
        for i_train_sec in range(self.n_train_sects):
            sub_validate_dict = {k: v[i_train_sec] for k, v in
                                 self.validate_metrics_results.items()}

            sub_validate_df = pd.DataFrame.from_dict(sub_validate_dict)
            sub_validate_df.rename(mapper=lambda x: f"{validate_metric_prefix}{x}",
                                   inplace=True,
                                   axis=1)
            sub_validate_df.insert(0, column=validate_section_name, value=sub_validate_df.index)
            sub_validate_df.insert(0, column=train_section_name, value=i_train_sec)

            if validate_df is None:
                validate_df = sub_validate_df
            else:
                validate_df = pd.concat([validate_df, sub_validate_df])
        metrics_df = pd.merge(train_df, validate_df, on=train_section_name)

        if self.n_test_sects is not None:
            test_df = None
            for i_train_sec in range(self.n_train_sects):
                sub_test_dict = {k: v[i_train_sec] for k, v in
                                     self.test_metrics_results.items()}

                sub_test_df = pd.DataFrame.from_dict(sub_test_dict)
                sub_test_df.rename(mapper=lambda x: f"{test_metric_prefix}{x}", inplace=True, axis=1)
                sub_test_df.insert(0, column=test_section_name, value=sub_test_df.index)
                sub_test_df.insert(0, column=train_section_name, value=i_train_sec)

                if test_df is None:
                    test_df = sub_test_df
                else:
                    test_df = pd.concat([test_df, sub_test_df])
            metrics_df = pd.merge(metrics_df, test_df, on=train_section_name)

        cols = metrics_df.columns.tolist()
        cols_new = cols.copy()

        if self.n_test_sects is not None:
            sec_cols = [train_section_name, validate_section_name, test_section_name]
        else:
            sec_cols = [train_section_name, validate_section_name]

        for section in sec_cols[::-1]:
            index = cols_new.index(section)
            cols_new.pop(index)
            cols_new.insert(0, section)

        metrics_df = metrics_df[cols_new]
        return metrics_df


class PredModelEnsembler:
    def __init__(self):
        self.master_seed: int | None = None
        self.seeds: list[int] | None = None
        self.rng: np.random.Generator | None = None
        self.n_ens: int | None = None

        self.metrics_df_list: None | list[pd.DataFrame] = None
        self.metrics_df_all: None | pd.DataFrame = None

        self.built_models: list | None = None

    def build_models(self,
                     model_class,
                     build_args: None | dict[str, Any] = None,
                     n_ens: int = 1,
                     seed: int = 0) -> None:
        """Function to build an ensemble of models each with a different seed.

        TODO: Add model metrics? E.g. measure some network quantities?.

        Saves all built models to the list: self.built_models.

        Args:
            model_object: The model object, as for example ESN_normal()
            build_args: The arguments parsed to model_object.build(**build_args)
            n_ens: The ensemble size.
            seed: The random seed.
        """
        self.n_ens = n_ens
        self.master_seed = seed
        self.rng = np.random.default_rng(self.master_seed)
        self.seeds = self.rng.integers(0, 10000000, size=self.n_ens)

        self.built_models = []
        for seed in self.seeds:
            with utilities.temp_seed(seed):
                model_object = model_class()
                build_args = utilities.remove_invalid_args(model_object.build, build_args)
                model_object.build(**build_args)
                self.built_models.append(model_object)

    def train_validate_test(
            self,
            train_data_list: list[np.ndarray],
            validate_data_list_of_lists: list[list[np.ndarray]],
            train_sync_steps: int = 0,
            validate_sync_steps: int = 0,
            opt_train_args: None | dict[str, Any] = None,
            opt_pred_args: None | dict[str, Any] = None,
            train_metrics: None | dict[str, Callable] = None,
            validate_metrics: None | dict[str, Callable] = None,
            test_data_list: np.ndarray | list[np.ndarray] | None = None,
            test_sync_steps: int = 0,
            test_metrics: None | dict[str, Callable] = None,
            opt_train_metrics_args: None | dict[str, dict[str, Any]] | None = None,
            opt_validate_metrics_args: None | dict[str, dict[str, Any]] | None = None,
            opt_test_metrics_args: None | dict[str, dict[str, Any]] | None = None,
            ) -> pd.DataFrame:
        """See PredModelValidator.train_validate_test docstring.
        -> The same just for an ensemble of models.
        """
        i_ens_name = f"{PART_PREFIX}ens"

        self.metrics_df_list = []
        for i_ens in range(self.n_ens):
            built_model = self.built_models[i_ens]
            validator = PredModelValidator(built_model)
            metrics_df = validator.train_validate_test(
                train_data_list=train_data_list,
                validate_data_list_of_lists=validate_data_list_of_lists,
                train_sync_steps=train_sync_steps,
                validate_sync_steps=validate_sync_steps,
                opt_train_args=opt_train_args,
                opt_pred_args=opt_pred_args,
                train_metrics=train_metrics,
                validate_metrics=validate_metrics,
                test_data_list=test_data_list,
                test_sync_steps=test_sync_steps,
                test_metrics=test_metrics,
                opt_train_metrics_args=opt_train_metrics_args,
                opt_validate_metrics_args=opt_validate_metrics_args,
                opt_test_metrics_args=opt_test_metrics_args
            )
            metrics_df.insert(0, i_ens_name, i_ens)
            self.metrics_df_list.append(metrics_df)

        self.metrics_df_all = self.combine_metric_dfs()
        return self.metrics_df_all

    def combine_metric_dfs(self):
        """Combine all dataframes from PredModelValidator, into one single df.

        Returns:
            The resulting dataframe with an additional column: "i_ens" numbering the ensemble
            members.
        """
        metrics_df_all = pd.concat(self.metrics_df_list, ignore_index=True)
        return metrics_df_all


class PredModelSweeper:
    """Class to sweep parameters of a model and for each parameter setting create a metrics_df.

    -> See PredModelEnsembler to see how to create a metrics_df.

    """
    def __init__(self,
                 parameter_transformer: Callable[[dict[str, float | int | str | bool]], tuple]
                 ) -> None:
        """Set the parameter transformer function.

        Args:
            parameter_transformer: The parameter_transformer is a function that takes a
            dict of parameters (with str keys and float/int/str/bool values) as its argument
            and outputs:
                build_models_args = {ALL THE ARGUMENTS USED TO RUN PredModelEnsembler.build_models}
                train_validate_test_args = {ALL THE ARGUMENTS USED TO RUN
                                            PredModelEnsembler.train_validate_test_args}
        """

        self.set_parameter_transformer(parameter_transformer)
        self.metric_results: list[tuple[dict[str, float | int | str], pd.DataFrame]] | None = None

    def set_parameter_transformer(
            self,
            parameter_transformer: Callable[[dict[str, float | int | str | bool]], tuple[dict, dict]]
        ) -> None:
        """Set the parameter transformer function.
        """

        self.parameter_transformer = parameter_transformer

    def dict_to_dict_of_tuples(self, inp: dict[str, Any]) -> dict[str, tuple]:
        """Function to turn a dictionary of objects into a dictionary of tuples of the obects.

        Args:
            inp: A dictionary like: {"a": 1.0, "b": [1, 2, 3], "c": (2, 4)}.

        Returns:
            The transformed dictionary: {"a": (1.0, ), "b": (1, 2, 3), "c": (2, 4)}.
        """
        return {key: ((val,) if not type(val) in (list, tuple) else tuple(val))
                for key, val in inp.items()}

    def unpack_parameters(self, parameters: dict[str, Any]) -> list[dict[str, Any]]:
        """Unpack parameters dictionary which may contain lists, into a list of dicts.

        Args:
            parameters: A dictionary like: {"a": 1.0, "b": [1, 2, 3], "c": (2, 4)}.

        Returns:
            list_of_params as [{"a": 1.0, "b": 1, "c": 2}, {"a": 1.0, "b": 2, "c": 2}, ...].
        """
        list_of_params = []
        parameters_w_list = self.dict_to_dict_of_tuples(parameters)

        keys = parameters_w_list.keys()
        values = parameters_w_list.values()

        for x in list(itertools.product(*values)):
            d = dict(zip(keys, x))
            list_of_params.append(d)

        return list_of_params

    def sweep(self, parameters: dict[str, Any]
              ) -> pd.DataFrame:
        """Sweep some parameters, and for each point in parameter space to an ensemble test.

        save results internally to self.metric_results:
            A list of 2-tuples. The list has an element for each parameter-space-point,
            the tuple is: (parameters for point, metric_df).

        Args:
            parameters:
                A pretty free form parameters dictionary.

        Returns:
            Pandas dataframe containing all the information.
        """

        list_of_params = self.unpack_parameters(parameters)
        self.metric_results = []
        for i, params in enumerate(list_of_params):
            print(f"Sweep: {i + 1}/{len(list_of_params)}")
            print(params)

            build_models_args, train_validate_test_args = self.parameter_transformer(params)

            ensembler = PredModelEnsembler()
            ensembler.build_models(**build_models_args)
            metrics_df = ensembler.train_validate_test(**train_validate_test_args)

            self.metric_results.append((params, metrics_df))

        results_df = self.to_big_pandas(self.metric_results)
        return results_df

    @staticmethod
    def to_big_pandas(metric_results) -> pd.DataFrame:
        """ Turn metric_results into a big Dataframe with all the parameters as columns.

        Returns:
            Pandas DataFrame which has additional parameters columns compared
             to ensembler metrics_df df.
        """
        big_df: pd.DataFrame | None = None
        for (params, metric_df) in metric_results:
            nr_rows = len(metric_df.index)
            params_df = pd.DataFrame.from_dict({f"{PARAMETER_PREFIX}{k}": [v] * nr_rows for k, v
                                                in params.items()})

            out = pd.concat([params_df, metric_df],
                            join="inner",
                            axis=1)
            if big_df is None:
                big_df = out
            else:
                big_df = pd.concat([big_df, out], axis=0, ignore_index=True)
        return big_df

    def to_hdf5(self, file_path: None | str = None):
        """Save self.metric_results data to hdf5 file.
        # TODO: Not really used.
        USAGE: One can optionally save the results to a hdf5 file, which then can be loaded,
        using the from_hdf5 function.

        Save the parameters as attributes, and save the pandas df using pd.HDFStore.

        get back self.metric_results by running "from_hdf5".
        """
        if file_path is None:
            file_path = "default_file_path.h5"

        data = self.metric_results

        store = pd.HDFStore(file_path)
        for i_data, data_point in enumerate(data):
            params, metric_df = data_point
            group_name = f"sweep_{i_data}"
            store.put(group_name, metric_df)
            store.get_storer(group_name).attrs.params = params
        store.close()

    @staticmethod
    def from_hdf5(file_path: str
                  ) -> list[tuple[Any, object]]:
        """Reverse operation to "to_hdf5", i.e. read a saved h5 file and turn it into
        self.metric_results.
        # TODO: Not really used.
        Args:
            file_path: The path to the .h5 file

        Returns:
            A list of 2-tuples. The list has an element for each parameter-space-point,
            the tuple is: (parameters for point, metric_df).
        """
        metric_results = []

        store = pd.HDFStore(file_path, mode="r")
        for key in store.keys():
            params = store.get_storer(key).attrs["params"]
            metric_df = pd.read_hdf(store, key=key)
            metric_results.append((params, metric_df))
        store.close()

        return metric_results


def save_pandas_to_pickles(df: pd.DataFrame,
                           name: str | None = None,
                           directory: None | str = None) -> str:
    """Save pandas DF to pickle.

    If directory is None: Save to relative directory of file.
    The filename in directory will be: "name__date__nr"
    nr is a 3-digit number xxx counting up, if it's the same name and date.
    date will be dd_mm_yyyy.

    Args:
        df: The dataframe to save.
        name: The base name identified of the file.
        directory: The directory to save it to.
    """

    # Create a path "directory" if it does not exist.
    if directory is not None:
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
    else:
        directory = pathlib.Path().absolute()

    if name is None:
        name = "default_name"

    today = date.today().strftime("%d_%m_%Y")
    file_name_no_nr = f"{name}__{today}"

    # Other files in directory with same name and date:
    potential_other_files = [x for x in os.listdir(directory) if x.startswith(file_name_no_nr)]

    if len(potential_other_files) == 0:
        file_name = f"{file_name_no_nr}__000.pkl"
    else:
        previous_highest_nr = max([int(x.split(".")[0].split("__")[-1]) for x in
                                   potential_other_files])

        file_name = f"{file_name_no_nr}__{previous_highest_nr + 1:03d}.pkl"

    file_path = pathlib.Path.joinpath(directory, file_name)
    df.to_pickle(file_path)
    return file_path


def metrics_from_hdf5(file_path: str ) -> list[tuple[Any, object]]:
        """Reverse operation to "sweeper.to_hdf5", i.e. read a saved h5 file and turn it into
        self.metric_results.
        # TODO: Not really used.
        Args:
            file_path: The path to the .h5 file

        Returns:
            A list of 2-tuples. The list has an element for each parameter-space-point,
            the tuple is: (parameters for point, metric_df).
        """
        metric_results = []

        store = pd.HDFStore(file_path, mode="r")
        for key in store.keys():
            params = store.get_storer(key).attrs["params"]
            metric_df = pd.read_hdf(store, key=key)
            metric_results.append((params, metric_df))
        store.close()

        return metric_results


def results_to_param_sweep(metric_results: list[tuple[Any, object]],
                          stripped: bool = True) -> dict[str, Any]:
    """Reverse operation to create the parameter sweep dict from the metric results.

    # TODO not really used.

    Take the output from ProdModelSweeper.sweep (-> metric_results) and use it here
    to create the sweep parameter dict again: {param1: [1, 2, 3], param2: 4, ...}

    Args:
        metric_results: The output from ProdModelSweeper.sweep().
        stripped:

    Returns:
        The sweep parameter dict, i.e. the input to ProdModelSweeper.sweep(parameters).
    """

    params_dict = {}
    for data_point in metric_results:
        params = data_point[0]
        for key, val in params.items():
            if key in params_dict.keys():
                if stripped:
                    if val not in params_dict[key]:
                        params_dict[key].append(val)
                else:
                    params_dict[key].append(val)
            else:
                params_dict[key] = [val, ]
    return params_dict


def time_series_creator(sys_obj: Any,
                        t_train_disc: int,
                        t_train_sync: int,
                        t_train: int,
                        t_validate_disc: int,
                        t_validate_sync: int,
                        t_validate: int,
                        t_test_disc: int | None = None,
                        t_test_sync: int | None = None,
                        t_test: int | None = None,
                        n_train_sects: int = 1,
                        n_validate_sects: int = 1,
                        n_test_sects: int = 1,
                        normalize_and_center: bool = True,
                        ) -> tuple[list, list] | tuple[list, list, list]:
    """Create data from the sys_obj, be defining all the sections used for train/validate/test.

    The sections are defined as:

        ##### for the first of n_train_sects: #####
        ([t_train_disc] -> [t_train_sync]-> [t_train]) ->
        ([t_validate_disc] -> [t_validate_sync] -> [t_validate]) ->
        ([t_validate_disc] -> [t_validate_sync] -> [t_validate]) ->
        ... (repeat n_validate_sects times).

        ##### for the second of n_train_sects: #####
        ([t_train_disc] -> [t_train_sync]-> [t_train]) * 2 ->
        ([t_validate_disc]->[t_validate_sync]->[t_validate]) ->
        ([t_validate_disc]->[t_validate_sync]->[t_validate]) ->
        ... (repeat n_validate_sects times).

        ...

        ##### for the last of n_train_sects: get the test data: #####
        ([t_train_disc] -> [t_train_sync]-> [t_train]) * n_train_sects ->
        ([t_validate_disc]->[t_validate_sync]->[t_validate] ->) * n_validate_sects ->
        ([t_test_disc] -> [t_test_sync] -> [t_test] ->) * n_test_sects

    Args:
        sys_obj: The system object which has a method sys_obj.simulate(time_steps: int).
        t_train_disc: The discard steps before the training.
        t_train_sync: The number of steps used for sync before actual training (driving).
        t_train: The number of train steps.
        t_validate_disc: The number of steps discarded after training and before validation.
        t_validate_sync: The number of steps used to synchronize before validation (prediction).
        t_validate: The number of steps to validate.
        t_test_disc: The number of steps discarded after validation and before testing.
        t_test_sync: The number of steps used to synchronize before testing (prediction)
        t_test: The number of steps to test.
        n_train_sects: The number of disjoint train sections.
        n_validate_sects: The number of disjoint validate sections.
        n_test_sects: The number of train sections.

    Returns:
        (train_data_list, validate_data_list_of_lists, test_data_list) or
        (train_data_list, validate_data_list_of_lists), as used in
        PredModelValidator.train_validate_test() and PredModelEnsembler.train_validate_test().
    """
    # Calculate the total number of time steps:
    total_steps = n_train_sects * (t_train_disc + t_train + t_train_sync) + \
                  n_validate_sects * (t_validate_disc + t_validate_sync + t_validate)

    # If testing is required also add the test steps.
    if t_test_disc is not None and t_test_sync is not None and t_test is not None:
        create_test_data = True
        total_steps +=  n_test_sects * (t_test_disc + t_test_sync + t_test)
    else:
        create_test_data = False

    # Simulate all the time steps:
    data = sys_obj.simulate(total_steps)

    # Normalize and center:
    # Note: This is note ML-perfect since we are preprocessing not only based on
    # training data.
    if normalize_and_center:
        data = datapre.scale_and_shift(data, return_scale_shift=False)

    # Initialize the output lists:
    train_data_list = []
    validate_data_list_of_lists = []

    # Populate the lists:
    for i_tr in range(n_train_sects):

        # Get the training data for each training section:
        train_start_index = i_tr * (t_train_disc + t_train_sync + t_train) + t_train_disc
        train_end_index = train_start_index +  t_train_sync + t_train
        train_data_list.append(data[train_start_index: train_end_index, :])

        # Initialize the validate_sub_list:
        validate_sub_list = []

        # Populate the validate_sub_list:
        for i_v in range(n_validate_sects):
            validate_start_index = train_end_index + \
                                   i_v * (t_validate_disc + t_validate_sync + t_validate) + \
                                   t_validate_disc
            validate_end_index = validate_start_index + t_validate_sync + t_validate
            validate_sub_list.append(data[validate_start_index: validate_end_index, :])

        # Populate validate_data_list_of_lists:
        validate_data_list_of_lists.append(validate_sub_list)

    # Create test data (optionally):
    if create_test_data:
        test_data_list = []
        for i_test in range(n_test_sects):
            test_start_index = validate_end_index + t_test_disc + \
                               i_test * (t_test_disc + t_test_sync + t_test)
            test_end_index = test_start_index + t_test_sync + t_test
            test_data_list.append(data[test_start_index: test_end_index, :])

    if create_test_data:
        return train_data_list, validate_data_list_of_lists, test_data_list
    else:
        return train_data_list, validate_data_list_of_lists
