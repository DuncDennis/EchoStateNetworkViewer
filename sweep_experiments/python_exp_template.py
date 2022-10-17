"""A test experiment template to show how to start an ensemble experiment.

# Some notes about the experiment:

"""


from __future__ import annotations
from datetime import datetime

import src.esn_src.simulations as sims
import src.esn_src.esn as esn
import src.esn_src.utilities as utilities
import src.ensemble_src.sweep_experiments as sweep

# TRACKED PARAMETERS:
parameters={
    # Data Meta:
    "system": ["Lorenz63"],
    "dt": 0.01,
    "normalize_and_center": True,

    # Data steps (used in sweep.time_series_creator(ARGS):
    "t_train_disc": [100],
    "t_train_sync": [100],
    "t_train": [1000],
    "t_validate_disc": [500],
    "t_validate_sync": [100],
    "t_validate": [1000],
    "n_train_sects": [3],
    "n_validate_sects": [3],

    # ESN Meta:
    "esn_type": ["ESN_normal"],

    # ESN Build (data that is fed into esn.build(ARGS):
    "r_dim": [300],
    "n_rad": [0.1],
    "n_avg_deg": [6.0],
    "n_type_opt": ["erdos_renyi"],
    "r_to_r_gen_opt": ["output_bias", "linear"],
    "act_fct_opt": ["tanh"],
    "node_bias_opt": ["random_bias"],
    "bias_scale": [0.1],
    "w_in_opt": ["random_sparse"],
    "w_in_scale": [1.0],
    "input_noise_scale": [0.0],

    # Experiment parameters:
    "seed": [308],
    "n_ens": 2
}


# PARAMETER TO ARGUMENT TRANSFOMER FUNCTION:
def parameter_transformer(parameters: dict[str, float | int | str]):
    """Transform the parameters to be usable by PredModelEnsembler.

    Args:
        parameters: The parameter dict defining the sweep experiment.
            Each key value pair must be like: key is a string, value is either a string,
            int or float.

    Returns:
        All the data needed for PredModelEnsembler.
    """
    p = parameters

    # System:
    sys_class = sims.SYSTEM_DICT[p["system"]]
    sys_args = utilities.remove_invalid_args(sys_class.__init__, p)
    sys_obj = sys_class(**sys_args)

    # Create Data:
    ts_creator_args = utilities.remove_invalid_args(sweep.time_series_creator, p)
    train_data_list, validate_data_list_of_lists = sweep.time_series_creator(sys_obj,
                                                                             **ts_creator_args)
    # Build ESN:
    esn_class = esn.ESN_DICT[p["esn_type"]]
    build_args = utilities.remove_invalid_args(esn_class.build, p)
    build_args["x_dim"] = train_data_list[0].shape[1]

    # Experiment args:
    n_ens = p["n_ens"]
    seed = p["seed"]


    build_models_args = {"model_class": esn_class,
                         "build_args": build_args,
                         "n_ens": n_ens,
                         "seed": seed}

    train_validate_test_args = {
        "train_data_list": train_data_list,
        "validate_data_list_of_lists": validate_data_list_of_lists,
        "train_sync_steps": p["t_train_sync"],
        "validate_sync_steps": p["t_validate_sync"],
        # "opt_validate_metrics_args": {"VT": {"dt": p["t_validate_sync"], }}
    }

    return build_models_args, train_validate_test_args

# Set up Sweeper.
sweeper = sweep.PredModelSweeper(parameter_transformer)

# Print start time:
start = datetime.now()
start_str = start.strftime("%Y-%m-%d %H:%M:%S")
print(f"Start: {start_str}")

# Sweep:
results_df = sweeper.sweep(parameters)

# End time:
end = datetime.now()
end_str = end.strftime("%Y-%m-%d %H:%M:%S")
print(f"End: {end_str}")

# Ellapsed time:
diff = end - start
diff_str = diff
print(f"Time difference: {diff_str}")

# Save results:
print("Saving...")
file_path = sweep.save_pandas_to_pickles(df=results_df,
                                         name="first_python_test")
print(f"Saved to: {file_path}")
print("FINISHED! ")
