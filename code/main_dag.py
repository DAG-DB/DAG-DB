""" The main program for DAG-DB.  Acts as junction box for key settings.
It creates the MAP solver and then loops over the DAGs to predict.  For
each DAG to predict, it sets up learn_infer.dag_learner.OverallNet and calls
learn_infer.dag_learner.dag_learner
"""

import os

import joblib
import optuna
import numpy as np
import torch
from torch import Tensor

import lib.ml_utilities as mlu
from discrete_backprop.dag_imle import imle
from discrete_backprop.dag_sfe_reinforce import sfe
from discrete_backprop.dag_ste import ste
from lib.ml_utilities import c, h
from discrete_backprop.target import TargetDistribution
from learn_infer.dag_learner import Mode, dag_learner, OverallNet
from data_management.get_save_data import generate_dag_data, load_dag_data
from utils.er_sf_sizes import er_size, sf_size
from utils.hf_optimizer import hfOptimizer
from utils.loss_fns import NullRegularizer
from utils.metric_handlers import TrialStore

if c.WANDB:
    assert c.N_TRIALS <= 1  # To avoid multiple overwrites
    import wandb
    from utils.wandb_util import finalise_wandb


# Used dynamically:
from discrete_backprop.noise import GumbelNoiseDistribution, LogisticNoiseDistribution, \
    NoNoise, SumOfGammaNoiseDistribution
from learn_infer.afas_solver import RegularSolverScheduler
from utils.loss_fns import GOLEMLossEV, GOLEMLossNV, Golem1ER4Loss, \
    MeanSquareSigmoidRegularizer  # These were tried but not useful
from utils.z_regularizers import GolemZRegularizer, McKayZRegularizer, \
    NoTearsZRegularizer  # The Golem... and McKay... were tried but not useful

# Needed for engine='odf' in df.to_excel:
import odf


def objective(trial):
    """
    Main loop over DAG data determined by config file, predicting DAG for each

    Acts as junction box for key settings other than wandb and Optuna.
    It creates the MAP solver and learn_infer.dag_learner.OverallNet and calls
    learn_infer.dag_learner.dag_learner

    :param trial: Optuna trial or None if no Optuna
    :return: mean of last epoch's nSHD_c over the DAGs predicted
    """
    # Determine key settings

    n_real_runs = h.N_REAL_RUNS if (
            (h.DATA_CATEGORY == 'real') and ('N_REAL_RUNS' in h)) else 1
    dataset = load_dag_data(h.DATA_CATEGORY, h.DAGS, n_real_runs)
    trial_store = TrialStore()
    for key, value in h.items():
        if isinstance(value, str) and value.startswith('trial.'):
            if key in [
                    'LOSS_FN', 'OPTIMIZER', 'LR_SCHEDULER',
                    'REGULARIZER', 'z_REGULARIZER',  # To allow parameters
                # to be determined first
                    'h_OPTIMIZER', 'f_OPTIMIZER',
                    'AFAS_SCHEDULER']:
                raise NotImplementedError  # As these use strings even
                # without trial; and as they often use other h values
            trial_store.append(key)  # To allow later trials to
            # re-evaluate the trial string
            h[key] = eval(value)
            if h[key] in c.OPTUNA_STRINGS:
                h[key] = c.OPTUNA_STRINGS[h[key]]

    d = h.D

    if 'LOG2_BATCH_SIZE' in h:
        if h.LOG2_BATCH_SIZE is None:
            batch_size = h.BATCH_SIZE
        else:
            assert h.BATCH_SIZE is None
            batch_size = 2 ** h.LOG2_BATCH_SIZE
    else:
        batch_size = h.BATCH_SIZE

    if isinstance(h.MAX_SIZE, float):
        if h.GRAPH_TYPE.startswith('ER'):
            h.MAX_SIZE = er_size(h.D, int(h.GRAPH_TYPE[2:]), h.MAX_SIZE)
        elif h.GRAPH_TYPE.startswith('SF'):
            if ('OFFSET' in h) and (h.OFFSET is not None):
                h.MAX_SIZE = sf_size(
                    h.D, int(h.GRAPH_TYPE[2:]), offset=h.OFFSET)
            elif ('MULTIPLIER' in h) and (h.MULTIPLIER is not None):
                h.MAX_SIZE = sf_size(
                    h.D, int(h.GRAPH_TYPE[2:]), multiplier=h.MULTIPLIER)
        else:
            raise NotImplementedError

    noise_distribution = eval(h.NOISE_DISTRIBUTION)
    target_distribution = TargetDistribution(
        beta=h.LAMBDA * h.NOISE_TEMPERATURE,  # Putting temp. tau != 1 => need to adjust I-MLE eq. 11
        do_gradient_scaling=True
    )

    noise_temperature = h.NOISE_TEMPERATURE
    theta_noise_temperature = target_noise_temperature = noise_temperature

    mode = Mode(d, h.MODE)

    p_layer = None
    sfe_flag = False
    match mode.mode:
        case 'd_p':
            if h.p_LAYER is None:
                p_layer = imle
                # p_layer = d_p_imle
            elif h.p_LAYER == 'STE':
                raise NotImplementedError
        case _:
            if h.p_LAYER is None:
                p_layer = imle
            elif h.p_LAYER == 'STE':
                p_layer = ste
            elif h.p_LAYER == 'SFE':
                p_layer = sfe
                sfe_flag = True

    # Creates the MAP solver

    solver = mode.solver(
        d, h.N_SAMPLES,
        max_size=h.MAX_SIZE,
        threshold=h.THRESHOLD,
        mode=mode
    )

    if mode.mode == 'max_dag_test_only' and h.MINIZINC_SOLVER == 'Final':
        from learn_infer.minizinc_solver import MiniZincSolverFinal
        minizinc_solver = MiniZincSolverFinal(
            d, 1, max_size=h.MAX_SIZE, threshold=h.THRESHOLD, mode=mode)
    else:
        minizinc_solver = None

    def torch_solver(theta: Tensor) -> Tensor:
        """
        From I-MLE.  Thanks to Pasquale Minervini

        Wrapper around the `solver` function from discrete_backprop.

        :param theta: PyTorch tensor Theta
        :return: torch Tensor for Z
        """
        theta = theta.detach().cpu().numpy()
        z = solver(theta)
        return torch.tensor(
            z, requires_grad=False, device=c.DEVICE, dtype=torch.float32)

    @p_layer(
        target_distribution=target_distribution,
        noise_distribution=noise_distribution,
        nb_samples=h.N_SAMPLES,
        theta_noise_temperature=theta_noise_temperature,
        target_noise_temperature=target_noise_temperature,
        streamline=h.STREAMLINE
    )
    def imle_solver(theta: Tensor) -> Tensor:
        return torch_solver(theta)

    #  Loop over DAGs to be predicted
    key_metrics = list()
    wandbs = np.full(h.N_EPOCHS, dict(), dtype=dict)
    for i, data in enumerate(dataset):
        # Get the true DAG (for metrics) and the data
        X, B = data
        adjacency_true = B.astype(bool).astype(int)

        mlu.save_to_ods(i, adjacency_true, 'adj_true')
        mlu.save_to_ods(i, B, 'w_true')

        size_true = adjacency_true.sum()
        mlu.log(f'\n\nDAG {i}: {size_true=}\t{X.var()=:.4f}')

        # Create the learn_infer.dag_learner.OverallNet and associated
        # optimizers, schedulers, loss function, regularizer
        torch.cuda.empty_cache()
        overall_net = OverallNet(
            d, h.N_SAMPLES, h.BATCH_SIZE, c.DEVICE, solver, torch_solver,
            imle_solver, h_linear_layers=h.h_LINEAR_LAYERS,
            f_linear_layers=h.f_LINEAR_LAYERS,
            h_null=h.h_NULL, f_bias=h.f_BIAS, mode=mode,
        ).to(device=c.DEVICE)

        h_optimizer = eval(h.h_OPTIMIZER)
        f_optimizer = eval(h.f_OPTIMIZER)
        optimizer = hfOptimizer(overall_net, h_optimizer, f_optimizer)
        if h.LR_SCHEDULER is not None:
            lr_scheduler = dict()
            lr_scheduler['h'] = mlu.get_lr_scheduler(
                trial, optimizer.h_opt, h.LR_SCHEDULER)
            lr_scheduler['f'] = mlu.get_lr_scheduler(
                trial, optimizer.f_opt, h.LR_SCHEDULER)
        else:
            lr_scheduler = None
        if h.AFAS_SCHEDULER is not None:
            assert mode.mode == 'max_dag_schedule'
            afas_scheduler = eval(h.AFAS_SCHEDULER)
        else:
            afas_scheduler = None
        if h.z_REGULARIZER is not None:
            z_regularizer = eval(h.z_REGULARIZER)
        else:
            z_regularizer = NullRegularizer()

        # Call learn_infer.dag_learner.dag_learner
        metrics = dag_learner(
            i,
            X,
            adjacency_true,
            overall_net,
            trial,
            device=c.DEVICE,
            val_prop=h.VAL,
            batch_size=batch_size,
            n_epochs=h.N_EPOCHS,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            batch_shuffle=h.BATCH_SHUFFLE,
            z_regularizer=z_regularizer,
            solver = solver,
            wandbs=wandbs,
            afas_scheduler=afas_scheduler,
            minizinc_solver=minizinc_solver,
            trial_store=trial_store,
            noise_temperature=noise_temperature,
            sfe_flag=sfe_flag,
        )

        if (not hasattr(c, 'SAVE_RESULTS')) or c.SAVE_RESULTS:
            mlu.save_metrics(i, metrics)
        key_metrics.append(metrics['nshd_c'])

    if c.WANDB:
        finalise_wandb(wandbs)

    trial_store.deploy()  # So in later trials the trial string is re-evaluated

    return np.mean(key_metrics)


def save_study(study, trial):
    """
    Save Optuna study

    :param study: Optuna study to be saved
    :param trial: Mandatory under Optuna rules
    """
    joblib.dump(study, os.path.join(c.LOGS_FOLDER, mlu.LOG_FILESTEM + '.pkl'))


def main():
    """
    Determines if using wandb and/or Optuna.  (Best used separately.)  Then
    calls objective(trial) to execute training

    :return: Optuna study, or None
    """
    if c.WANDB:
        wandb.login()
        wandb.init(project=c.WANDB_PROJECT)
    if c.N_TRIALS > 1:
        if hasattr(c, 'LOAD_OPTUNA_STUDY') and (
                c.LOAD_OPTUNA_STUDY is not None):
            study_location = os.path.join(
                c.LOGS_FOLDER, c.LOAD_OPTUNA_STUDY + '.pkl')
            study = joblib.load(study_location)
        else:
            study = optuna.create_study(
                directions=["minimize"],
                pruner=optuna.pruners.MedianPruner()
            )  # Pruning will only be done wrt the first DAG

        # From https://optuna.readthedocs.io/en/v1.5.0/reference/logging.html
        optuna.logging.enable_propagation()  # Propagate logs to the root logger.
        optuna.logging.disable_default_handler()  # Stop showing logs in sys.stderr.

        study.optimize(
            objective, n_trials=c.N_TRIALS, timeout=None, callbacks=[save_study])

        df = study.trials_dataframe()
        mlu.log('study.trials_dataframe()=')
        mlu.log(df)

        df.to_excel(
            os.path.join(c.LOGS_FOLDER, mlu.LOG_FILESTEM + '.ods'), engine='odf')
    else:
        # noinspection PyTupleAssignmentBalance
        last_mean_nshd_c = objective(None)
        mlu.log(f'\n{last_mean_nshd_c=:.4f}')
        study = None

    if c.WANDB:
        mlu.log(
            f'\nWandb project {c.WANDB_PROJECT}, run name is'
            f' {wandb.run.name}\n')

    mlu.end_log_message()

    return study


if __name__ == "__main__":
    study = main()
