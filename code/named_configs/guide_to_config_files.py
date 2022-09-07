""" A guide to the contents of DAG-DB config files for running
code/main_dag.py.  For configs to create synthetic DAGs/data see
code/named_configs/create_dags.py and code/data_management/create_dags.py

Except when indicated, e.g. for `SAVE_RESULTS', all fields should be included.
"""

import os


COMMENT = "..."  # Adding a comment is optional

# File Structure       # Standard - do no alter
DATA_FOLDER = os.path.join('..', 'data')
LATENTS_FOLDER = os.path.join('..', 'latents')
LOGS_FOLDER = os.path.join('..', 'logs')
MODEL_FOLDER = os.path.join('..', 'models')
RESULTS_FOLDER = os.path.join('..', 'results')

# Device
DEVICE = None  # "cpu", "cuda" or None.  None selects cuda if available,
 # otherwise selects cpu

# Reporting
# The following are old means of tracking information about Z, potentially
# giving more info that the change_adj_count in logs.  May not work well now
LATENTS = False
MATRIX_REPORT = False
LABEL = True

SAVE_TO_ODS = False  # If true, save the  true and predicted adj matrices plus
 # f net's weights to .ods file in logs/  Can only save for one DAG in a run
VAL_LOSS_FROM_DAG = True  # Relevant only if mode is 'max_dag_test_only',
 # for which should be set to True
WANDB = False  # Whether to use wandb to record metrics
WANDB_PROJECT = 'Thetas'  # Set to the wandb project name you want to use
SAVE_RESULTS = False  # If absent or True, save metrics for each DAG in a
# .csv file in results/  File will be named after the config file,
# and results will be appended to any existing such file

# Optuna
N_TRIALS = 1  # To do multiple Optuna trials set to integer > 1
LOAD_OPTUNA_STUDY = None  # If string load and resume that Optuna study from
 # logs/  If None, start new study
OPTUNA_STRINGS = {'None': None, 'True': True, 'False': False}  # Leave
 # unchanged.  In Optuna categorical trial settings use the keys instead of
# the values.  This makes sure they are properly recorded in logs/*.ods files

# NOTE: Optuna and wandb cannot be used together

TQDM = False  # Whether to show TQDM counter in training

# Hyperparameters

hyperparameters = {
	'GRAPH_TYPE': 'ER2',  #  For synthetic graphs, should be of form ERk or
	 #  SFk for positive integer k. See thesis sec. 3.5.1 for explanation.
	 # With real data, set to None
	'D': 30,  # Number of nodes (for synthetic or real graph)

	'MAX_SIZE': None,  # Maximum size constraint on the number of edges in
	 # the predicted latent graph: a positive integer or None.

	'SEM_NOISE': 'gaussian_ev',  # The type of SEM noise for synthetic DAGs.
	 # See thesis sec 2.1.  Always set to `gaussian_ev' for thesis, but
	 # 'gaussian_nv', 'exponential', 'gumbel' also possible.

	'N': 1200,  # The number of data points in the training plus the
	 # (somewhat redundant) validation set
	'VAL': 1 / 6,  # The proportion of 'N' that is in the validation set

	'N_EPOCHS': 1_000,

	# Batch size: exactly one of the following two should be a positive
	# integer, the other None.
	'BATCH_SIZE': None,
	'LOG2_BATCH_SIZE': 3,  # To help Optuna trials

	'BATCH_SHUFFLE': True,  # Whether to shuffle batches between each epoch
	'N_SAMPLES': 47,  # The number of perturb--and--MAP samples to make on
	 # the forward pass

	'MODE': 'max_dag_test_only',
	""" 'MODE' is one of:
	- 'max_dag_test_only': the standard setting;
	- 'max_dag': generates a DAG with each batch in training and uses for f.  
	    Usually too slow;
	-  'max_dag_schedule': experimental setting for scheduling which 
	epoch to generate DAG.  Did not prove useful;
	- 'lt_p_matrix' and 'lt_p_vector_argsort_in_std_autograd': variants of 
	the first alternative architecture discussed in sec 5.6
	- 'd_p' the second alternative architecture discussed in sec 5.6 (the 
	real vectors to binary vectors version)
	"""
	
	'AFAS_SCHEDULER': None,  # Experimental: set to None. Was used to enable
	 # the scheduler in code/learn_infer/afas_solver.py
	'MINIZINC_SOLVER': None,  # Experimental: set to None.  Was used to
	 # enable the MiniZinc maximum DAG solver in
	 # code/learn_infer/minizinc_solver.py
	'f_LINEAR_LAYERS': tuple(),  # Width of hidden layers in the f neural
	 # net.  Always empty tuple for thesis as targeting linear SEMs
	'f_BIAS': False,  #  Should the f NN have a bias.  Always False for thesis

	'p_LAYER': None,  # Set p_LAYER to one of None (for I-MLE), 'STE',
	# 'SFE'.  Blackbox (BB) is done by setting to None, setting N_SAMPLES=1,
	# NOISE_TEMPERATURE=1 and NOISE_DISTRIBUTION='NoNoise()'
	'STREAMLINE': False,  # Experimental.  Keep set to False as did not
	 # prove useful
	'THRESHOLD': 0.,  # Keep as 0, except for `stochastic
	# bias' experiment described in sec 9.2

	# If h_NULL is a number then Theta will be a learned parameter (always
	# the case for the thesis).  If h_NULL is None, then h_LINEAR_LAYERS
	# sets the width of hidden layers in a neural net outputting Theta.
	'h_LINEAR_LAYERS': tuple(),
	'h_NULL': 0.00011373901985660933,

	# Learning rates for h (i.e. Theta) and f, plus their optimizers
	'h_LR': 0.00161584503375007,
	'h_OPTIMIZER': f'torch.optim.Adam(overall_net.h_net.parameters(), lr=h.h_LR)',
	'f_LR': 0.3720066380139224,
	'f_OPTIMIZER': f'torch.optim.Adam(overall_net.f_net.parameters(), lr=h.f_LR)',

	# Noise temperature and distribution.  Distribution must be either
	# 'LogisticNoiseDistribution()', 'GumbelNoiseDistribution()',
	# 'SumOfGammaNoiseDistribution()' or 'NoNoise()',
	'NOISE_TEMPERATURE': 0.8785929148606203,
	'NOISE_DISTRIBUTION': 'LogisticNoiseDistribution()',

	'LAMBDA': 27.13959516077541,  # The Domke lambda hyperparameter

	'LOSS_FN': 'torch.nn.MSELoss()',  # Loss function: always MSE

	'z_RHO': 0.157451211201,  # Weight of DAG regularizer (sec 5.2)
	'z_MU': 0.00120806965197976,  # Weight of sparsity regularizer (sec 5.2)
	'z_REGULARIZER': 'NoTearsZRegularizer(h.D, h.z_RHO, h.z_MU, c.DEVICE)',
      # Leave z_REGULARIZER as this
	'LR_SCHEDULER': None,  # Experimental: not yet found useful as enough
	 # hyperparameters to tune without this

	'DATA_CATEGORY': 'test',  # the sub-directory of data/ directory to look
	 # in for data
	'DAGS': '030_ER2_gaussian_ev_2022-08-12_BST_09:55:21,293_create_dags',
	 # The directory or file for the specific set of DAGs or DAG.  EXCEPT
	# when the config file is being used by code/data_management/create_dags.py
	# to create new synthetic DAGs and data.  Then it should be a list of
	# random seeds, one for each DAG/dataset to be created.
}
