import os


COMMENT = "For look inside an SF2"

# File Structure
DATA_FOLDER = os.path.join('..', 'data')
LATENTS_FOLDER = os.path.join('..', 'latents')
LOGS_FOLDER = os.path.join('..', 'logs')
MODEL_FOLDER = os.path.join('..', 'models')
RESULTS_FOLDER = os.path.join('..', 'results')

# Device
DEVICE = None  # "cpu" # "cuda"  # "cpu"  # None

# Reporting
LATENTS = False
MATRIX_REPORT = False
LABEL = True
SAVE_TO_ODS = True
VAL_LOSS_FROM_DAG = True  # Relevant only if mode is 'max_dag_test_only'
WANDB = True
WANDB_PROJECT = 'Thetas'
GRAD_LOG = False
SAVE_RESULTS = False
DAG_EVERY_EPOCH = True

# Optuna - putting these as strings in hyperparameters ensures properly
# recorded in .ods
N_TRIALS = 1
LOAD_OPTUNA_STUDY = None
OPTUNA_STRINGS = {'None': None, 'True': True, 'False': False}

TQDM = False

# Hyperparameters

hyperparameters = {
	'GRAPH_TYPE': None,
	'D': None,

	'MAX_SIZE': None,

	'SEM_NOISE': 'gaussian_ev',

	'N': 1200,
	'VAL': 1 / 6,

	'N_EPOCHS': 1_000,
	'BATCH_SIZE': None,
	'LOG2_BATCH_SIZE': 3,  # v size 32
	'BATCH_SHUFFLE': True,
	'N_SAMPLES': 47,  # v 10

	'MODE': 'max_dag_test_only',
	# MODE from 'lt_p_matrix', 'lt_p_vector_argsort_in_std_autograd',
	# 'max_dag', 'max_dag_test_only', 'max_dag_schedule'
	'AFAS_SCHEDULER': None,
	'MINIZINC_SOLVER': None,
	'f_LINEAR_LAYERS': tuple(),
	'f_BIAS': False,

	'p_LAYER': None,
	'STREAMLINE': False,
	'THRESHOLD': 0.,  # STANDARD

	'h_LINEAR_LAYERS': tuple(),
	'h_NULL': 0.00011373901985660933,  # v 5e-3

	'h_LR': 0.0014196699981655287 / 0.8785929148606203,  #
	'h_OPTIMIZER': f'torch.optim.Adam(overall_net.h_net.parameters(), lr=h.h_LR)',
	'f_LR': 0.3720066380139224,  # v 0.01
	'f_OPTIMIZER': f'torch.optim.Adam(overall_net.f_net.parameters(), lr=h.f_LR)',

	'NOISE_TEMPERATURE': 0.8785929148606203,  # v 0.16
	'NOISE_DISTRIBUTION': 'LogisticNoiseDistribution()',
	'LAMBDA': 27.13959516077541,  # v 5.69

	'LOSS_FN': 'torch.nn.MSELoss()',

	'ALPHA': None,
	'BETA': None,
	'REGULARIZER': None,
	'z_RHO': 0.157451211201,  #  v 0.02
	'z_MU': 0.00120806965197976,  # v 0.22
	'z_REGULARIZER': 'NoTearsZRegularizer(h.D, h.z_RHO, h.z_MU, c.DEVICE)',
	'LR_SCHEDULER': None,

	'DATA_CATEGORY': 'look',
	 # Good:
	# 'DAGS': '010_SF2_gaussian_ev_2022-08-20_BST_11:39:28,507_create_dags',
	 # Bad:
	'DAGS': '010_ER4_gaussian_ev_2022-08-20_BST_11:38:06,145_create_dags',
}


""" Options:
'NOISE_TEMPERATURE':
	[0.1, 0.3, 1., 3., 10.]  
	
'NOISE_DISTRIBUTION':
	'None'
	'GumbelNoiseDistribution()'  # This may be best 2022-06-15_BST_10:57:47,304
	'SumOfGammaNoiseDistribution(k=5)'
	
'SHUFFLE_BATCHES': [False, True]
'BATCH_SIZE': [10, 32, 100, 300, 1_000]
	# True, 32 may be best 2022-06-15_BST_11:42:15,382 

'LOSS_FN':
	'order_loss'
	'arrangement_loss'
	'supervised_order_loss'
	'permutation_comparison_loss'
	
'LR_SCHEDULER':
	None
	'torch.optim.lr_scheduler.CyclicLR(base_lr=1e-4, '\
			 'max_lr=0.01, cycle_momentum=False, step_size_up=64)'
	'torch.optim.lr_scheduler.CyclicLR(base_lr=1e-3, '\
		'max_lr=0.1, cycle_momentum=False, step_size_up=2000)',
"""