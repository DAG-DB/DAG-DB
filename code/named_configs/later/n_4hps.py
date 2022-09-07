import os


COMMENT = "Test best MAX_SIZE=None Gumbel values with Logistic"

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
SAVE_TO_ODS = False
VAL_LOSS_FROM_DAG = True  # Relevant only if mode is 'max_dag_test_only'
WANDB = False
WANDB_PROJECT = 'Thetas'
GRAD_LOG = False

# Optuna - putting these as strings in hyperparameters ensures properly
# recorded in .ods
N_TRIALS = 1
LOAD_OPTUNA_STUDY = None
OPTUNA_STRINGS = {'None': None, 'True': True, 'False': False}

TQDM = False

# Hyperparameters

hyperparameters = {
	'GRAPH_TYPE': 'ER2',
	'D': 30,

	'MAX_SIZE': None,

	'SEM_NOISE': 'gaussian_ev',

	'N': 1200,
	'VAL': 1 / 6,

	'N_EPOCHS': 200,  # 1_000,
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
	'h_NULL': None,  # 0.00011373901985660933,

	'h_LR': None,
	'h_OPTIMIZER': f'torch.optim.Adam(overall_net.h_net.parameters(), lr=h.h_LR)',
	'f_LR': 0.3720066380139224,
	'f_OPTIMIZER': f'torch.optim.Adam(overall_net.f_net.parameters(), lr=h.f_LR)',

	'NOISE_TEMPERATURE': None,  # 0.8785929148606203,
	'NOISE_DISTRIBUTION': 'LogisticNoiseDistribution()',
	'LAMBDA': None,  # 27.13959516077541,

	'LOSS_FN': 'torch.nn.MSELoss()',

	'REGULARIZER': None,
	'z_RHO': 0,  # 0.157451211201,
	'z_MU': 0,  # 0.00120806965197976,
	'z_REGULARIZER': 'NoTearsZRegularizer(h.D, h.z_RHO, h.z_MU, c.DEVICE)',
	'LR_SCHEDULER': None,

	'DATA_CATEGORY': 'test',
	'DAGS': '030_ER2_gaussian_ev_2022-08-12_BST_09:55:21,293_create_dags',
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