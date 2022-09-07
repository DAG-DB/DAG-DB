import os


COMMENT = "From Optuna on STE, MAX_SIZE=None; best result sticking with Gumbel"

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

	'N_EPOCHS': 1_000,
	'BATCH_SIZE': None,
	'LOG2_BATCH_SIZE': 4,
	'BATCH_SHUFFLE': True,
	'N_SAMPLES': 50	,

	'MODE': 'max_dag_test_only',
	'AFAS_SCHEDULER': None,
	'MINIZINC_SOLVER': None,
	'f_LINEAR_LAYERS': tuple(),
	'f_BIAS': False,

	'p_LAYER': 'STE',
	'STREAMLINE': False,
	'THRESHOLD': 0.,

	'h_LINEAR_LAYERS': tuple(),
	'h_NULL': 0.0109843638170302,

	'h_LR': 0.0535900780042863,
	'h_OPTIMIZER': f'torch.optim.Adam(overall_net.h_net.parameters(), lr=h.h_LR)',
	'f_LR': 0.00238538719734531,
	'f_OPTIMIZER': f'torch.optim.Adam(overall_net.f_net.parameters(), lr=h.f_LR)',

	'NOISE_TEMPERATURE': 0.0443157790358656,
	'NOISE_DISTRIBUTION': 'LogisticNoiseDistribution()',
	'LAMBDA': 1,

	'LOSS_FN': 'torch.nn.MSELoss()',

	'z_RHO': 0.526864469836609,
	'z_MU': 0.0119699206451764,
	'z_REGULARIZER': 'NoTearsZRegularizer(h.D, h.z_RHO, h.z_MU, c.DEVICE)',
	'LR_SCHEDULER': None,

	'DATA_CATEGORY': 'test',
	'DAGS': '030_ER2_gaussian_ev_2022-08-12_BST_09:55:21,293_create_dags',
}
