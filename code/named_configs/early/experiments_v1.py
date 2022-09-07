import os


COMMENT = "Experiments v1"

# File Structure
MODEL_FOLDER = os.path.join('..', 'models')
LOGS_FOLDER = os.path.join('..', 'logs')
LATENTS_FOLDER = os.path.join('..', 'latents')
DATA_FOLDER = os.path.join('..', 'data')

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
N_TRIALS = 1  # 50  # 100
OPTUNA_STRINGS = {'None': None, 'True': True, 'False': False}

TQDM = False

# Hyperparameters

hyperparameters = {
	'GRAPH_TYPE': 'ER2',
	'D': 30,

	'MAX_SIZE': 72,

	'SEM_NOISE': 'gaussian_ev',

	'N': 1200,
	'VAL': 1 / 6,

	'N_EPOCHS': 1_000,
	'BATCH_SIZE': 32,
	'BATCH_SHUFFLE': True,
	'N_SAMPLES': 10,

	'MODE': 'max_dag_test_only',
	# MODE from 'lt_p_matrix', 'lt_p_vector_argsort_in_std_autograd',
	# 'max_dag', 'max_dag_test_only', 'max_dag_schedule'
	'AFAS_SCHEDULER': None,
	'MINIZINC_SOLVER': None,
	'f_LINEAR_LAYERS': tuple(),
	'f_BIAS': False,

	'p_LAYER': None, ################
	'STREAMLINE': False,
	'THRESHOLD': 0.,

	'h_LINEAR_LAYERS': tuple(),
	'h_NULL': 0.006,  # 0.00581130989059376,

	'h_LR': None,  # 0.000655054828435191,
	'h_OPTIMIZER': f'torch.optim.Adam(overall_net.h_net.parameters(), lr=h.h_LR)',
	'f_LR': 0.12,  # 0.0117744218586073,
	'f_OPTIMIZER': f'torch.optim.Adam(overall_net.f_net.parameters(), lr=h.f_LR)',

	'NOISE_TEMPERATURE': None,  # 0.16213500214616,
	'NOISE_DISTRIBUTION': 'GumbelNoiseDistribution()',
	'LAMBDA': None,  # 5.69792034959308,

	'LOSS_FN': 'torch.nn.MSELoss()',

	'ALPHA': None,
	'BETA': None,
	'REGULARIZER': None,
	'z_RHO': 0,  # 0.0280244755372679,
	'z_MU': 0,  # 0.220018032970804,
	'z_REGULARIZER': 'NoTearsZRegularizer(h.D, h.z_RHO, h.z_MU, c.DEVICE)',
	'LR_SCHEDULER': None,

	'DATA_CATEGORY': 'train',
	'DAGS': '030_ER2_gaussian_ev_2022-08-02_BST_16:51:14,282_generate_dags',
}
