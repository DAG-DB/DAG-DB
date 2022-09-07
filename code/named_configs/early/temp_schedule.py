import os


COMMENT = "Best Optuna trial values of max dag test only, hNull, with z " \
		  "regularizer"

# File Structure
MODEL_FOLDER = os.path.join('..', 'models')
LOGS_FOLDER = os.path.join('..', 'logs')
LATENTS_FOLDER = os.path.join('..', 'latents')

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
	'D': 30,
	'SIZE_MIN': 20,
	'SIZE_MAX': 100,

	'MAX_SIZE': 72,
	# 'MAX_SIZE': 'trial.suggest_categorical("max_size", [72, "None"])',

	'SEM_NOISE': 'gaussian_ev',
	# 'SEM_NOISE': ['gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'],

	'N': 1400,
	'VAL': 0.2,
	'TEST': 0.1,

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
	'h_NULL': 0.00581130989059376,

	'h_LR': 0.000655054828435191,
	'h_OPTIMIZER': f'torch.optim.Adam(overall_net.h_net.parameters(), lr=h.h_LR)',
	'f_LR': 0.0117744218586073,
	'f_OPTIMIZER': f'torch.optim.Adam(overall_net.f_net.parameters(), lr=h.f_LR)',
	'N_EPOCHS': 1_000,
	'BATCH_SIZE': 32,
	'BATCH_SHUFFLE': True,
	'N_SAMPLES': 10,

	'NOISE_TEMPERATURE': 10 * 0.16213500214616,
	'TEMPERATURE_SCHEDULE': 'TemperatureGeometricSchedule('
							'temperature, 0.01, 100, '
							'-100, h.N_EPOCHS)',
	'NOISE_DISTRIBUTION': 'GumbelNoiseDistribution()',
	'LAMBDA': 5.69792034959308,

	'LOSS_FN': 'torch.nn.MSELoss()',

	'ALPHA': None,
	'BETA': None,
	'REGULARIZER': None,
	'z_RHO': 0.0280244755372679,
	'z_MU': 0.220018032970804,
	'z_REGULARIZER': 'NoTearsZRegularizer(h.D, h.z_RHO, h.z_MU, c.DEVICE)',
	'LR_SCHEDULER': None,

	'RANDOM_SEEDS': [
			(293980, 609514, 429458),
			# (821551, 857364, 588550),
			# (371855, 15930, 244995),
			# (662199, 525268, 239428),
			# (73172, 445778, 775056),
			# (277503, 929817, 567779),

			# (218880, 613892, 234212),
			# (339207, 560204, 281571),
			# (550293, 104365, 501887),
			# (707601, 895448, 703033),
			# (556735, 301962, 832213),
			# (281566, 365419, 34791),

			# (2245, 849552, 1523),
			# (990070, 28565, 775784),
			# (682481, 138627, 47263),
			# (136955, 697796, 757754),
			# (876391, 241970, 564937),
			# (37799, 421942, 748067),
			# (437605, 125781, 646824),
			# (321803, 577399, 715741),
			# (986054, 358591, 399445),
			# (829031, 827673, 257153),
			# (291957, 738715, 728288),
			# (711905, 393652, 17188),
			# (75208, 860744, 567212),
			# (929614, 146979, 804696),
			# (315652, 448043, 77861),
			# (962986, 182228, 99230),
			# (414248, 401339, 243024),
			# (803859, 503112, 50981)
		],
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