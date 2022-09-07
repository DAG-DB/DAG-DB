import os


COMMENT = ""

# File Structure
MODEL_FOLDER = os.path.join('..', 'models')
LOGS_FOLDER = os.path.join('..', 'logs')
LATENTS_FOLDER = os.path.join('..', 'latents')

# Device
DEVICE = None  # "cpu" # "cuda"  # "cpu"  # None

# Reporting
MATRIX_REPORT = False
LABEL = True
SAVE_TO_ODS = False
VAL_LOSS_FROM_DAG = True  # Relevant only if mode is 'max_dag_test_only'

# Optuna - putting these as strings in hyperparameters ensures properly
# recorded in .ods
N_TRIALS = 100
OPTUNA_STRINGS = {'None': None, 'True': True, 'False': False}

TQDM = False

# Hyperparameters

hyperparameters = {
	'D': 30,  # STANDARD
	'SIZE_MIN': 20,
	'SIZE_MAX': 100,

	'MAX_SIZE': 72,
	# 'MAX_SIZE': 'trial.suggest_categorical("max_size", [72, "None"])',

	'SEM_NOISE': 'gaussian_ev',
	# 'SEM_NOISE': ['gaussian_ev', 'gaussian_nv', 'exponential', 'gumbel'],

	'N': 1400,
	'VAL': 0.2,
	'TEST': 0.1,

	'MODE': 'max_dag',
	# MODE from 'lt_p_matrix', 'lt_p_vector_argsort_in_std_autograd',
	# 'max_dag',
	'f_LINEAR_LAYERS': tuple(),
	'f_BIAS': False,

	'p_LAYER': None,
	'STREAMLINE': False,
	'THRESHOLD': 0.,  # STANDARD

	'h_LINEAR_LAYERS': tuple(),
	# 'h_NULL': 0.348678556296974,
	'h_NULL': 'trial.suggest_float("h_null", 1e-3, 1., log=True)',

	# 'LR': 0.00443142553171277,
	# 'LR': 'trial.suggest_float("lr", 1e-4, 1e-2)',
	# 'LR': 0.00197268311739694,  # 'trial.suggest_float("lr", 1e-4, 1e-2)',
	'h_LR': 'trial.suggest_float("h_lr", 1e-5, 1., log=True)',
	'h_OPTIMIZER': f'torch.optim.Adam(overall_net.h_net.parameters(), lr=h.h_LR)',
	'f_LR': 'trial.suggest_float("f_lr", 1e-5, 1., log=True)',
	'f_OPTIMIZER': f'torch.optim.Adam(overall_net.f_net.parameters(), lr=h.f_LR)',
	'N_EPOCHS': 400,  # 150,  # NON-STANDARD
	'BATCH_SIZE': 32,  # STANDARD
	'BATCH_SHUFFLE': True,
	'N_SAMPLES': 10,

	'NOISE_TEMPERATURE': 'trial.suggest_float("noise_temp", 0.03, 30, log=True)', # 10.0,  # STANDARD
	'NOISE_DISTRIBUTION': 'GumbelNoiseDistribution()',
	'LAMBDA': 'trial.suggest_float("lambda", 0.03, 30, log=True)',
	# 10., STANDARD

	'LOSS_FN':
		# 'Golem1ER4Loss(alpha=10)',
		'torch.nn.MSELoss()'  # STANDARD
		# 'golem_loss_1',
		# 'golem_loss_2'
	,
	'ALPHA': None,
	'BETA': None,
	'REGULARIZER': None,  # 'MeanSquareSigmoidRegularizer(d, c.DEVICE, '
				   # 'alpha=h.ALPHA, beta=h.BETA)',
	'LR_SCHEDULER': None,  # STANDARD
	# 'LR_SCHEDULER': 'torch.optim.lr_scheduler.MultiStepLR(milestones=[40], '
	# 				'gamma=0.1)',
	# 'LR_SCHEDULER': 'torch.optim.lr_scheduler.CyclicLR(base_lr=1e-3, '
	# 				'max_lr=1e-2, step_size_up=300, cycle_momentum=False)',

	'RANDOM_SEEDS': [
			(293980, 609514, 429458),
			(821551, 857364, 588550),
			(371855, 15930, 244995),
			(662199, 525268, 239428),
			(73172, 445778, 775056),
			(277503, 929817, 567779),
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