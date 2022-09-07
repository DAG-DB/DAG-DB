import os


COMMENT = "Creata DAGs"

# File Structure
LOGS_FOLDER = os.path.join('..', 'logs')
DATA_FOLDER = os.path.join('..', 'data')

DEVICE = None

# Hyperparameters

hyperparameters = {
	'DATA_CATEGORY': 'v_methods_test',

	'GRAPH_TYPE': 'SF4',
	'D': 100,
	'SEM_NOISE': 'gaussian_ev',

	'N': 1200,
	'VAL': 1 / 6,


	'DAGS': [
			991800,
			853188,
			125570,
			203101,
			322761,
			91494,
			454855,
			355304,
			942050,
			157863,
			84333,
			365716,
			780854,
			30415,
			502396,
			111466,
			90963,
			732624,
			455798,
			871895,
			876436,
			6698,
			428139,
			447652
		],
}

