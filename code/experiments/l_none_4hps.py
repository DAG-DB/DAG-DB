from utils.write_script import write_script


code_main = '$HOME/I-MLE-DAG/code/main_dag.py'
config = None  # If None, same as this file's name
arg_hp = {
	'lamb': [0.1, 1, 10, 100],
	'temperature': [0.01, 0.1, 1, 10],
	'h_null': [0, 1e-3, 1e-2, 1e-1, 1],
	'h_lr': [1e-4, 1e-3, 1e-2, 1e-1],
}

write_script(code_main, config, arg_hp)
