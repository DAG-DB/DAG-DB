from utils.write_script import write_script


code_main = '$HOME/I-MLE-DAG/code/main_dag.py'
config = None  # If None, same as this file's name
arg_hp = {
	'lamb': [0.1, 1],
	'temperature': [0.01],
	'h_null': [0],
	'h_lr': [1e-4],
}

write_script(code_main, config, arg_hp)
