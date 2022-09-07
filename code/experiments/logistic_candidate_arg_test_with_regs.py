from utils.write_script import write_script


code_main = '$HOME/I-MLE-DAG/code/main_dag.py'
config = None  # If None, same as this file's name
arg_hp = {
	'max_size': [54, 60, 66, 72, 78, 84],
}

write_script(code_main, config, arg_hp)
