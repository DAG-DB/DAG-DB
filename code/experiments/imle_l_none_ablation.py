from utils.write_script import write_script


code_main = '$HOME/I-MLE-DAG/code/main_dag.py'
config = None  # If None, same as this file's name
arg_hp = {
	'max_size': [None, 66],
    'rho': [0, 0.157451211201],
    'mu': [0, 0.00120806965197976],
}
print(arg_hp)
write_script(code_main, config, arg_hp, hrs=96, mem=16)
