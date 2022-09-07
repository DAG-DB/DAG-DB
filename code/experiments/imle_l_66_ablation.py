from utils.write_script import write_script


code_main = '$HOME/I-MLE-DAG/code/main_dag.py'
config = None  # If None, same as this file's name
arg_hp = {
	'max_size': [None, 66],
    'rho': [0, 0.0280244755372679],
    'mu': [0, 0.220018032970804],
}
print(arg_hp)
write_script(code_main, config, arg_hp, hrs=96, mem=16)
