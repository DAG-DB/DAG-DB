from utils.write_script import write_script


code_main = '$HOME/I-MLE-DAG/code/main_dag.py'
config = 'l_66none_4hps'  # If None, same as this file's name
hrs = 6
arg_hp = {
	'lamb': [0.1, 1, 10, 100],
	'temperature': [0.01, 0.1, 1, 10],
	'h_null': [0, 1e-3, 1e-2, 1e-1, 1],
	'h_lr': [1e-4, 1e-3, 1e-2, 1e-1],
}
run_jobs = [ 33,  37,  89, 119, 132, 135, 139, 146, 150, 152, 158, 159, 169,
       172, 175, 176, 181, 185, 206, 208, 210, 226, 228, 234, 240, 246,
       257, 269, 277, 282, 286]

write_script(code_main, config, arg_hp, hrs=hrs, run_jobs=run_jobs)

