from utils.write_script import write_script


code_main = '$HOME/I-MLE-DAG/code/main_dag.py'
config = 'l_66_4hps'  # If None, same as this file's name
hrs = 6
arg_hp = {
	'lamb': [0.1, 1, 10, 100],
	'temperature': [0.01, 0.1, 1, 10],
	'h_null': [0, 1e-3, 1e-2, 1e-1, 1],
	'h_lr': [1e-4, 1e-3, 1e-2, 1e-1],
}
run_jobs = [7,  17,  36,  39,  50,  57,  59,  60,  70,  81,  83,  85,  88,
        99, 100, 101, 111, 113, 114, 118, 122, 124, 125, 129, 130, 139,
       140, 148, 152, 153, 154, 155, 156, 157, 158, 159, 161, 162, 163,
       164, 165, 166, 167, 173, 174, 179, 181, 183, 187, 188, 195, 200,
       202, 221, 223, 226, 255, 256, 261, 266, 282, 285, 292, 294, 306,
       315]
write_script(code_main, config, arg_hp, hrs=hrs, run_jobs=run_jobs)
