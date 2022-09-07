from utils.write_script import write_script


code_main = '$HOME/I-MLE-DAG/code/main_dag.py'
config = None  # If None, same as this file's name
# arg_hp['dags'] = {"\"\'" + name + "\'\"" for name in arg_hp['dags']}
arg_hp = {
	'dags': ["\"\'" + str(n).zfill(3) + "\'\"" for n in range(10)]
}
print(arg_hp)
write_script(code_main, config, arg_hp, hrs=24 ,mem=10)
