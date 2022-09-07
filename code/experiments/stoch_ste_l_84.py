from utils.write_script import write_script


code_main = '$HOME/I-MLE-DAG/code/main_dag.py'
config = None  # If None, same as this file's name
# arg_hp['dags'] = {"\"\'" + name + "\'\"" for name in arg_hp['dags']}
arg_hp = dict()
print(arg_hp)
write_script(code_main, config, arg_hp, hrs=96 ,mem=12, multiple_runs=10)
