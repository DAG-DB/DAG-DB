from utils.write_script import write_script


code_main = '$HOME/I-MLE-DAG/code/main_dag.py'
config = None  # If None, same as this file's name
arg_hp = {
	'dags': [
		"010_ER1_gaussian_ev_2022-08-20_BST_11:36:33,306_create_dags",
		"010_ER2_gaussian_ev_2022-08-20_BST_11:37:40,257_create_dags",
		"010_ER4_gaussian_ev_2022-08-20_BST_11:38:06,145_create_dags",
		"010_SF2_gaussian_ev_2022-08-20_BST_11:39:28,507_create_dags",
		"010_SF4_gaussian_ev_2022-08-20_BST_11:39:51,306_create_dags",
		"030_ER1_gaussian_ev_2022-08-20_BST_11:37:19,132_create_dags",
		"030_ER2_gaussian_ev_2022-08-20_BST_11:37:48,933_create_dags",
		"030_ER4_gaussian_ev_2022-08-20_BST_11:39:01,675_create_dags",
		"030_SF2_gaussian_ev_2022-08-20_BST_11:39:32,564_create_dags",
		"030_SF4_gaussian_ev_2022-08-20_BST_11:39:59,723_create_dags",
		"100_ER1_gaussian_ev_2022-08-20_BST_11:37:28,139_create_dags",
		"100_ER2_gaussian_ev_2022-08-20_BST_11:37:54,238_create_dags",
		"100_ER4_gaussian_ev_2022-08-20_BST_11:39:15,764_create_dags",
		"100_SF2_gaussian_ev_2022-08-20_BST_11:39:41,987_create_dags",
		"100_SF4_gaussian_ev_2022-08-20_BST_11:40:07,367_create_dags"
	]
}
arg_hp['dags'] = {"\"\'" + name + "\'\"" for name in arg_hp['dags']}
print(arg_hp)
write_script(code_main, config, arg_hp, hrs=24 ,mem=12)
