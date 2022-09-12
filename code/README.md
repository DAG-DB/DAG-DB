# Learning directed acyclic graphs by backpropagation: DAG-DB `code/` directory

This README gives instructions for using DAG-DB.  Information on the repo 
as a whole is in the repo root README, including licence information.

## Installation

Run `conda env create -f environment.yml` in the repo root directory to 
create a conda environment DAG-DB in Python 3.10.  I ran this in the Linux 
Mint 20.3 Cinnamon operating system.

## Using DAG-DB

Start by running `example.ipynb` for a demonstration.  The 
`code/` contents are as follows:

- `analyis/`, miscellaneous analytical or drawing programs, including 
  for the figures in Ch&nbsp;1 project code in `draw_some_dags.ipynb` and a 
  simple demonstration of SHD for random DAGs in `random_shd.py`
- `cluster_files/` containing `qsub` shell files for UCL cluster experiments 
  created by code in `experiments/`
- `collated_results/` where most of the figures and tables for the 
  thesis were produced
- `data_management/` contains code to create synthetic DAGs/data and code 
  to use both synthetic and real data
- `discrete_backprop/` I-MLE and other discrete backprop methods.  Based on 
  the I-MLE code framework
- `experiments/` code that generates cluster `qsub` shell files placed in 
  `cluster_files`.  Each `experiment/` file has the same name as a 
  `named_config/` file, and the `qsub` shell file will share the same 
  basename stem
- `for_external/` contains files that need to be transferred to clones 
  of other repos to run experiments in Ch&nbsp;6 of the thesis.  These 
  other repos are [`py-causal`](https://github.com/bd2kccd/py-causal), 
  [NOTEARS](https://github.com/xunzheng/notears) and [GOLEM](https://github.com/xunzheng/notears)
- `learn_infer/` used by in-model tracking option
- `lib/` library code used by DAG-DB 
- `named_configs/` log files grouped together for processing results of 
  certain experiments
- TODO delete `old/`
- `utils/` miscellaneous utilities for DAG-DB
- `wandb/` stores data created and used by wandb
- `create_dags.py` the main code to create new synthetic DAGs and data 
- `example.ipynb` the example mentioned above - start here
- `main_dag.py` the main program for running DAG-DB.

## Running `main_dag.py`

Start by trying `example.ipynb`.  This gives an example of how to run 
`main_dag.py` with a config file as the only argument:

`python3 main_dag.py -c <name of config file without .py>`, 

for example 

`python3 main_dag.py -c imle_logistic_none`.

The config file must be a Python `.py` file in the `code/named_configs/` 
directory. To do a single run, the easiest method is often to create your 
own config file.  The file `code/named_configs/guide_to_config_files.py` 
explains the options.

Exception: if we call `python3 main_dag.py` with no `-c` arguments, 
including if we 
pass no arguments at all, `main_dag.py` will look for a config file 
`code/config.py`.  This can be used for 'scratch' config files.

To help with batch runs, where hyperparameters vary, additional command line 
arguments can be passed, with or without a `-c` command.  
The possible additional command line arguments, and the config fields 
they set, are in the following table.

| Command line | Config field |
| --- | --- |
| `--max_size` | `MAX_SIZE` |
| `--lamb` | `LAMBDA` |
| `--temperature` | `NOISE_TEMPERATURE` |
| `--h_null` | `h_NULL` |
| `--h_lr` | `h_LR` |
| `--rho` | `z_RHO` |
| `--mu` | `z_MU` |
| `--dags` | `DAGS` |
 
Before such a command line argument is used, the corresponding 
field in the 
config file must have been set to `None`.  These command line arguments were 
used to create many of the thesis figures, for example fig.&nbsp;
6.3 used a single config file and passed `--lamb`, `--temperature`, `--h_null`
and `--h_lr` arguments.