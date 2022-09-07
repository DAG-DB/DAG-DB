# Learning directed acyclic graphs by backpropagation

This is the repo for the UCL Machine Learning MSc (2021-22) thesis by 
candidate RTSC3. It contains the DAG-DB framework for learning directed 
acyclic 
graphs (**DAG**s) by **D**iscrete **B**ackpropagation. 

## Abstract
Directed acyclic graphs (DAGs) occur in a wide range of contexts, including 
project management, version control systems, evolutionary biology, and 
graphical models. DAGs are very numerous, but the requirement for no 
cycles also makes them very rare within the wider set of all directed graphs.  
They can be challenging discrete objects to handle, because the absence of 
cycles is typically only established by looking beyond the immediate 
neighbourhood of individual graph nodes.  Nonetheless, the problem of 
learning DAGs from associated data has been widely studied, and 
considerable progress has been made, at least in the context of 
'sparse' DAGs with relatively few edges per node.

Backpropagation is the fundamental tool for neural network learning, and,
because it relies on differentiating the functions which neural network 
layers represent, it can only be directly applied for continuous quantities.  
However, a number of techniques have been developed for backpropagation through discrete variables.

This thesis constructs a framework, DAG-DB, for learning sparse DAGs from data 
by discrete backpropagation.  DAG-DB draws on the framework used for discrete
backpropagation in implicit maximum likelihood estimation 
(I-MLE)&nbsp;[[1]](#Bibliography), and performs competitively using either of 
two 
discrete-backpropagation techniques, I-MLE itself or 
straight-through estimation&nbsp;[[2, 3]](#Bibliography).  This is demonstrated in 
experiments on selected synthetic data
and on a real dataset.  These experiments see DAG-DB mostly  out-performing
well-established combinatoric methods.  Its predictions are less good than 
those for the best-known continuous optimisation methods, but it may be 
possible to narrow the gap by further hyperparameter optimisation.  Moreover,
DAG-DB could be incorporated in wider neural network structures to enable 
deep learning to deal more easily with problems which can be represented
by directed acyclic graphs.

### Bibliography

[1] Mathias Niepert, Pasquale Minervini, and Luca
Franceschi.  [Implicit MLE: Backpropagating
through discrete exponential family distributions](https://proceedings.neurips.cc/paper/2021/hash/7a430339c10c642c4b2251756fd1b484-Abstract.html), 2021.

[2] Geoffrey Hinton, Nitish Srivastava, and Kevin Swersky. Neural networks for
machine learning. Coursera, video lectures, 264(1):2146–2153, 2012.

[3] Yoshua Bengio, Nicholas Léonard, and Aaron
Courville. [Estimating or propagating gradients
through stochastic neurons for conditional computation](https://doi.org/10.48550/arxiv.1308.3432), 2013.

## Installation

Run `conda env create -f environment.yml` to create a conda environment 
DAG-DB in Python 3.10. 

I used the the Linux Mint 20.3 Cinnamon operating system, but any Linux (at 
least) should work.

As described in `code/README.md`, some experiments use clones of other repos:
- [`py-causal`](https://github.com/bd2kccd/py-causal)
- [NOTEARS](https://github.com/xunzheng/notears)
- [GOLEM](https://github.com/xunzheng/notears).

## Using DAG-DB

Go to './code' for `example.ipynb` and a README with further details.  The 
repo contents are as follows:

- `code/` project code, probably the first place to visit.  Includes more 
  detailed README.md for the code
- `data/` synthetic and real data used in the thesis
- `latents/` used by former in-model tracking option
- `logs/` log files from training
- `log_to_collate/` log files grouped together for processing results of 
  certain experiments
- `models/` to contain any saved models
- `results/` contains csv files with results from certain experiments
- `environment.yml` details of the Python 3.10 environment for DAG-DB
- `LICENCE`
- this `README.md`
- `requirements_for_cluster.txt` used only for running in the UCL computing 
  cluster.

## Acknowledgements

I am very grateful to my thesis supervisors, Dr Pasquale Minervini, 
Dr Valentina Zantedeschi and Dr Luca Franceschi.  They have all been immensely
enthusiastic and generous with their time.  I have benefited hugely from 
their guidance, perspective and challenge, very much enjoying our discussions.
I am grateful too to them for introducing me to a fascinating subject area.
My thanks also to Dr Pontus Stenetorp for taking on the formal sponsorship
of the project. 

## Citing this work

If you use this work, please cite my MSc thesis:

@mastersthesis{DAG-DB,  
&nbsp;&nbsp;&nbsp;&nbsp;title={Learning directed acyclic graphs by backpropagation},  
&nbsp;&nbsp;&nbsp;&nbsp;author={Candidate~RTSC3},  
&nbsp;&nbsp;&nbsp;&nbsp;year={2022},  
&nbsp;&nbsp;&nbsp;&nbsp;month={Sept},  
&nbsp;&nbsp;&nbsp;&nbsp;school={University College London},  
&nbsp;&nbsp;&nbsp;&nbsp;url={ https://github.com/DAG-DB/DAG-DB }  
}

## Disclaimer

This report is submitted as part requirement for the MSc Machine Learning
at UCL. It is substantially the result of my own work except where explicitly indicated
in the text. The report may be freely copied and distributed provided the source is explicitly
acknowledged.

## Licence

See ./LICENCE
