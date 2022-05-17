# MAREO: Memory- and Attention- based visual REasOning

Code for the paper 'Memory- and Attention- based visual REasOning'.

There are four visual reasoning tasks evaluated in this paper, i.e., same-different (SD), relation match to sample (RMTS), distribution of three (Dist3) and identity rule (ID). For each task, there is a script folder with the task name appended at the end containing files to run for different holdout values and 10 runs. To run any particular script basic command is: `./script_{task_name} architecture_name learning_rate steps`. 

For example, to reproduce the results for the MAREO architecture on the same/different discrimination task and 0 holdout case, run the following command:
```
./scripts_sd/m_0.sh MAREO .0001 4
```

## Installation instructions:
```
python3 -m venv ../env/mareo  
source ../env/mareo/bin/activate  
python -m pip install --upgrade pip   
python -m pip install -r requirements.txt
```

## Authorship
TBD
