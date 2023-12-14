# Classification of Multi-label 12-Lead Electrocardiogram Readings Using Deep Neural Network

This repository is forked from the work of [Zhibin Zhao](https://zhaozhibin.github.io/).
   

## Getting Started    
1. Install the necessary libraries.
This repo is tested with Ubuntu 22.04, PyTorch 2.0.1, Python3.6. Install package dependencies as follows:

```
pip install -r requirements.txt    
```

2. Download the PhysioNet Challenge 2020 database.   
```
wget -r -N -c -np https://physionet.org/files/challenge-2020/1.0.2/
```

3. Prepare dataset
Run [stratification_split_single.py] # make sure you add the dataset root directory 

4. Run training
Run [train_model.py] or [a-e] scripts # make sure you have provided the right parameter as an argument.