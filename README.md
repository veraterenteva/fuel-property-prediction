# fuel-property-prediction
An ML project for prediction of motor octane number, research octane number, and cetane number of fuel blends and individual components.

## Data
The source of data on MON and RON: 

1. Kuzhagaliyeva, N., Horváth, S., Williams, J. et al. Artificial intelligence-driven design of fuel mixtures. Commun Chem 5, 111 (2022). https://doi.org/10.1038/s42004-022-00722-3
   
The source on CN:

2. Numerical Approaches to Determine Cetane Number of Hydrocarbons and Oxygenated Compounds, Mixtures, and their Blends Benoit Creton, Nathalie Brassart, Amandine Herbaut, and Mickael Matrat Energy & Fuels 2024 38 (16), 15652-15661 DOI: 10.1021/acs.energyfuels.4c03007

Included data contains SMILES strings, MON, RON, and CN values of inidividual components and their blends.

Data files in /data/:

- pure_for_mix.csv - processed data on pure components from [1]
- mix_combined.csv - processed data on blends and pure components from [1]
- mix_combined.csv - the same as previous + data on pure components and blends from [2]

## Model
We reproduced the model from [1] with several changes:

- We use RDKit descriptors instead of Mordred ones
- The descriptor encoder has lower amount of neurons
- We filter 100 most relevant descriptors using Random Forest
- The composition finder algorithm is stochastic

## Repository structure


```
fuel-property-prediction/
│
├── data/ # processed data used both in training and predictions
│   ├──mix_combined.csv # blends and pure components with MON and RON values
│   ├──mix_combined_cn.csv # blends and pure components with MON, RON, and CN values
│   └──pure_to_mix.csv # pure components with MON and RON values
│   
│── model/ # the model itself and auxiliary functions
│      ├── __init__.py
│      ├── blending_optimizer.py # blends finder functions (not execution)
│      ├── config.py # model parameters
│      ├── data_preprocessing.py # data preprocessing functions 
│      ├── datasets.py # datasets classes and functions
│      ├── evaluation.py # evaluation functions
│      ├── find_blends.py # blends finder execution
│      ├── models.py # basic model classes
│      ├── train_model.py # training execution
│      ├── training.py # training cycle function
│      └── requirements.txt
│   
│ ── static/
│      └── style.css 
│   
│── templates/ 
│      └── index.html
|
├── LICENSE
├── README.md # this file
├── best_model.pth # weights dictionary of the model (gets overwritten with training)
├── main.py # main execution script
├── model_service.py 
├── requirements.txt
├── schemas.py 
```


## Run

Python of version 3.10 or newer is recommended.

To install required packages, run in bash:

`!pip install requirements.txt`

To host the web server on the local machine, run:

`python -m uvicorn main:app --reload`

The web service with model interface can be accessed through the http://127.0.0.1:8000/ port.
