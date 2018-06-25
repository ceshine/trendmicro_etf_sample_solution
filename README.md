# Sample Solution to Taiwan ETF Price Prediction Competition
[台灣ETF價格預測競賽](https://tbrain.trendmicro.com.tw/Competitions/Details/2)

This solution uses a [Bayesian hierarchical model](https://www.wikiwand.com/en/Bayesian_hierarchical_modeling) largely based on [ceshine/kaggle-winton-2016](https://github.com/ceshine/kaggle-winton-2016) repo (which is based on [Tsakalis Kostas's model](Tsakalis Kostas)).

WARNING: This is a very simple baseline model. It is not ready for real trading.

## Data

Unzip `TBrain_Round2_DataSet_20180615.zip` to get the sample data. Everything is actually public information. The zip file is provided only to make reproducing results easier.


## Model Description

Please check the [Stan model](scripts/bayesian_linear_model.stan).

It uses the last five trading day to predict the price at the end of the target day. We train one independent model for each weekday (so exactly 5 models are trained).

Public holidays are ignored (which is not ideal and definitely can be improved).

## Usage

A Dockerfile is included.

1. Firstly, build the image (example: `docker build -t pystan .`)
2. Start a container and mount the project dir. (example: `docker run -ti -v $(pwd):/lab pystan bash`).
3. Run `run.sh` script. (example: `cd /lab && ./run.sh`)

The predictions will be saved in [cache/baseline.csv](cache/baseline.csv).

Change the `target_date` variable in `make_submission` function from [scripts/bayesian.py](scripts/bayesian.py) to the first day (**t**) you want to predict. The script will output prediction from **t** to **t+4**.

### Brief Summary of Modules

1. [scripts/fix_csv_files.py](scripts/fix_csv_files.py): convert Big-5 to UTF-8 and fix some trailing commas.
2. [scripts/preprocess.py](scripts/preprocess.py): conver the csv file into a more computer-friendly data frame and store it as a feather file.
3. [scripts/bayesian.py](scripts/bayesian.py): model training, evaluation, and prediction.
4. [scripts/prepare_features.py](scripts/prepare_features.py): some very basic feature engineering utility functions.
