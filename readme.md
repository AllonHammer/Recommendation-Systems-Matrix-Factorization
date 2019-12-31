# Predicting User Movie Rating Using Matrix Factorization
This project implements a full matrix factorization algorithm from scratch optimized by
1. Stochastic Gradient Descend
2. Alternating Least Squares

More information about the models is provided in: 
1. /resources/SGD-report.pdf
2. /resources/ALS-report.pdf

Here are brief descriptions of the data:


Train.csv FILE DESCRIPTION
================================================================================
859395 ratings by 6040 users on 3224 items
this is a comma-delimited list of
              User_ID_Alias, Movie_ID_Alias, Rating


Validation.csv FILE DESCRIPTION
================================================================================
6040 ratings by 6040 (1 movie per user)
this is a comma-delimited list of
              User_ID_Alias, Movie_ID_Alias, Rating


Test.csv FILE DESCRIPTION
================================================================================
6040 ratings of 6040 users to forecast (1 movie per user)
this is a comma-delimited list of
              User_ID_Alias, Movie_ID_Alias






## Quick Setup


#### Prerequisites

    * Python >= 3.5

> **NOTICE:** [`pyenv`](https://github.com/pyenv/pyenv) is recommended for python installation.

#### Setting up the project

Set up virtualenv

```sh
$ make virtualenv
$ source venv/bin/activate
```

(You can wipe out virtualenv by running;)

```sh
$ make cleanenv
```

Once project is set, export its PYTHONPATH.

```sh
$ pip install -r requirements.txt
```



#### Usage 
To execute the code please run the main.py file.
There are 3 arguments to pass:
1. --model (required)  sgd/als
2. --load (optional) to use pre-trained data 
3. --tune (optional) to conduct hyper-param optimization (might take some time)


```sh
$ python main.py --model als --load --tune
***** ALS MODEL****
Loading pretraind DataSet()
Starting Hyper Param Tuning
 Set Hyperparams to d:8.0 lu: 1.388933955540989 li:4.994310872919233 lbu:2.014823102181593 lbi:2.1503823779278517
  0%|                                    | 0/1000 [00:00<?, ?it/s, best loss: ?]
```


```sh
$ python main.py --model als --load 
***** ALS MODEL****
Loading pretraind DataSet()
Starting iteration 1 out of 100
Train set : R2 0.34554270795597675       MAE 0.7104441174095031      RMSE 0.9018811978921434
Validation set : R2 0.24806139943931593       MAE 0.762475309782658      RMSE 0.9669345514610405
Starting iteration 2 out of 100
Train set : R2 0.46869290949915776       MAE 0.6389996530020154      RMSE 0.8126087293514701
Validation set : R2 0.30239932225530686       MAE 0.731343893918957      RMSE 0.9313423168508486
```


```sh
$ python main.py --model sgd 
***** SGD MODEL****
item removed:  3698
train biased params:  iter: 0 prev_rmse_v:  1000 curr_rmse:  1.093695 cur_mae:  0.84974 cur_r_2:  0.038 no_improve_counter:  0
train biased params:  iter: 1 prev_rmse_v:  1.093695 curr_rmse:  1.077682 cur_mae:  0.83977 cur_r_2:  0.066 no_improve_counter:  0
train biased params:  iter: 2 prev_rmse_v:  1.077682 curr_rmse:  1.065179 cur_mae:  0.83133 cur_r_2:  0.0875 no_improve_counter:  0
train biased params:  iter: 3 prev_rmse_v:  1.065179 curr_rmse:  1.05401 cur_mae:  0.82369 cur_r_2:  0.1065 no_improve_counter:  0
```


```sh
$ python main.py --model sgd 
***** SGD MODEL****
item removed:  3698
train biased params:  iter: 0 prev_rmse_v:  1000 curr_rmse:  1.093695 cur_mae:  0.84974 cur_r_2:  0.038 no_improve_counter:  0
train biased params:  iter: 1 prev_rmse_v:  1.093695 curr_rmse:  1.077682 cur_mae:  0.83977 cur_r_2:  0.066 no_improve_counter:  0
train biased params:  iter: 2 prev_rmse_v:  1.077682 curr_rmse:  1.065179 cur_mae:  0.83133 cur_r_2:  0.0875 no_improve_counter:  0
train biased params:  iter: 3 prev_rmse_v:  1.065179 curr_rmse:  1.05401 cur_mae:  0.82369 cur_r_2:  0.1065 no_improve_counter:  0
```