from ALS.als_utils import *
import numpy as np
from hyperopt import hp, fmin, tpe,  Trials






class Model():
    """ A class used to represent a model

    :arg latent_dim: int the size of the latent dimension d
    :arg user_items: mapping <key: user_index, value: relevant items, 2d array [item_index, rating]>
    :arg item_users: mapping  <key: item_index, value: relevant users, 2d array [user_index, rating]>
    :arg ranking_matrix_train: np.array two dimensional array, rows=users columns=items values=ratings
    :arg ranking_matrix_validation: np.array two dimensional array, rows=users columns=items values=ratings
    :arg l_users: float regularization term for users
    :arg l_items: float regularization term for items
    :arg l_bias_users: float regularization term of bias of users
    :arg l_bias_items: float regularization term of bias of items



    """
    def __init__(self, latent_dim, user_items, item_users,  ranking_matrix_train, ranking_matrix_validation,   l_users, l_items, l_bias_users, l_bias_items):
        self.d=latent_dim
        self.user_items=user_items
        self.item_users=item_users
        self.m=len(user_items.keys())
        self.n=len(item_users.keys())
        self.R=ranking_matrix_train
        self.R_valid=ranking_matrix_validation
        self.U,self.V,self.Bu,self.Bi=init_variables(self.d, self.m,self.n)
        self.lu=l_users
        self.li=l_items
        self.lbu=l_bias_users
        self.lbi=l_bias_items



    def train_model(self, iterations, early_stop=10, verbose=True):
        """Alternate updates  of U,  Bu, and then  V, Bi. This is done iteratively or until early stopping.

            :param train: pd.DataFrame of training data as retrieved from DataSet()
            :param validation: pd.DataFrame of validation data as retrieved from DataSet()
            :param iterations: int number of iterations in ALS
            :param early_stop: int number of consecutive epochs that the validation score has not improved


            :return  float of rmse on validation set """

        self.mu=self.R[self.R>0].mean()
        cnt = 0 #counter for early stopping
        best_rmse= 999

        for iter in range(iterations):
            if verbose: print('Starting iteration {} out of {}'.format(iter+1,iterations))
            for user in self.user_items:
                #print('Updating Um, Bm for user {} out of {}'.format(user+1, self.m))
                self.U[user]  = update_Um(user, self.user_items[user], self.V,  self.Bu, self.Bi, self.lu, self.mu)
                self.Bu[user] = update_bm(user, self.user_items[user], self.U, self.V,  self.Bi, self.lbu, self.mu)

            for item in self.item_users:
                #print('Updating Vn, Bn for item {} out of {}'.format(item + 1, self.n))
                self.V[item] =  update_Vn(item, self.item_users[item], self.U,  self.Bu, self.Bi, self.li, self.mu)
                self.Bi[item] = update_bn(item, self.item_users[item], self.U, self.V,  self.Bu, self.lbi, self.mu)

            r2_score_train, mae_train, rmse_train= self.evaluate(is_train=True)
            r2_score_valid, mae_valid, rmse_valid= self.evaluate(is_train=False)

            if verbose: print('Train set : R2 {}       MAE {}      RMSE {}'.format(r2_score_train, mae_train, rmse_train))
            if verbose: print('Validation set : R2 {}       MAE {}      RMSE {}'.format(r2_score_valid, mae_valid, rmse_valid))

            cnt+=1
            if rmse_valid<best_rmse:
                cnt=0
                best_rmse=rmse_valid
            if cnt>early_stop:
                break

        return best_rmse


    def evaluate(self, is_train):
        """Evaluates  R sqaure, MAE, RMSE for dataset.
            :param      bool: use train set or validation

            :return r2 float
            :return mae float
            :return rmse float

             """

        #yhat=validation_set.apply(lambda row: self.predict(row.User_ID_Alias, row.Movie_ID_Alias), axis=1)

        #print(yhat)
        # Get the correct values according to train/test
        r_true= self.R if is_train else self.R_valid

        #Get prediction for entire dataset
        r_pred=np.dot(self.U, self.V.T)+self.Bu.reshape(-1,1)+self.Bi.reshape(-1)+self.mu
        r_pred=np.clip(r_pred,1,5)

        idx=r_true>0

        r2_score=calc_r2(r_true[idx], r_pred[idx])
        mae=calc_mae(r_true[idx], r_pred[idx])
        rmse=calc_rmse(r_true[idx], r_pred[idx])

        return r2_score,mae,rmse







    def predict(self, user, item):
        """Predicts yhat=U.t * V +Bu+Bi+mu. If this is a new user outputs mean of items. If his is a new item outputs mean of users.
           If it a new user and item outputs mean of dataset

                            :param user: int index of user
                            :param item: int index of item

                            :return  float prediction"""

        #unknown or user should get -1
        if item==-1 and user != -1:
            return self.user_items[user][1].mean()
        if user==-1 and item !=-1:
            return self.item_users[item][1].mean()
        if item==-1 and user==-1:
            return self.mu
        else:
            pred=np.dot(self.U[user, :].T, self.V[item, :])+self.Bu[user]+self.Bi[item]+self.mu
            return float(np.clip(pred, 1,5))

    def hyperparmas_tune(self, max_evals=100):
        """Tunes hyperparams in bayesian optimization:


         :param max_evals: int number of iterations of bayesian optimization
         :return dict <key: hyperparam, value: best value>

        """

        params = {
            'd': hp.quniform('d', 5,50,1),
            'lu': hp.loguniform('lu', -1,4.5),
            'li': hp.loguniform('li', -3, 3),
            'lbu': hp.loguniform('lbu', -1, 4),
            'lbi': hp.loguniform('lbi', -3, 3)

        }

        trials = Trials()
        best = fmin(fn=self.objective, space=params, algo=tpe.suggest, max_evals=max_evals, trials=trials)
        return best


    def objective(self, params):
        """ helper function for hyperparams_tune
            :param dict <key: hyperparam, value: value to check>
            :return float"""

        for key in params.keys():
            exec("self.{}={}".format(key, params[key]))
        print(' Set Hyperparams to d:{} lu: {} li:{} lbu:{} lbi:{}'.format(self.d,self.lu,self.li,self.lbu,self.lbi))
        rmse = self.train_model(iterations=100, early_stop=10, verbose=False)
        print('rmse validation set: {}'.format(rmse))
        return rmse










