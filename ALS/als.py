from data_preprocess.preprocess import DataSet
from data_preprocess.preprocess_utils import convert_to_mappings
from ALS.als_utils import *
import numpy as np
import time







class Model():
    """ A class used to represent a model

    :arg latent_dim: int the size of the latent dimension d
    :arg num_users: int number of users in train set
    :arg num_items: int number of items in train set
    :arg ranking_matrix_train: np.array two dimensional array, rows=users columns=items values=ratings
    :arg ranking_matrix_validation: np.array two dimensional array, rows=users columns=items values=ratings
    :arg l_users: float regularization term for users
    :arg l_items: float regularization term for items
    :arg l_bias_users: float regularization term of bias of users
    :arg l_bias_items: float regularization term of bias of items



    """
    def __init__(self, latent_dim, num_users, num_items,  ranking_matrix_train, ranking_matrix_validation,   l_users, l_items, l_bias_users, l_bias_items):
        self.d=latent_dim
        self.m=num_users
        self.n=num_items
        self.R=ranking_matrix_train
        self.R_valid=ranking_matrix_validation
        self.U,self.V,self.Bu,self.Bi=init_variables(self.d, self.m,self.n)
        self.lu=l_users
        self.li=l_items
        self.lbu=l_bias_users
        self.lbi=l_bias_items



    def train_model(self, iterations, early_stop=10):
        """Alternate updates  of U,  Bu, and then  V, Bi. This is done iteratively or until early stopping.

            :param train: pd.DataFrame of training data as retrieved from DataSet()
            :param validation: pd.DataFrame of validation data as retrieved from DataSet()
            :param iterations: int number of iterations in ALS
            :param early_stop: int number of consecutive epochs that the validation score has not improved


            :return  float of rmse on validation set """

        self.mu=train.Ratings_Rating.mean()
        cnt = 0 #counter for early stopping
        best_rmse= 999

        for iter in range(iterations):
            print('Starting iteration {} out of {}'.format(iter+1,iterations))
            for user in range(self.m):
                print('Updating Um, Bm for user {} out of {}'.format(user+1, self.m))
                self.U[user] = update_Um(user, self.V, self.R, self.Bu, self.Bi, self.lu, self.mu)
                self.Bu[user] = update_bm(user, self.U, self.V, self.R, self.Bi, self.lbu, self.mu)
            for item in range(self.n):
                print('Updating Vn, Bn for item {} out of {}'.format(item + 1, self.n))
                self.V[item] = update_Vn(item, self.U, self.R, self.Bu, self.Bi, self.li, self.mu)
                self.Bi[item] = update_bn(item, self.U, self.V, self.R, self.Bu, self.lbi, self.mu)

            r2_score_train, mae_train, rmse_train= self.evaluate(is_train=True)
            r2_score_valid, mae_valid, rmse_valid= self.evaluate(is_train=False)

            print('Train set : R2 {}       MAE {}      RMSE {}'.format(r2_score_train, mae_train, rmse_train))
            print('Validation set : R2 {}       MAE {}      RMSE {}'.format(r2_score_valid, mae_valid, rmse_valid))

            cnt+=1
            if rmse_valid<best_rmse:
                cnt=0
                best_rmse=rmse_valid
            if cnt>early_stop:
                break

        return best_rmse

    def train_model_(self, iterations, early_stop=10):
        self.mu = train.Ratings_Rating.mean()
        cnt = 0  # counter for early stopping
        best_rmse = 999

        for iter in range(iterations):
            print('Starting iteration {} out of {}'.format(iter + 1, iterations))
            for user in range(self.m):
                print('Updating Um, Bm for user {} out of {}'.format(user + 1, self.m))
                self.Bu[user] = update_bm_new(user, self.U, self.V, self.R, self.Bi, self.lbu, self.mu)

    def evaluate(self, is_train):
        """Evaluates  R sqaure, MAE, RMSE for dataset.
            :param      bool: use train set or validation

            :return r2 float
            :return mae float
            :return rmse float

             """

        # Get the correct values according to train/test
        r_true= self.R if is_train else self.R_valid

        #Get prediction for entire dataset
        r_pred=np.dot(self.U, self.V.T)+self.Bu.reshape(-1,1)+self.Bi.reshape(-1)+self.mu

        idx=r_true>0

        r2_score=calc_r2(r_true, r_pred)
        mae=calc_mae(r_true, r_pred)
        rmse=calc_rmse(r_true, r_pred)

        return r2_score,mae,rmse







    def predict(self, user, item):
        """Predicts yhat=U.t * V +Bu+Bi+mu. If this is a new user outputs mean of items. If his is a new item outputs mean of users.
           If it a new user and item outputs mean of dataset

                            :param user: int index of user
                            :param item: int index of item

                            :return  float prediction"""

        #unknown or user should get -1
        if item==-1 and user != -1:
            user_mean=self.R[user, :]
            return user_mean[~np.isnan(user_mean)].mean()
        if user==-1 and item !=-1:
            item_mean=self.R[item, :]
            return item_mean[~np.isnan(item_mean)].mean()
        if item==-1 and user==-1:
            return self.R[~np.isnan(self.R)].mean()
        else:
            return np.dot(self.U[user, :].T, self.V[item, :])+self.Bu[user]+self.Bi[item]+self.mu


    def tune(self, iterations, early_stop, tuning_dict):
        """Performs iterative tuning for hyperparams. For simplicity
           we tune  each hyperparam separately instead of all possible combinations.

                    :param iterations: int number of iterations in ALS
                    :param early_stop: int number of consecutive epochs that the validation score has not improved
                    :param tuning_dict: dict <key: hyperparam, value: [value 1, value 2 ... ]>



                    :return  dict of best values  <key: hyperparam, value: best value> """


        best_values={}

        for key in tuning_dict.keys():
            assert key in ['d', 'lu', 'li', 'lbu', 'lbi'], "key must be in  ['d', 'lu', 'li', 'lbu', 'lbi']"
            print('Tuning hyperparam {}', key)
            best_rmse = 999
            best_value=0
            for val in tuning_dict[key]:
                print('Check value {}',val)
                exec("self.{}={}".format(key,val)) #Set hyperparam according to what appears in the dict
                rmse=self.train_model(iterations, early_stop)
                if rmse<best_rmse:
                    best_rmse=rmse
                    best_value=val
            best_values[key]=best_value
        return best_values

data_set=DataSet()


print(data_set.R_train.shape)
print(data_set.R_validation.shape)

_users=data_set.index_to_user.keys()
_items=data_set.index_to_item.keys()

#Create copy of dfs
train=data_set.train.copy()
validation=data_set.validation.copy()
test=data_set.test.copy()


# Convert original user and items to their mappings
train=convert_to_mappings(train, data_set.user_to_index, data_set.item_to_index)
validation=convert_to_mappings(validation, data_set.user_to_index, data_set.item_to_index)
test=convert_to_mappings(test, data_set.user_to_index, data_set.item_to_index)


d=3
m=len(_users)
n=len(_items)
iters=10
lu=0.01
li=0.01
lbu=0.01
lbi=0.01
r_train,r_valid=data_set.R_train,data_set.R_validation

#print((r_train>0).shape, (r_valid>0).shape)
#exit()
model=Model(latent_dim=d, num_users=m, num_items=n, ranking_matrix_train=r_train, ranking_matrix_validation=r_valid, l_users=lu, l_items=li, l_bias_users=lbu,  l_bias_items=lbi)



start = time.time()

model.train_model(iterations=10, early_stop=10)

test=convert_to_mappings(test, data_set.user_to_index, data_set.item_to_index)
test['yhat']=test.apply(lambda row: model.predict(row.User_ID_Alias, row.Movie_ID_Alias), axis=1)

exit()


d={'d': [3,4], 'lu': [0.01,0.05], 'li': [0.01], 'lbu': [0.01,0.05], 'lbi':[0.01]}
q=model.tune(iterations=10, early_stop=10, tuning_dict=d)
print(q)
exit()




