import numpy as np
import random
import pandas as pd
from data_preprocess import preprocess as pre
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from scipy.optimize import minimize

def data_to_index(dataset, tune=None):
    """ convert items and users to indexes in test and validation sets.
    if item/user does not exist in the validation set then remove it"""

    for i in range(0, dataset.D_arr_train.shape[0]):
        dataset.D_arr_train[i, 0] = dataset.user_to_index[dataset.D_arr_train[i, 0]]
        dataset.D_arr_train[i, 1] = dataset.item_to_index[dataset.D_arr_train[i, 1]]

    rows_2_del_lst = []
    for j in range(0, dataset.D_arr_validate.shape[0]):
        if dataset.D_arr_validate[j, 0] in dataset.user_to_index.keys():
            dataset.D_arr_validate[j, 0] = dataset.user_to_index[dataset.D_arr_validate[j, 0]]
        else:
            rows_2_del_lst.append(j)
            print("user removed: ", dataset.D_arr_validate[j, 0])
            continue

        if dataset.D_arr_validate[j, 1] in dataset.item_to_index.keys():
            dataset.D_arr_validate[j, 1] = dataset.item_to_index[dataset.D_arr_validate[j, 1]]
        else:
            print("item removed: ", dataset.D_arr_validate[j, 1])
            rows_2_del_lst.append(j)

    if rows_2_del_lst:
        dataset.D_arr_validate = np.delete(dataset.D_arr_validate, rows_2_del_lst, axis=0)


def get_biased_params(hyparams, dataset, mean, sigma, mu):
    """ get initial trained biased params (b_m, b_n) for the model"""

    # create inital random biased parameters
    np.random.seed(10)
    b_m = np.random.normal(mean, sigma, dataset.R_train.shape[0])
    b_n = np.random.normal(mean, sigma, dataset.R_train.shape[1])
    a = hyparams["a"]
    improve_counter, iter_num = 0, 0
    prev_rmse_v = 1000
    # do SGD only for biased params:
    while True:
        for i in range(0, dataset.D_arr_train.shape[0]):
            user_idx = dataset.D_arr_train[i, 0]
            item_idx = dataset.D_arr_train[i, 1]
            e_mn = dataset.D_arr_train[i, 2] - mu - b_m[user_idx] - b_n[item_idx]

            # derivative steps:
            b_m[user_idx] = b_m[user_idx] + a * (e_mn - 0.1 * b_m[user_idx])
            b_n[item_idx] = b_n[item_idx] + a * (e_mn - 0.1 * b_n[item_idx])

        a = 0.9 * a
        mae_v, rmse_v, r_2 = validation_handling(dataset, None, None, b_m, b_n, hyparams['mu'])
        # with no improvement: add 1 to counter
        if rmse_v >= prev_rmse_v:
            no_improve_counter += 1
        else:
            no_improve_counter = 0
        print("train biased params: ", "iter:", iter_num,"prev_rmse_v: ", prev_rmse_v, "curr_rmse: ", rmse_v, "cur_mae: ", mae_v,
              "cur_r_2: " , r_2,  "no_improve_counter: ", no_improve_counter)
        prev_rmse_v = rmse_v
        iter_num += 1
        # early stop if we don't get improvement after 10 iterations
        if no_improve_counter == 10:
            break

    return b_m, b_n


def get_validation_results(dataset, r, b_m, b_n , mu):
    """return a matrix(user,item,original rate, predicted rate)
    with the rate prediction of each user and item (in the validation set)"""
    res_arr = np.zeros((dataset.D_arr_validate.shape[0], 4))
    for i in range(0, dataset.D_arr_validate.shape[0]):

        user_idx = dataset.D_arr_validate[i, 0]
        item_idx = dataset.D_arr_validate[i, 1]
        r_ui = r[user_idx, item_idx] + mu + b_m[user_idx] + b_n[item_idx]
        # if r_ui is below 1 then set to 1. if it above 5 then set it to 5
        res_arr[i, :] = [user_idx, item_idx, dataset.D_arr_validate[i, 2], min(max(1, r_ui), 5)]

    return res_arr


def test_validation_results(res_arr):
    """ check the validation prediction from the matrix of the results. then return MAE, RMSE and R squared"""

    actual = res_arr[:, 2]
    predicted = res_arr[:, 3]

    rmse_v = np.sqrt(mean_squared_error(actual, predicted))
    r_2 = r2_score(actual, predicted)
    mae_v = mean_absolute_error(actual, predicted)
    return mae_v, rmse_v, r_2



def validation_handling(dataset, u, v, b_m, b_n, mu):
    """ validation handling: first predict validation set according to current U,V,B_M, B_N.
                             Then return evaluation with MAE, RMSE and R SQUARED"""

    if u is None and v is None:
        r = np.zeros((dataset.R_train.shape[0], dataset.R_train.shape[1]))
    else:
        r = np.dot(u, v.transpose())
    res_arr = get_validation_results(dataset, r, b_m, b_n, mu)
    mae_v, rmse_v, r_2 = test_validation_results(res_arr)

    return round(mae_v, 5) ,round(rmse_v, 6), round(r_2, 4)


def generate_hyperparams(hyparams):

    hyparams['a'] = round(random.uniform(0.001, 0.2), 3)
    hyparams['l_u'], hyparams['l_v'] = round(random.uniform(0.001, 0.3), 3), round(random.uniform(0.001, 0.3), 3)
    hyparams['l_bm'], hyparams['l_bn'] = round(random.uniform(0.001, 0.3), 3), round(random.uniform(0.001, 0.3), 3)
    hyparams['k'] = random.randrange(6, 25)


def tune_hyparams_handling(dataset, hyparams, b_m, b_n, method= 'random'):
    """ tune hyperparameters according to strategy:
           1. generate hyperparameters randomly for 20 times / Nelder Mead method
           2. train the model with the selected hyperparameters
           3. if the strategy is random - check for best results according to RMSE
           4. return best results"""
    results_lst = []
    best_rmse = {}

    if method == 'random':
        print("tune hyperparameters started, method - random:")
        rounds_num = 20
        for i in range(rounds_num):
            print("round {} out of {} started:".format(i+1, rounds_num))
            generate_hyperparams(hyparams)
            u, v, b_m, b_n, mae_v, rmse_v, r_2 = sgd_training(dataset, hyparams, b_m, b_n)
            res = {'round_num': i, 'mae_v': mae_v, 'rmse_v': rmse_v, 'r_2': r_2, 'u': u, 'v': v, 'b_m': b_m, 'b_n': b_n}
            results_lst.append(res)

            if i == 0:
                best_rmse = res
            else:
                if res['rmse_v']< best_rmse['rmse_v']:
                    best_rmse = res
            print('cur_res: ', res['rmse_v'])
            print('best_rmse: ', best_rmse['rmse_v'])
        return best_rmse
    #use Nelder mead method:
    else:
        print("tune hyperparameters started, method - Nelder Mead:")
        x0 = np.array([0.24499584, 0.10445769, 0.03893219, 0.11829654, 0.10579032])
        # save local in order to run only once while doing nelder mid
        np.savetxt("b_m.csv", b_m, delimiter=",")
        np.savetxt("b_n.csv", b_n, delimiter=",")
        res = minimize(objective, x0, method='nelder-mead', options={"disp": True})
        print("best hyperparams: ", res.x)
        hyparams['a'], hyparams['l_u'], hyparams['l_v'] = res.x[0], res.x[1], res.x[2]
        hyparams['l_bm'], hyparams['l_bn'] = res.x[3], res.x[4]
        u, v, b_m, b_n, mae_v, rmse_v, r_2 = sgd_training(dataset, hyparams, b_m, b_n)
        best_rmse = {'round_num': None, 'mae_v': mae_v, 'rmse_v': rmse_v, 'r_2': r_2, 'u': u, 'v': v, 'b_m': b_m, 'b_n': b_n}
        return best_rmse

def sgd_training(dataset, hyparams, b_m, b_n):
    """ training the model with SGD method"""
    print(hyparams)
    # create random U and V according to selected K from normal distribution (mean=0, sigma= 0.1)
    np.random.seed(10)
    u = np.random.normal(hyparams['mean'], hyparams['sigma'], size=(dataset.R_train.shape[0], hyparams["k"]))
    v = np.random.normal(hyparams['mean'], hyparams['sigma'], size=(dataset.R_train.shape[1], hyparams["k"]))
    a = hyparams["a"]
    no_improve_counter, iter_num = 0, 0
    prev_rmse_v = 1000000
    while True:
        for i in range(0, dataset.D_arr_train.shape[0]):

            user_idx = dataset.D_arr_train[i, 0]
            item_idx = dataset.D_arr_train[i, 1]
            u_m = np.array(u[user_idx, :])
            v_n = np.array(v[item_idx, :])

            e_mn_t = dataset.D_arr_train[i, 2] - np.dot(u_m, v_n) - hyparams['mu'] - b_m[user_idx] - b_n[item_idx]

            # derivative steps:
            u[user_idx, :] = u_m + a * (e_mn_t * v_n - hyparams["l_u"] * u_m)
            v[item_idx, :] = v_n + a * (e_mn_t * u_m - hyparams["l_v"] * v_n)
            b_m[user_idx] = b_m[user_idx] + a * (e_mn_t - hyparams["l_bm"] * b_m[user_idx])
            b_n[item_idx] = b_n[item_idx] + a * (e_mn_t - hyparams["l_bn"] * b_n[item_idx])

        a = 0.95 * a
        # when a full round is complete, check results in validation set:
        mae_v, rmse_v, r_2 = validation_handling(dataset, u, v, b_m, b_n, hyparams['mu'])
        # with no improvement: add 1 to counter
        if rmse_v >= prev_rmse_v:
            no_improve_counter += 1
        else:
            no_improve_counter = 0
        print("train model: ", "iter:", iter_num,"prev_rmse_v: ", prev_rmse_v, "curr_rmse: ", rmse_v, "cur_mae: ", mae_v,
              "cur_r_2: " , r_2,  "no_improve_counter: ", no_improve_counter)
        prev_rmse_v = rmse_v
        iter_num += 1
        # early stop if we don't get improvement after 10 iterations
        if no_improve_counter == 20:
            break
    return u, v, b_m, b_n, mae_v, rmse_v, r_2


def train_sgd(dataset, hyparams, b_m, b_n, tune):
    """ SGD training handling. return the final result of all of the parameters and Hyperparameters in order to
    predict test set  """
    # if tune hypereparamters is needed:
    if tune:
        # tune hyperparameters according to method {'random', 'nelder mead'}
        res = tune_hyparams_handling(dataset, hyparams, b_m, b_n, method=hyparams["strategy"])
    # if tune hypereparamters is not necessary and we use hyperparams from config:
    else:
        u, v, b_m, b_n, mae_v, rmse_v, r_2 = sgd_training(dataset, hyparams, b_m, b_n)

        res = {'round_num': None, 'mae_v': mae_v, 'rmse_v': rmse_v, 'r_2': r_2, 'u': u, 'v': v, 'b_m': b_m, 'b_n': b_n}
    # return params and hyperparams
    final_res = {**res, **hyparams}

    return final_res


def get_res_for_new(dataset, user, item, r, mu, b_m, b_n):
    """ return prediction for cold user/item"""

    # if the item is cold but the user exists, then return the average prediction of the user's rates:
    if item not in dataset.item_to_index.keys() and user in dataset.user_to_index.keys():
        x = r[dataset.user_to_index[user], :] + mu + b_m[dataset.user_to_index[user]]
        print("cold item:", item)
        return x.mean()
    # if the user is cold but the item exists, then return the average prediction of the item's rates:
    elif item in dataset.item_to_index.keys() and user not in dataset.user_to_index.keys():
        x = r[:, dataset.item_to_index[item]] + mu + b_n[dataset.item_to_index[item]]
        print("cold user:", user)
        return x.mean()
    # both user and item are cold:
    else:
        print("cold user and item:", user, item)
        return mu


def get_test_results(dataset, r, b_m, b_n , mu):
    """return an array(user,item,prediction) with the rate of each user and item (in the test set)"""

    test_arr = dataset.test.to_numpy()
    res_arr = np.zeros((test_arr.shape[0], 3))

    for i in range(0, test_arr.shape[0]):
        user = test_arr[i, 0]
        item = test_arr[i, 1]
        if user in dataset.user_to_index.keys() and item in dataset.item_to_index.keys():
            user_idx = dataset.user_to_index[user]
            item_idx = dataset.item_to_index[item]
            r_ui = r[user_idx, item_idx] + mu + b_m[user_idx] + b_n[item_idx]
        else:
            r_ui = get_res_for_new(dataset, user, item, r, mu, b_m, b_n)

        res_arr[i, :] = [user, item, min(max(1, r_ui), 5)]

    return res_arr


def predict_test(dataset, res, path):
    """ test set prediction: first predict test set, then create a dataframe with the results and save to a csv file"""

    print("hyperparams for predict test: ", res['a'], res['l_u'], res['l_v'], res['l_bm'], res['l_bn'], res['k'])
    r = np.dot(res['u'], res['v'].transpose())
    res_arr = get_test_results(dataset, r, res['b_m'], res['b_n'], res['mu'])
    df = pd.DataFrame(res_arr, columns=['User_ID_Alias', 'Movie_ID_Alias', 'Ratings_Rating'])
    df['User_ID_Alias'] = df['User_ID_Alias'].apply(lambda x: int(x))
    df['Movie_ID_Alias'] = df['Movie_ID_Alias'].apply(lambda x: int(x))
    df.to_csv(path, index=False)


def objective(x):
    from resources import config
    print("Nelder Mead iteration started for hyperparamters:", x)
    b_m, b_n = np.genfromtxt('b_m.csv', delimiter=','), np.genfromtxt('b_n.csv', delimiter=',')
    dataset = pre.DataSet()
    data_to_index(dataset)
    hyparams = config.hyperparams_sgd

    np.random.seed(10)
    u = np.random.normal(hyparams['mean'], hyparams['sigma'], size=(dataset.R_train.shape[0], hyparams["k"]))
    v = np.random.normal(hyparams['mean'], hyparams['sigma'], size=(dataset.R_train.shape[1], hyparams["k"]))
    a = x[0]
    no_improve_counter, iter_num = 0, 0
    prev_rmse_v = 1000000
    while True:
        for i in range(0, dataset.D_arr_train.shape[0]):
            user_idx = dataset.D_arr_train[i, 0]
            item_idx = dataset.D_arr_train[i, 1]
            u_m = np.array(u[user_idx, :])
            v_n = np.array(v[item_idx, :])

            e_mn_t = dataset.D_arr_train[i, 2] - np.dot(u_m, v_n) - hyparams['mu'] - b_m[user_idx] - b_n[item_idx]

            # derivative steps:
            u[user_idx, :] = u_m + a * (e_mn_t * v_n - x[1] * u_m)
            v[item_idx, :] = v_n + a * (e_mn_t * u_m - x[2] * v_n)
            b_m[user_idx] = b_m[user_idx] + a * (e_mn_t - x[3] * b_m[user_idx])
            b_n[item_idx] = b_n[item_idx] + a * (e_mn_t - x[4] * b_n[item_idx])

        a = 0.95 * a
        # when a full round is complete, check results in validation set:
        mae_v, rmse_v, r_2 = validation_handling(dataset, u, v, b_m, b_n, hyparams['mu'])
        # with no improvement: add 1 to counter
        if rmse_v >= prev_rmse_v:
            no_improve_counter += 1
        else:
            no_improve_counter = 0
        # print("train model: ", "iter:", iter_num, "prev_rmse_v: ", prev_rmse_v, "curr_rmse: ", rmse_v, "cur_mae: ",
        #       mae_v,"cur_r_2: ", r_2, "no_improve_counter: ", no_improve_counter)
        prev_rmse_v = rmse_v
        iter_num += 1
        # early stop if we don't get improvement after 10 iterations
        if no_improve_counter == 10:
            break
    print("iteration finished for:", x, "rmse: ", rmse_v, "r_2: ", r_2)
    return rmse_v