import numpy as np
import random
import pandas as pd


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


def get_biased_params(dataset, mean, sigma, mu):
    """ get initial trained biased params (b_m, b_n) for the model"""

    # create inital random biased parameters
    b_m = np.random.normal(mean, sigma, dataset.R_train.shape[0])
    b_n = np.random.normal(mean, sigma, dataset.R_train.shape[1])
    a = 0.1
    improve_counter, iter_num = 0, 0
    prev_rmse = 1000
    # do SGD only for biased params:
    while True:
        err_power2 = 0
        for i in range(0, dataset.D_arr_train.shape[0]):
            user_idx = dataset.D_arr_train[i, 0]
            item_idx = dataset.D_arr_train[i, 1]
            e_mn = dataset.D_arr_train[i, 2] - mu - b_m[user_idx] - b_n[item_idx]

            err_power2 += e_mn ** 2
            # derivative steps:
            b_m[user_idx] = b_m[user_idx] + a * (e_mn - 0.1 * b_m[user_idx])
            b_n[item_idx] = b_n[item_idx] + a * (e_mn - 0.1 * b_n[item_idx])

        rmse = round(np.sqrt(err_power2 / dataset.D_arr_train.shape[0]), 4)
        iter_num += 1
        print("train biased params: ", "iter:", iter_num, "prev_rmse: ", prev_rmse, "curr_rmse: ", rmse,
              "no_improve_counter: ", improve_counter)
        if rmse >= prev_rmse:
            improve_counter += 1
        # early stop if we don't get improvement after 5 iterations
        if improve_counter == 5:
            break
        prev_rmse = rmse
    return b_m, b_n


def get_validation_results(dataset, r, b_m, b_n , mu):
    """return a matrix(user,item,original rate, predicted rate)
    with the rate prediction of each user and item (in the validation set)"""
    res_arr = np.zeros((dataset.D_arr_validate.shape[0], 4))
    for i in range(0, dataset.D_arr_validate.shape[0]):

        user_idx = dataset.D_arr_validate[i, 0]
        item_idx = dataset.D_arr_validate[i, 1]
        r_ui = r[user_idx, item_idx] + mu + b_m[user_idx] + b_n[item_idx]
        # if r_ui is below 0 then set to 0. if it above 5 then set it to 5
        res_arr[i, :] = [user_idx, item_idx, dataset.D_arr_validate[i, 2], min(max(0, r_ui), 5)]

    return res_arr


def test_validation_results(res_arr, mu):
    """ check the validation prediction from the matrix of the results. then return MAE, RMSE and R squared"""
    ss_res = 0
    ss_tot = 0
    cum_abs_err = 0
    for i in range(0, res_arr.shape[0]):
        cum_abs_err += abs(res_arr[i, 2] - res_arr[i, 3])
        ss_res += (res_arr[i, 2] - res_arr[i, 3])**2
        ss_tot += (res_arr[i, 2] - mu)**2

    mae_v = cum_abs_err/res_arr.shape[0]
    rmse_v = np.sqrt(ss_res / res_arr.shape[0])
    r_2 = 1 - (ss_res/ss_tot)
    return mae_v, rmse_v, r_2


def validation_handling(dataset, u, v, b_m, b_n, mu):
    """ validation handling: first predict validation set according to current U,V,B_M, B_N.
                             Then return evaluation with MAE, RMSE and R SQUARED"""

    r = np.dot(u, v.transpose())
    res_arr = get_validation_results(dataset, r, b_m, b_n, mu)
    mae_v, rmse_v, r_2 = test_validation_results(res_arr, mu)

    return round(mae_v, 5) ,round(rmse_v, 6), round(r_2, 4)


def generate_hyperparams(hyparams):

    hyparams['a'] = round(random.uniform(0.001, 0.3), 3)
    hyparams['l_u'], hyparams['l_v'] = round(random.uniform(0.001, 0.3), 3), round(random.uniform(0.001, 0.3), 3)
    hyparams['l_bm'], hyparams['l_bn'] = round(random.uniform(0.001, 0.3), 3), round(random.uniform(0.001, 0.3), 3)
    hyparams['k'] = random.randrange(6, 25)


def tune_hyparams_handling(dataset, hyparams, b_m, b_n, tune):
    """ tune hyperparameters according to strategy:
           1. generate hyperparameters randomly for 20 times
           2. train the model with the selected hyperparameters
           3. check for best results according to RMSE
           4. return best results"""
    results_lst = []
    best_rmse = {}
    best_r_2 = {}

    for i in range(20):
        generate_hyperparams(hyparams)
        u, v, b_m, b_n, mae_v, rmse_v, r_2 = sgd_training(dataset, hyparams, b_m, b_n, tune)
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


def sgd_training(dataset, hyparams, b_m, b_n, tune):
    """ training the model with SGD method"""
    print(hyparams)
    # create random U and V according to selected K from normal distribution (mean=0, sigma= 0.1)
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

        a = 0.9 * a
        # when a full round is complete, check results in validation set:
        mae_v, rmse_v, r_2 = validation_handling(dataset, u, v, b_m, b_n, hyparams['mu'])
        # with no improvement: add 1 to counter
        if rmse_v >= prev_rmse_v :
            no_improve_counter += 1
        print("train model: ", "iter:", iter_num,"prev_rmse_v: ", prev_rmse_v, "curr_rmse: ", rmse_v, "cur_mae: ", mae_v,
              "cur_r_2: " , r_2,  "no_improve_counter: ", no_improve_counter)
        prev_rmse_v = rmse_v
        iter_num += 1
        # early stop if we don't get improvement after 10 iterations
        if no_improve_counter == 10:
            break
    return u, v, b_m, b_n, mae_v, rmse_v, r_2


def train_sgd(dataset, hyparams, b_m, b_n, tune):
    """ SGD training handling. return the final result of all of the parameters and Hyperparameters in order to
    predict test set  """
    # if tune hypereparamters is needed:
    if tune:
        res = tune_hyparams_handling(dataset, hyparams, b_m, b_n, tune)
    # if tune hypereparamters is not necessary and we use hyperparams from config:
    else:
        u, v, b_m, b_n, mae_v, rmse_v, r_2 = sgd_training(dataset, hyparams, b_m, b_n, tune)

        res = {'round_num': None, 'mae_v': mae_v, 'rmse_v': rmse_v, 'r_2': r_2, 'u': u, 'v': v, 'b_m': b_m, 'b_n': b_n}
    # return params and hyperparams
    final_res = {**res, **hyparams}

    return final_res


def get_res_for_new(dataset, user, item, r, mu, b_m, b_n):
    """ return prediction for cold user/item"""
    print(user, item)
    # if the item is cold but the user exists, then return the average prediction of the user's rates:
    if item not in dataset.item_to_index.keys() and user in dataset.user_to_index.keys():
        x = r[dataset.user_to_index[user], :] + mu + b_m[dataset.user_to_index[user]]
        print("cold item:", item)
        print(x.mean())
        return x.mean()
    # if the user is cold but the item exists, then return the average prediction of the item's rates:
    elif item in dataset.item_to_index.keys() and user not in dataset.user_to_index.keys():
        x = r[:, dataset.item_to_index[item]] + mu + b_n[dataset.item_to_index[item]]
        print("cold user:", user)
        print(x.mean())
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

        res_arr[i, :] = [user, item, min(max(0, r_ui), 5)]

    return res_arr


def predict_test(dataset, res, path):
    """ test set prediction: first predict test set, then create a dataframe with the results and save to a csv file"""

    print(res['a'], res['l_u'], res['l_v'], res['l_bm'], res['l_bn'], res['k'])
    r = np.dot(res['u'], res['v'].transpose())
    res_arr = get_test_results(dataset, r, res['b_m'], res['b_n'], res['mu'])
    df = pd.DataFrame(res_arr, columns=['User_ID_Alias', 'Movie_ID_Alias', 'Ratings_Rating'])
    df['User_ID_Alias'] = df['User_ID_Alias'].apply(lambda x: int(x))
    df['Movie_ID_Alias'] = df['Movie_ID_Alias'].apply(lambda x: int(x))
    df.to_csv(path, index=False)