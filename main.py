import sys
import os
import time
import argparse


def args_parsing():

    parser = argparse.ArgumentParser()

    parser.add_argument("-t", "--tune", action='store_true')
    parser.add_argument("-l", "--load", action='store_true')

    return parser.parse_args()



if __name__ == '__main__':
    args = args_parsing()

    main_path = os.path.dirname(os.path.realpath(__file__))  # Get full path of main.py

    # Add packages to $PYTHONPATH
    sys.path.append(os.path.join(main_path, "ALS"))
    sys.path.append(os.path.join(main_path, "resources"))
    sys.path.append(os.path.join(main_path, "data_preprocess"))

    from data_preprocess.preprocess import DataSet
    from data_preprocess.preprocess_utils import *
    from ALS.als import Model
    from resources import config

    if args.load:
        print('Loading pretraind DataSet()')

        path_to_prepared_data=os.path.join(main_path,'resources/prepared_data.pkl')
        data_set=load_data_set(path_to_prepared_data)
    else:
        data_set = DataSet()
        path_to_prepared_data = os.path.join(main_path, 'resources/prepared_data.pkl')
        save_data_set(data_set, path_to_prepared_data)



    start = time.time()



    # Init Hyper params

    d=config.d #latent dim
    iters=config.iters #iterations
    early_stop=config.early_stop #number of unimproved iterations to break program
    lu=config.lu #lambda users
    li=config.li #lambda items
    lbu=config.lbu #lambda bias users
    lbi=config.lbi #lambda bias items


    #get R matrices
    r_train,r_valid=data_set.R_train,data_set.R_validation
    # Init Model
    model=Model(latent_dim=d, user_items=data_set.user_items, item_users=data_set.item_users, ranking_matrix_train=r_train, ranking_matrix_validation=r_valid, l_users=lu, l_items=li, l_bias_users=lbu,  l_bias_items=lbi)

    if args.tune:
        # Tune hyperparams
        print('Starting Hyper Param Tuning')

        tuning_results=model.tune(iterations=100, early_stop=10, tuning_dict=config.tuning_dict)
        model=Model(latent_dim=tuning_results['d'], user_items=data_set.user_items, item_users=data_set.item_users, ranking_matrix_train=r_train, ranking_matrix_validation=r_valid, l_users=tuning_results['lu'], l_items=tuning_results['li'], l_bias_users=tuning_results['lbu'],  l_bias_items=tuning_results['lbi'])

    # Train model
    model.train_model(iters, early_stop)


    # Tune hyperparams

    #tuning_dict={'d': [8], 'lu': [100,90,80,50], 'li': [0.5], 'lbu': [10,8,5,3], 'lbi':[0.5]}
    #tuning_results=model.tune(iterations=100, early_stop=10, tuning_dict=tuning_dict)
    #print(tuning_results)

    #exit()


    #Predict on train set

    test=data_set.validation.copy()
    test=convert_to_mappings(test, data_set.user_to_index, data_set.item_to_index) # Convert original user and items to their mappings
    yhat=test.apply(lambda row: model.predict(row.User_ID_Alias, row.Movie_ID_Alias), axis=1)


    test=data_set.test.copy()
    test['Rating']=yhat
    test.to_csv(os.path.join(main_path, "resources/results.csv"),index=False)

    end = time.time()

    print('Wall time is {}'.format(end-start))







