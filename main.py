import sys
import os
import time

main_path = os.path.dirname(os.path.realpath(__file__)) # Get full path of main.py

# Add packages to $PYTHONPATH
sys.path.append(os.path.join(main_path, "ALS"))
sys.path.append(os.path.join(main_path, "resources"))
sys.path.append(os.path.join(main_path, "data_preprocess"))

from data_preprocess.preprocess import DataSet
from data_preprocess.preprocess_utils import *
from ALS.als import Model


start = time.time()

data_set=DataSet()

#path_to_prepared_data=os.path.join(main_path,'resources/prepared_data.pkl')
#data_set=load_data_set(path_to_prepared_data)


# Init Hyper params

d=8 #latent dim
iters=100 #iterations
early_stop=10 #number of unimproved iterations to break program
lu=100 #lambda users
li=0.5 #lambda items
lbu=10 #lambda bias users
lbi=0.5 #lambda bias items


#get R matrices
r_train,r_valid=data_set.R_train,data_set.R_validation
# Init Model
model=Model(latent_dim=d, user_items=data_set.user_items, item_users=data_set.item_users, ranking_matrix_train=r_train, ranking_matrix_validation=r_valid, l_users=lu, l_items=li, l_bias_users=lbu,  l_bias_items=lbi)
# Train model
model.train_model(iters, early_stop)


# Tune hyperparams

#tuning_dict={'d': [10,50,100], 'lu': [100,10,1,0.1], 'li': [100,10,1,0.1], 'lbu': [100,10,1,0.1], 'lbi':[100,10,1,0.1]}
#tuning_results=model.tune(iterations=100, early_stop=20, tuning_dict=tuning_dict)
#print(tuning_results)




#Predict on train set

test=data_set.validation.copy()
test=convert_to_mappings(test, data_set.user_to_index, data_set.item_to_index) # Convert original user and items to their mappings
yhat=test.apply(lambda row: model.predict(row.User_ID_Alias, row.Movie_ID_Alias), axis=1)


test=data_set.test.copy()
test['Rating']=yhat
test.to_csv(os.path.join(main_path, "resources/results.csv"),index=False)

end = time.time()

print('Wall time is {}'.format(end-start))
exit()







