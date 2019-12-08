import pandas as pd
from data_preprocess.preprocess_utils import *
import numpy as np
from os.path import dirname, abspath, join

class DataSet():
    """ A class used to represent a dataset"""
    def __init__(self):
        #
        parent_path= dirname(dirname(abspath(__file__)))

        self.train = pd.read_csv(join(parent_path,'resources/Train.csv'))
        self.validation = pd.read_csv(join(parent_path,'resources/Validation.csv'))
        self.test = pd.read_csv(join(parent_path,'resources/Test.csv'))
        # Get user-index, index-user, item-index, index-item mappings  from train set
        self.user_to_index, self.index_to_user=self.create_mapping(self.train.User_ID_Alias.unique())
        self.item_to_index, self.index_to_item=self.create_mapping(self.train.Movie_ID_Alias.unique())
        # Get user-items and item-users mapping
        self.user_items=self.create_user_items_mapping()
        self.item_users=self.create_item_users_mapping()
        # Create rating matrix
        self.R_train, self.R_validation=self.create_rating_matrix()
        # For SGD we hold the whole dataset as numpy for better performance
        self.D_arr_train=self.train.to_numpy()
        self.D_arr_validate=self.validation.to_numpy()



    def create_mapping(self, lst):
        """ Create mapping between users/items and their indexes
            :param lst: list of values to map
            :return:    dict <item:index>
            :return:    dict <index:item>"""

        item_to_index = {v: k for k,v in enumerate(lst)}
        index_to_item = {v: k for k, v in item_to_index.items()}
        return item_to_index, index_to_item

    def create_user_items_mapping(self):
        """ Creates a mapping <key: user_index, value: relevant items, 2d array [item_index, rating]>

            :return:    dict <item:index>"""

        train=convert_to_mappings(self.train.copy(), self.user_to_index, self.item_to_index)
        d={}
        for user in self.index_to_user.keys():
          d[user]=(train[train.User_ID_Alias==user][['Movie_ID_Alias', 'Ratings_Rating']]).values
        return d

    def create_item_users_mapping(self):
        """ Creates a mapping  <key: item_index, value: relevant users, 2d array [user_index, rating]>

            :return:    dict """

        train=convert_to_mappings(self.train.copy(), self.user_to_index, self.item_to_index)
        d={}
        for item in self.index_to_item.keys():
            d[item]=(train[train.Movie_ID_Alias==item][['User_ID_Alias', 'Ratings_Rating']]).values
        return d


    def create_rating_matrix(self):
        """ Create rating matrix on training set

            :return:    np.array two dimensional array, rows=users columns=items values=ratings
            :return:    np.array two dimensional array, rows=users columns=items values=ratings
            """




        # Create R train

        #convert to indexes
        df=convert_to_mappings(self.train.copy(), self.user_to_index, self.item_to_index)
        r_train=df.pivot(index='User_ID_Alias', columns='Movie_ID_Alias', values='Ratings_Rating').fillna(0).values

        # Create R validation

        # convert to indexes
        df = convert_to_mappings(self.validation.copy(), self.user_to_index, self.item_to_index)
        #remove entries from validation set that did not appear in training set
        df=df[(df.User_ID_Alias > -1) & (df.Movie_ID_Alias >-1 )]


        #init r_validation in same size as r_train
        r_validation=np.zeros(shape=r_train.shape)
        # update r_i,j according to validation data
        for row in df.iterrows():
            r_validation[row[1].User_ID_Alias, row[1].Movie_ID_Alias]=row[1].Ratings_Rating
        r_validation=np.nan_to_num(r_validation)

        return r_train, r_validation


    def check_for_new(self):
        """ Check what users/items are in the validation and test set and do not appear in train set"""

        print('Items from Test set that do not appear in Train set')
        print(set_diff(self.test.Movie_ID_Alias, self.train.Movie_ID_Alias))
        print('Items from Validation set that do not appear in Train set')
        print(set_diff(self.validation.Movie_ID_Alias, self.train.Movie_ID_Alias))
        print('Users from Test set that do not appear in Train set')
        print(set_diff(self.test.User_ID_Alias, self.train.User_ID_Alias))
        print('Users from Validation set that do not appear in Train set')
        print(set_diff(self.validation.User_ID_Alias, self.train.User_ID_Alias))

















