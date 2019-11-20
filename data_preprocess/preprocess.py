import pandas as pd



class DataSet():
    def __init__(self):
        # Read data
        self.train = pd.read_csv('../resources/Train.csv')
        self.validation = pd.read_csv('../resources/Validation.csv')
        self.test = pd.read_csv('../resources/Test.csv')
        # Get mapping for users and items from train set
        self.user_to_index, self.index_to_user=self.create_mapping(self.train.User_ID_Alias.unique())
        self.item_to_index, self.index_to_item=self.create_mapping(self.train.Movie_ID_Alias.unique())
        # Create rating matrix
        self.R=self.create_rating_matrix()


    def create_mapping(self, lst):
        """ Create mapping between users/items and thier indexes
            :param lst: list of values to map
            :return:    dict <item:index>
            :return:    dict <index:item>"""

        item_to_index = {v: k for k,v in enumerate(lst)}
        index_to_item = {v: k for k, v in item_to_index.items()}
        return item_to_index, index_to_item

    def create_rating_matrix(self):
        """ Create rating matrix on training set
            :return:    np.array two dimensional array, rows=users columns=items values=ratings"""

        df=self.train.copy()
        #convert to indexes
        df.User_ID_Alias=df.User_ID_Alias.apply(lambda x: self.user_to_index[x])
        df.Movie_ID_Alias=df.Movie_ID_Alias.apply(lambda x: self.item_to_index[x])
        #create matrix
        return df.pivot(index='User_ID_Alias', columns='Movie_ID_Alias', values='Ratings_Rating').values


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



def set_diff(lst_a, lst_b):
    """ Calculate a-b. Meaning: items from set a that do not appear in set b"""
    return set(lst_a).difference(set(lst_b))










