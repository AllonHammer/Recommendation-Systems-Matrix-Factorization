
def set_diff(lst_a, lst_b):
    """ Calculate a-b. Meaning: items from set a that do not appear in set b"""
    return set(lst_a).difference(set(lst_b))


def convert_to_mappings(df, user_to_index, item_to_index):
    """ Convert to mappings if an Unknown user or item is in the dataset, mark it with -1"""
    df.User_ID_Alias = df.User_ID_Alias.apply(
        lambda x: -1 if x not in user_to_index.keys() else  user_to_index[x])
    df.Movie_ID_Alias = df.Movie_ID_Alias.apply(
        lambda x: -1 if x not in item_to_index.keys() else  item_to_index[x])

    return df

