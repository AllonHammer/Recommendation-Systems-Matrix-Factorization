import numpy as np
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import time

def init_variables(d,m,n):
    """ Init Variables
        :param d: int latent dimension
        :param m: int number of users in train set
        :param n: int number of items in train set

        :return np.array of U - user matrix (mXd)
        :return np.array of V - item matrix (nXd)
        :return np.array of Bm - user bias vector (mX1)
        :return np.array of Bn- item bias vector (nX1)

        :"""

    u = np.random.normal(0, 1 / np.sqrt(d), (m, d))
    v = np.random.normal(0, 1 / np.sqrt(d), (n, d))
    bm = np.random.normal(0, 0.1, (m, 1))
    bn = np.random.normal(0, 0.1, (n, 1))
    return u,v,bm,bn


def update_Um(user, v, r, user_biases,item_biases, lu,mu):
    """Calculate Um= (V.t * V +lambda_users *I)^-1 * V.t * (rmn-bm-bn-mu) )

            :param user: int index of the user
            :param v: np.array two dimensional matrix (items, d)
            :param r: np.array two dimensional ranking matrix (users, items)
            :param user_biases: np.array one dimensional array of user biases (users,)
            :param item_biases: np.array one dimensional array of item biases (items,)
            :param lu: float regularization term for users
            :param mu: float mean coefficient of the data set


            :return    np.array of Um (1X d) the latent vector of user m """


    #Get the list of items that the user ranked
    start = time.time()

    relevant_items=(r>0)[user,:]
    end = time.time()
    print('Get index', np.round(end - start,5))

    #Get relevnt rankings, latent vectors and biases

    start = time.time()

    ru_ = r[user, relevant_items]  # (relevant_items, )


    end = time.time()
    print('Filter R ', np.round(end - start,5))

    start = time.time()
    v_ = v[relevant_items, :]  # (relevant items X d)

    end = time.time()
    print('Filter V ', np.round(end - start,5))

    start = time.time()

    bm = user_biases[user]  # (scalar)
    bn = item_biases[relevant_items].reshape(-1)  # (relevant_items, )

    end = time.time()
    print('Filter Biases ', np.round(end - start,5))

    start = time.time()

    #Calc Um
    i=np.eye(v_.shape[1]) # Identity matrix in the size of (dXd)
    first_term=np.linalg.inv(np.dot(v_.T, v_)+lu*i) #(dXrelevant items) * (relevant_itemsXd)+ I = dXd
    second_term=np.dot(v_.T, (ru_-bm-bn-mu)) #(dXrelevant items) * ((relevant_items X1)- (relevant_items X 1)- (1X1)- (1X1))= dX1

    Um=np.dot(first_term, second_term) #(dX1)
    end = time.time()
    print('calcs  ', np.round(end - start,5))

    return Um



def update_Vn(item, u, r, user_biases,item_biases, li,mu):
    """Calculate Vn= (U.t * U +lambda_items *I)^-1 * U.t * (rmn-bm-bn-mu) )

            :param item: int index of the item
            :param u: np.array two dimensional matrix (users, d)
            :param r: np.array two dimensional ranking matrix (users, items)
            :param user_biases: np.array one dimensional array of user biases (users,)
            :param item_biases: np.array one dimensional array of item biases (items,)
            :param li: float regularization term for items
            :param mu: float mean coefficient of the data set


            :return    np.array of Vn (1X d) the latent vector of item n """

    #Get the list of users that have ranked the item
    relevant_users =  (r>0)[:, item]


    #Get relevnt rankings, latent vectors and biases
    r_v = r[relevant_users, item]  # (relevant_users, )
    u_ = u[relevant_users, :]  # (relevant users X d)
    bm = user_biases[relevant_users].reshape(-1)  # (relevant_users, )
    bn = item_biases[item]  # (scalar )

    #Calc Um
    i=np.eye(u_.shape[1]) # Identity matrix in the size of (dXd)
    first_term=np.linalg.inv(np.dot(u_.T, u_)+li*i) #(dXrelevant_users) * (relevant_usersXd)+ I = dXd
    second_term=np.dot(u_.T, (r_v-bm-bn-mu)) #(dXrelevant dXrelevant_users) * ((dXrelevant_users X1)- (dXrelevant_users X 1)- (1X1)- (1X1))= dX1

    Vn=np.dot(first_term, second_term) #(dX1)

    return Vn



def update_bm(user, u,v, r, item_biases, lbu,mu):
    """Calculate Um= (|D|+lbu)^-1 *  (rmn-Um*Vn.T-bn-mu) )

            :param user: int index of the user
            :param u: np.array two dimensional matrix (users, d)
            :param v: np.array two dimensional matrix (items, d)
            :param r: np.array two dimensional ranking matrix (users, items)
            :param item_biases: np.array one dimensional array of item biases (items,)
            :param lbu: float regularization term for the bias of the users
            :param mu: float mean coefficient of the data set


            :return    float value of bm the bias of user m """

    # Get the list of items that the user ranked
    relevant_items = (r > 0)[user, :]

    #Get relevnt rankings, latent vectors and biases
    ru_ = r[user, relevant_items]  # (relevant_items, )
    u_ = u[user,:].reshape(-1) # (d,)
    v_ = v[relevant_items, :]  # (relevant items X d)
    bn = item_biases[relevant_items].reshape(-1)  # (relevant_items, )

    #Calc bm
    first_term=np.full(shape=v_.shape[0], fill_value= 1/(v_.shape[0]+lbu)) #  we matrix in the size of (relevant_items,),
    # since the second term is also in the same dim and we will get 1X1 which is desired for bm. Remember we need the inverse of (|D|+ lbu) but since it is just a scalar we can divide by it
    second_term=(ru_- np.dot(u_,v_.T)-bn-mu) #(relevant items.) - ((d, )X (dXrelevant_items)- (relevant_items,)- (1X1))= (relevant_items,)

    bm=np.dot(first_term, second_term) #(1X1)
    return bm


def update_bm_new(user, u,v, r, item_biases, lbu,mu):
    """Calculate Um= (|D|+lbu)^-1 *  (rmn-Um*Vn.T-bn-mu) )

            :param user: int index of the user
            :param u: np.array two dimensional matrix (users, d)
            :param v: np.array two dimensional matrix (items, d)
            :param r: np.array two dimensional ranking matrix (users, items)
            :param item_biases: np.array one dimensional array of item biases (items,)
            :param lbu: float regularization term for the bias of the users
            :param mu: float mean coefficient of the data set


            :return    float value of bm the bias of user m """

    # Get the list of items that the user ranked
    r=csr_matrix(r)
    relevant_items=r.getrow(user).nonzero()[1]

    #relevant_items = (r > 0)[user, :]

    #Get relevnt rankings, latent vectors and biases
    ru_ = r[user, relevant_items]  # (relevant_items, )
    ru_=ru_.todense()
    u_ = u[user,:].reshape(-1) # (d,)
    v_ = v[relevant_items, :]  # (relevant items X d)
    bn = item_biases[relevant_items].reshape(-1)  # (relevant_items, )
    print(ru_.T.flatten().shape)
    print(u_.shape)
    print(v_.shape)
    print(bn.shape)
    #Calc bm
    first_term=np.full(shape=v_.shape[0], fill_value= 1/(v_.shape[0]+lbu)) #  we matrix in the size of (relevant_items,),
    # since the second term is also in the same dim and we will get 1X1 which is desired for bm. Remember we need the inverse of (|D|+ lbu) but since it is just a scalar we can divide by it
    second_term=(ru_.T.reshape(-1)- np.dot(u_,v_.T)-bn-mu) #(relevant items.) - ((d, )X (dXrelevant_items)- (relevant_items,)- (1X1))= (relevant_items,)

    bm=np.dot(first_term, second_term) #(1X1)
    return bm

def update_bn(item, u,v, r, user_biases, lbi,mu):
    """Calculate Um= (|D|+lbi)^-1 *  (rmn-Um*Vn.T-bm-mu) )

            :param item: int index of the item
            :param u: np.array two dimensional matrix (users, d)
            :param v: np.array two dimensional matrix (items, d)
            :param r: np.array two dimensional ranking matrix (users, items)
            :param user_biases: np.array one dimensional array of item biases (users,)
            :param lbi: float regularization term for the bias of the items
            :param mu: float mean coefficient of the data set


            :return    float value of bn the bias of item n """

    # Get the list of users that have ranked the item
    relevant_users = (r > 0)[:, item]

    #Get relevnt rankings, latent vectors and biases
    r_v = r[relevant_users, item]  # (relevant_users, )
    u_ = u[relevant_users,:] # (relevant_users , d)
    v_ = v[item, :].reshape(-1)  # ( d, )
    bm = user_biases[relevant_users].reshape(-1)  # (relevant_items, )

    #Calc bn
    first_term=np.full(shape=u_.shape[0], fill_value= 1/(u_.shape[0]+lbi)) #  we matrix in the size of (relevant_users,),
    # since the second term is also in the same dim and we will get 1X1 which is desired for bn. Remember we need the inverse of (|D|+ lbi) but since it is just a scalar we can divide by it
    second_term=(r_v- np.dot(u_,v_.T)-bm-mu) #(relevant users) - ((d, )X (dXrelevant_userse)- (relevant_users,)- (1X1))= (relevant_users,)

    bn=np.dot(first_term, second_term) #(1X1)
    return bn



def calc_r2(r_true, r_pred):
    """calc R2 score --> Sigma (y_pred- y_mean)^2 / Sigma(y_true- y_mean)^2
        :param r_true: np.array two dimensional array, rows=users columns=items values=ratings
        :param r_pred: np.array two dimensional array, rows=users columns=items values=ratings

        :return float rmse
        """
    #idx = r_true > -1
    return r2_score(r_true, r_pred)

def calc_rmse(r_true, r_pred):
    """calc RMSE
        :param r_true: np.array two dimensional array, rows=users columns=items values=ratings
        :param r_pred: np.array two dimensional array, rows=users columns=items values=ratings

        :return float rmse
        """
    #idx=r_true>-1
    #mse=sum((r_pred - r_true) ** 2) / np.count_nonzero(r_true)
    mse=mean_squared_error(r_true, r_pred)

    return np.sqrt(mse)

def calc_mae(r_true, r_pred):

    """calc MAE
            :param r_true: np.array two dimensional array, rows=users columns=items values=ratings
            :param r_pred: np.array two dimensional array, rows=users columns=items values=ratings

            :return float mae
            """
    #idx = r_true > -1
    #mae=sum(abs(r_pred - r_true)) / np.count_nonzero(r_true)
    return mean_absolute_error(r_true, r_pred)


