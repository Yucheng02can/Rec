# -*- coding: utf-8 -*-
# @Time    : 2023/10/13 10:21
# @Author  : colagold
# @FileName: rs_explicit_tvt.py
import copy
import copy
import io
import logging
import os
import pickle
import types
import pandas as pd
import json
import numpy as np
import scipy.sparse as sp
from collections import Counter
# ATP: 用于推荐的(train,validate,test)实验验证
import yaml
from scipy.sparse import csr_matrix


class RSImplicitTVT():  # Divide the dataset into trainset validset and testset
    def __init__(self):
        self.train_percent = 0.8  # Training set ratio 8:1:1
        self.valid_percent = 0.1  # Validation set ratio


    def split(self, data) -> tuple:
        """
          :param data:  csr_matrix
                      example：uids iids   data
                          （ 0  ,  0 ）   1
                          （ 2  ,  2 ）   4
                          （ 5  ,  4 ）   8
          :return:（train_set,validate_set,test_set）
                  train_set/validate_set/test_set:sp.csr_matrix
                      example：uids iids   data
                          （ 0  ,  0 ）   1
                          （ 0  ,  1 ）   3
                          （ 2  ,  2 ）   4
                          （ 3  ,  3 ）   5
                          （ 5  ,  4 ）   5
        """
        # data.data[:]=1
        data = data.tocoo()
        csr_data=data.tocsr()
        data01 = copy.deepcopy(data)
        self.data_shape = data.shape
        # data.data = data.data.astype(bool).astype(int)
        #  data.sum(axis=0) sum by column, argsort() reverse order, .A to array
        #res = (data.sum(axis=0).argsort()).A
        # entries = list(zip(data.row,data.col,data.data))
        entries = list(zip(data01.row, data01.col, data01.data))
        np.random.seed(1)
        np.random.shuffle(entries)  # Generate a random list
        N = len(entries)


        train_list = entries[:int(N * self.train_percent)]  # 80 per cent as a training set
        test_list = entries[
                    int(N * self.train_percent):int(N * (self.train_percent + self.valid_percent))]  # 10% test: [0.8,0.9]
        valid_list = entries[int(N * (self.train_percent + self.valid_percent)):]
        # The COO format is suitable for sparse matrix creation and modification operations, while the
        # The LIL format is suitable for row operations on sparse matrices.



        coo=data
        # Generate training set Validation set Test set
        train_set = merge([train_list],self.data_shape)
        validate_set = merge([valid_list],self.data_shape)
        test_set = merge([test_list],self.data_shape).tolil()
        origion_train_set=train_set.tocsr()
        origion_test_set=test_set.tocsr()
        origion_val_set=validate_set.tocsr()

        # Count the number of user interactions in the training set ->dict user:count
        n_user, n_item = coo.shape
        train_item_intera_count = dict(Counter(train_set.tocoo().col).most_common(n_item)) # Used to divide the head and tail goods, orderly, from big to small arranged

        # Product popularity grouping, divided into the head and tail, the head 10%, the rest of the tail
        bias_group=self.get_bias_group(n_item,train_item_intera_count,test_list,self.data_shape)


        return train_set, origion_val_set, origion_test_set,bias_group


    def test(self, algorithm, ds, metrics,another=None, cb_progress=lambda x: None):
        '''
              algorithm: derived class of util.model.
            ds: is a two-dimensional sp.csr_matrix, [n_users,n_items], rows represent users, columns represent items
        '''
        # Divide training, validation, testing, and cold-start datasets
        process_path = f"../process_data/{another}"
        if not os.path.isdir(process_path):
            os.makedirs(process_path)
        process_data_path = os.path.join(process_path, another + ".pkl")
        if not os.path.isfile(process_data_path):  # Save the data division, cold start division and bias division after adding noise to save time.
            train_data, valid_data, test_data, cold_start_group, popularity_group = self.split(ds)
            process_data = {
                "train_data": train_data,
                "test_data": test_data,
                "valid_data": valid_data,

                "popularity_group": popularity_group
            }
            with open(process_data_path, 'wb') as f:
                pickle.dump(process_data, f)
        else:
            with open(process_data_path, 'rb') as f:
                data = pickle.load(f)
                train_data = data["train_data"]
                test_data = data["test_data"]
                valid_data = data["valid_data"]
                cold_start_group = data["cold_group"]
                popularity_group = data["popularity_group"]
                print("Processing data loaded successfully")



        train_data, valid_data, test_data,popularity_group = self.split(ds)
        # [0,16,32] => [0,16],[16,32],[32,]; [0,20]
        # The training process of the algorithm is run, passing the training set, but also the test set, and the first test metric.
        valid_funs = [m for m in metrics]

        algorithm.train(train_data, valid_data, valid_funs[0])
        pred = algorithm.predict(test_data)
        results = [m( test_data,pred) for m in metrics]
        # groundTruth , _ = test_data
        logging.info(f"{str(self)}") #Save the parameters in the protocol to the log.

        headers = [str(m) for m in metrics]
        ##Random Recommendations####################################################
        rating=np.unique(test_data.data[:])
        random_pre=copy.deepcopy(test_data)
        random_pre.data[:]=np.random.choice(rating,size=test_data.nnz,replace=True)
        random_results = [m(test_data,random_pre) for m in metrics]
        print("RANDOM_RESULTS="+str(dict(zip(headers, random_results))))
        logging.info("RANDOM_RESULTS="+str(dict(zip(headers, random_results))))

        ##Recommended by popularity
        popularity_pre = copy.deepcopy(test_data)
        popularity_pre=csr_col_norm(train_data,popularity_pre)
        popularity_results = [m(test_data, popularity_pre) for m in metrics]
        print("POPULARITY_RESULTS=" + str(dict(zip(headers, popularity_results))))
        logging.info("POPULARITY_RESULTS=" + str(dict(zip(headers, popularity_results))))


        bias_result = {}
        item_inter_plan = ['head', 'tail']
        for index, bias_group in enumerate(popularity_group):
            pred = algorithm.predict(bias_group)
            bias_result[item_inter_plan[index]] = [{str(m):m(bias_group, pred)} for m in metrics]

        logging.info("BIAS_RESULTS=" + str(bias_result))  # Preservation of indicators and results
        print("BIAS_RESULTS=" + str(bias_result))
        return dict(zip(headers, results))


    def class_name(self):
        # Model name
        return str(self.__class__)[8:-2].split('.')[-1].lower()

    def __str__(self):
        parameters_dic=copy.deepcopy(self.__dict__)
        parameters=get_parameters_js(parameters_dic)
        return dict_to_yamlstr({self.class_name():parameters}) # Output model name + parameters

    def get_noise_train_csr(self,n_user,coo,train_set,validate_set,test_set,split_arrays):
        all_items = set(coo.col)  # Treat all interacted items as a candidate set.
        tr_noise_samples = ([], [], [])  # The list of noise samples in the training set, which holds the noise samples user id, product id,rating
        construct_train_data = ([], [], [])
        cold_start_data = [([], [], []), ([], [], []), ([], [], [])]  ##The rest as a cold start test
        csr_data=train_set
        train_lil=train_set.tolil()
        np.random.seed(1)
        ratings = set(train_set.data)
        for u in range(n_user):  # Iterate over the users and add noise to the users in the training set
            tr_items = set(train_lil.rows[u])  # Get all the goods that user u has interacted with in the training set
            # 采样
            if u in split_arrays[0]:
                construct_tr_item = np.random.choice(list(tr_items), size=np.random.randint(1,6), replace=False)
                construct_train_data[0].extend([u] * len(construct_tr_item))
                construct_train_data[1].extend(construct_tr_item)
                for i in construct_tr_item:
                    construct_train_data[2].append(csr_data[u, i])

                cold_test_item = list(tr_items - set(construct_tr_item))
                cold_start_data[0][0].extend([u] * len(cold_test_item))  # Group I
                cold_start_data[0][1].extend(cold_test_item)
                for i in cold_test_item:
                    cold_start_data[0][2].append(csr_data[u, i])

            if u in split_arrays[1]:
                construct_tr_item = np.random.choice(list(tr_items), size=np.random.randint(6,11), replace=False)
                construct_train_data[0].extend([u] * len(construct_tr_item))
                construct_train_data[1].extend(construct_tr_item)
                for i in construct_tr_item:
                    construct_train_data[2].append(csr_data[u, i])

                cold_test_item = list(tr_items - set(construct_tr_item))
                cold_start_data[1][0].extend([u] * len(cold_test_item))  # Group I
                cold_start_data[1][1].extend(cold_test_item)
                for i in cold_test_item:
                    cold_start_data[1][2].append(csr_data[u, i])

            if u in split_arrays[2]:
                construct_tr_item = np.random.choice(list(tr_items), size=np.random.randint(16,21), replace=False)
                construct_train_data[0].extend([u] * len(construct_tr_item))
                construct_train_data[1].extend(construct_tr_item)
                for i in construct_tr_item:
                    construct_train_data[2].append(csr_data[u, i])

                cold_test_item = list(tr_items - set(construct_tr_item))
                cold_start_data[2][0].extend([u] * len(cold_test_item))  # Group I
                cold_start_data[2][1].extend(cold_test_item)
                for i in cold_test_item:
                    cold_start_data[2][2].append(csr_data[u, i])

            val_items = set(validate_set.rows[u])  # Get all the items that user u has interacted with in the validation set
            te_items = set(test_set.rows[u])  # Get all the items that user u has interacted with in the test set
            if len(tr_items) == 0:  # There are no commodities in this user's training set, then there is no need to add noise
                continue
            if len(val_items) == 0 and len(te_items) == 0:  # User u has no interaction goods in both the test and validation sets, then there is no need to generate negative samples
                continue  #

            neg_items = all_items - val_items - te_items - tr_items  # Get a collection of products that have not been visited by the user
            sample_num = min(len(neg_items), self.noise_ratio)  # Set the number of negative samples, if the user interacts with too many items, the set of unvisited items may be less than self.n_neg
            if sample_num == 0: continue  # The user has no uninteracted items and cannot test the user
            noise_users = [u] * sample_num  # User u's id duplicated self.neg times
            if len(te_items) > 0:  # Negative samples are generated if user u has interacting goods in the validation set, otherwise they are not generated
                noise_item = np.random.choice(list(neg_items), sample_num, replace=False)  # Non-repeating negative sampling
                tr_noise_samples[0].extend(noise_users)
                tr_noise_samples[1].extend(noise_item)
                tr_noise_samples[2].extend(np.random.choice(list(ratings), size=sample_num, replace=True))

        # val_neg_samples[0] is the user id val_neg_samples[2] is the id of the item corresponding to the user id
        tr_noise_set = sp.csr_matrix((tr_noise_samples[2], (tr_noise_samples[0], tr_noise_samples[1])), csr_data.shape,
                                     dtype=np.float32)  #  The set of negative samples for the validation set
        sample_csr = sp.csr_matrix((construct_train_data[2], (construct_train_data[0], construct_train_data[1])),
                                   csr_data.shape, dtype=np.float32)
        return tr_noise_set,sample_csr,cold_start_data

    def get_bias_group(self,n_item:int,train_item_count:dict,test_list:list,data_shape):
        sample_num = int(n_item * 0.1)
        head, tail = list(train_item_count.keys())[:sample_num], list(train_item_count.keys())[
                                                                        sample_num:]  # headline product
        bias_group = [[], []]
        for i in test_list:
            if i[1] in head:
                bias_group[0].append(i)
            if i[1] in tail:
                bias_group[1].append(i)

        for i, data in enumerate(bias_group):
            logging.info(f"bias interaction num:{len(data)}")
            bias_group[i] = merge([data],data_shape)
        return bias_group

def merge(datalist,data_shape):
    data = datalist[0]
    for d in datalist:
        if d == data: continue
        data.extend(d)

    data = list(zip(*data))  # zip(a,b) packs a list of tuples, zip(*zipped) is the opposite of zip, *zipped can be interpreted as unpacking and returns a two-dimensional matrix form
    return sp.csr_matrix((data[2], (data[0], data[1])), shape=data_shape)

def csr_col_norm(train_csr: csr_matrix,pre:csr_matrix):
    csrT = train_csr.T.tocsr()
    ind, ptr = csrT.indices, csrT.indptr
    count=dict()
    for i in range(len(ptr) - 1): #Calculate the prevalence in the training set
        l = ptr[i]
        r = ptr[i + 1]

        s=len(csrT.data[l:r])
        count[i] = s
    csr=pre.T.tocsr()
    ind, ptr = csr.indices, csr.indptr
    for i in range(len(ptr) - 1):  # Calculate the prevalence in the training set
        l = ptr[i]
        r = ptr[i + 1]
        if l==r:continue
        csr.data[l:r]=count[i]

    return csr.T.tocsr()


def dict_to_yamlstr(d:dict)->str:
    with io.StringIO() as mio:
        json.dump(d, mio)
        mio.seek(0)
        if hasattr(yaml, 'full_load'):
            y = yaml.full_load(mio)
        else:
            y = yaml.load(mio)
        return yaml.dump(y)  #  Output model name + parameters

def get_parameters_js(js) -> dict:
    ans = None
    if isinstance(js, (dict)):
        ans = dict([(k,get_parameters_js(v)) for (k,v) in js.items() if not isinstance(v, types.BuiltinMethodType)])
    elif isinstance(js, (float, int, str)):
        # js[k] is a normal parameter, integer, float, string
        ans = js
    elif isinstance(js, (list, set, tuple)):
        # js[k] is an array
        ans = [get_parameters_js(x) for x in js]
    elif js is None:
        ans = None
    else:
        # js[k] is an object
        ans = {get_full_class_name(js): get_parameters_js(js.__dict__)}
    return ans

def get_full_class_name(c)->str:
    s = str(type(c))
    return s[8:-2]  # Remove ‘<classes “” and “”>’.
