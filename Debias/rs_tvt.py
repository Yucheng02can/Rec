# -*- coding: utf-8 -*-
# @Time    : 2023/10/13 10:21
# @Author  : colagold
# @FileName: rs_explicit_tvt.py
import copy
import logging
import os
import pickle

import numpy as np
import scipy.sparse as sp

# ATP: 用于推荐的(train,validate,test)实验验证
from scipy.sparse import csr_matrix

np.random.seed(2022)


class RSImplicitTVT():  # Divide the dataset into trainset validset and testset
    def __init__(self):
        self.train_percent = 0.8  # Training set ratio 8:1:1
        self.valid_percent = 0.1  # Validation set ratio
        self.n_neg = 500
        self.use_cold_testset = False
        self.cold_interval = [0, 20]

    def split(self, data) -> tuple:
        """
          :param data:  csr_matrix
                      example：uids iids   data
                          （ 0  ,  0 ）   1
                          （ 2  ,  2 ）   1
                          （ 5  ,  4 ）   1
          :return:（train_set,validate_set,test_set）
                  train_set/validate_set/test_set:sp.csr_matrix
                      example：uids iids   data
                          （ 0  ,  0 ）   1
                          （ 0  ,  1 ）   -1
                          （ 2  ,  2 ）   1
                          （ 3  ,  3 ）   -1
                          （ 5  ,  4 ）   1
                      1: A positive sample representing the user's preference, the observation is obtained
                     -1: represents a negative sample of user dislikes, obtained by randomly sampling from items not clicked by the user
        """
        data.data[:]=1
        data = data.tocoo()
        data01 = copy.deepcopy(data)
        self.data_shape = data.shape
        data.data = data.data.astype(bool).astype(int)
        # data.sum(axis=0)Sum by column, argsort() in reverse order, .A to arrays
        res = (data.sum(axis=0).argsort()).A
        # entries = list(zip(data.row,data.col,data.data))
        entries = list(zip(data01.row, data01.col, data01.data))

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
        train_set = merge([train_list],self.data_shape).tolil()
        validate_set = merge([valid_list],self.data_shape).tolil()
        test_set = merge([test_list],self.data_shape).tolil()
        origion_train_set=train_set.tocoo()
        # origion_test_set=test_set.tocsr()
        # origion_val_set=validate_set.tocsr()


        n_user, n_item = coo.shape
        all_items = set(coo.col)  # Treat all interacted items as a candidate set.

        val_neg_samples = ([], [])  # The list of negative samples of the validation set, which holds the negative sample user id, product id
        te_neg_samples = ([], [])  # The list of negative samples of the test set, which holds the negative sample user id, product id


        for u in range(n_user):  # Iterate through the users, sampling 100 negative samples for each user (not 100 negative samples for each interaction here)
            tr_items = set(train_set.rows[u])  # Get all the goods that user u has interacted with in the training set
            val_items = set(validate_set.rows[u])  # Get all the items that user u has interacted with in the validation set
            te_items = set(test_set.rows[u])  # Get all the items that user u has interacted with in the test set
            if len(tr_items) == 0:  # There are no goods in this user's training set, then no negative samples need to be generated
                continue
            if len(val_items) == 0 and len(te_items) == 0:  # User u has no interaction goods in both the test and validation sets, then there is no need to generate negative samples
                continue  #

            neg_items = all_items - val_items - te_items - tr_items  # Get a collection of products that have not been visited by the user
            sample_num = min(len(neg_items), self.n_neg)  # Set the number of negative samples, if the user interacts with too many items, the set of unvisited items may be less than self.n_neg
            if sample_num == 0: continue  # The user has no uninteracted items and cannot test the user
            neg_users = [u] * sample_num  # User u's id duplicated self.neg times
            if len(val_items) > 0:  # Negative samples are generated if user u has interacting goods in the validation set, otherwise they are not generated
                neg_item = np.random.choice(list(neg_items), sample_num, replace=False)  # Non-repeating negative sampling
                val_neg_samples[0].extend(neg_users)
                val_neg_samples[1].extend(neg_item)

            te_neg_item = None
            if len(te_items) > 0:  # Negative samples are generated if there are positive samples from users in the test set, otherwise they are not generated
                te_neg_item = np.random.choice(list(neg_items), sample_num,
                                               replace=False)  # Negative non-repeat sampling , replace Indicates whether put-back sampling is performed Used to randomly select elements from a specified sequence
                te_neg_samples[0].extend(neg_users)
                te_neg_samples[1].extend(te_neg_item)
        train_set = train_set.tocsr()
        # val_neg_samples[0] is the user id val_neg_samples[2] is the id of the item corresponding to the user id
        all_minus_one = [-1] * len(val_neg_samples[0])
        val_neg_set = sp.csr_matrix((all_minus_one, val_neg_samples), data.shape,
                                    dtype=np.float32)  # The set of negative samples for the validation set
        validate_set = validate_set.tocsr() + val_neg_set  # The sparse matrix of the summed validate_set contains the items that each user interacted with and the items that were not interacted with after 100 randomly sampled items

        all_minus_one = [-1] * len(te_neg_samples[0])
        te_neg_set = sp.csr_matrix((all_minus_one, te_neg_samples), data.shape,
                                   dtype=np.float32)  # The set of negative samples of the test set, value = -1
        test_set = test_set.tocsr() + te_neg_set
        #  Note that the users in the **training set** whose number of user interactions is less than inter_num_max and greater than or equal to inter_num_min are used as the cold-start test users
        # The users are grouped according to the number of interactions they have with the product to obtain a cold-start test set, e.g. [0,8), [8,16), [16,32), [32,64), [64,)
        # Save test sets for cold-start users (no need to re-negative-sample)
        # Product popularity grouping
        popularity_group, no_neg_popularity_group=self.get_item_fre_group(origion_train_set,test_list,te_neg_set)
        # Split into two sections by product popularity
        head_and_tail_group=self.get_item_group(origion_train_set,test_list,te_neg_set)
        print("Popularity processing complete")
        return train_set, validate_set, test_set,popularity_group,head_and_tail_group

    def get_item_fre_group(self,origion_train_set,test_list,te_neg_set):
        item, count = np.unique(origion_train_set.col, return_counts=True)
        train_item_inter_count = dict(zip(item, count))  # User records in the training set without added negative samples
        item_inter_plan = [(0, 15), (15, 20), (20, 25), (25, 30), (30, 35)]
        popularity_group = [[], [], [], [], []]
        for i, inter_count in train_item_inter_count.items():
            for index, t in enumerate(item_inter_plan):
                if inter_count >= t[0] and inter_count < t[1]:
                    popularity_group[index].extend([e for e in test_list if e[1] == i])  # Place the corresponding user interaction records in the test set into the corresponding grouping
                    break
        for i in popularity_group:
            print(len(i))
        no_neg_popularity_group = []
        for i, data in enumerate(popularity_group):
            no_neg_popularity_group.append(merge([data],self.data_shape))
            popularity_group[i] = merge([data],self.data_shape) + te_neg_set
        return popularity_group,no_neg_popularity_group

    def get_item_group(self, origion_train_set, test_list, te_neg_set):
        item, count = np.unique(origion_train_set.col, return_counts=True)
        train_item_inter_count = dict(zip(item, count))  # User records in the training set without added negative samples

        # Sort the item based on the number of interactions and calculate the top 20% thresholds
        sorted_items = sorted(train_item_inter_count.items(), key=lambda x: x[1], reverse=True)
        threshold_index = int(0.2 * len(sorted_items))

        high_popularity_items = {item[0] for item in sorted_items[:threshold_index]}  # Top 20 per cent of items
        low_popularity_items = {item[0] for item in sorted_items[threshold_index:]}  # Remaining 80 per cent of the item

        high_popularity_group = []
        low_popularity_group = []

        # Segment the interaction records in test_list based on whether the item belongs to the top 20% or not
        for i, inter_count in train_item_inter_count.items():
            if i in high_popularity_items:
                high_popularity_group.extend([e for e in test_list if e[1] == i])
            else:
                low_popularity_group.extend([e for e in test_list if e[1] == i])

        # Print the size of each group
        print(f"High popularity group size: {len(high_popularity_group)}")
        print(f"Low popularity group size: {len(low_popularity_group)}")

        # Combining positive and negative samples
        head_and_tail_group=dict()
        head_and_tail_group["Head"] = merge([high_popularity_group], self.data_shape) + te_neg_set
        head_and_tail_group["Tail"] = merge([low_popularity_group], self.data_shape) + te_neg_set

        return head_and_tail_group

    def test(self, algorithm, ds, metrics,another=None,logger=None, cb_progress=lambda x: None):
        '''
            algorithm: derived class of util.model.
            ds: is a two-dimensional sp.csr_matrix, [n_users,n_items], rows represent users, columns represent items
        '''
        # Divide training, validation, testing, and cold-start datasets
        process_path = f"process_data/"
        if not os.path.isdir(process_path):
            os.makedirs(process_path)
        process_data_path = os.path.join(process_path, another + ".pkl")
        if not os.path.isfile(process_data_path):  # Save the data division, cold start division and bias division after adding noise to save time.
            # Validation, testing, mixing of interacting and non-interacting samples in cold-start datasets
            datalist = self.split(ds)
            train_data, valid_data, test_data, popularity_group, head_and_tail_group = \
                datalist[0], datalist[1], datalist[2], datalist[3], datalist[4]
            process_data = {
                "train_data": train_data,
                "test_data": test_data,
                "valid_data": valid_data,
                "head_and_tail_group": head_and_tail_group,
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
                head_and_tail_group = data["head_and_tail_group"]
                popularity_group = data["popularity_group"]
                print("Processing data loaded successfully")



        # [0,16,32] => [0,16],[16,32],[32,]; [0,20]
        # The training process of the algorithm is run, passing the training set, but also the test set, and the first test metric.
        valid_funs = [m for m in metrics]

        algorithm.train(train_data, valid_data, valid_funs[0])

        pred = algorithm.predict(test_data)
        results = [m( test_data,pred) for m in metrics]
        headers = [str(m) for m in metrics]
        ##Random Recommendations####################################################
        rating = np.unique(test_data.data[:])
        random_pre = copy.deepcopy(test_data)
        random_pre.data[:] = np.random.choice(rating, size=test_data.nnz, replace=True)
        random_results = [m(test_data, random_pre) for m in metrics]
        print("RANDOM_RESULTS=" + str(dict(zip(headers, random_results))))
        logger.info("RANDOM_RESULTS=" + str(dict(zip(headers, random_results))))

        ##Recommended by popularity
        popularity_pre = copy.deepcopy(test_data)
        popularity_pre = csr_col_norm(train_data, popularity_pre)
        popularity_results = [m(test_data, popularity_pre) for m in metrics]
        print("POPULARITY_RESULTS=" + str(dict(zip(headers, popularity_results))))
        logger.info("POPULARITY_RESULTS=" + str(dict(zip(headers, popularity_results))))

        bias_result = {}
        item_inter_plan=['0-15', '15-20', '20-25', '25-30', '30-35']
        for index, bias_group in enumerate(popularity_group):
            pred = algorithm.predict(bias_group)
            res=[]
            for i,m in enumerate(metrics):
                #m.pred_ground_truth=no_neg_popularity_group[index]
                res.append({str(m):m(bias_group, pred)})
            bias_result[item_inter_plan[index]]=res
        # groundTruth , _ = test_data
        logger.info("GROUP_RESULTS=" + str(bias_result))
        print("GROUP_RESULTS=" + str(bias_result))

        bias_result = {}
        for key, bias_group in head_and_tail_group.items():
            pred = algorithm.predict(bias_group)
            res = []
            for i, m in enumerate(metrics):
                # m.pred_ground_truth=no_neg_popularity_group[index]
                res.append({str(m):m(bias_group, pred)})
            bias_result[key] = res
        # groundTruth , _ = test_data
        logger.info("HEAD_RESULTS=" + str(bias_result))
        print("HEAD_RESULTS=" + str(bias_result))


        headers = [str(m) for m in metrics]
        logger.info("RESULTS=" + str(dict(zip(headers, results))))
        logging.info("*************************************************")
        return dict(zip(headers, results))




def merge(datalist,shape):
    data = datalist[0]
    for d in datalist:
        if d == data: continue
        data.extend(d)

    data = list(zip(*data))  # zip(a,b) packs a list of tuples, zip(*zipped) is the opposite of zip, *zipped can be interpreted as unpacking and returns a two-dimensional matrix form
    return sp.csr_matrix((data[2], (data[0], data[1])), shape=shape)


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