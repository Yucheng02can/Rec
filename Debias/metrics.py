import logging
import math

import numpy as np
import scipy.sparse as sp
import scipy.optimize as opt
from sklearn.metrics import normalized_mutual_info_score, \
    adjusted_rand_score, rand_score
import heapq


rating_gate=0

class MetricAbstract:
    def __init__(self):
        self.bigger= True # The larger the indicator, the better, if False, it means the smaller the better

    def __str__(self):
        return self.__class__.__name__


    def __call__(self,groundtruth,pred ) ->float:
        raise Exception("Not callable for an abstract function")

class NDCG5():
    def __init__(self):
        self.topk = 5

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil() # test set
        pred = pred.tolil() #full forecast
        users = list(set(ground_truth.nonzero()[0]))  # Users with no recorded item interactions do not participate in the calculation
        ndcg = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate] # User-interactive product listings
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # Sort all items by their predicted ratings, taking only the topk value and its corresponding index (item_id).
            _, topk_items = list(zip(*rank_pair))
            idcg_len = min(len(topk_items), len(pos_items))
            idcg = sum([1.0 / math.log2(idx) for idx in range(2, idcg_len + 2)])
            dcg = sum(
                [1.0 / math.log2(idx2 + 2) if item in pos_items else 0.0 for idx2, item in enumerate(topk_items)])
            ndcg.append(dcg / idcg)
        return sum(ndcg) / len(ndcg)


class NDCG10():
    def __init__(self):
        self.topk = 10

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # Users with no recorded item interactions do not participate in the calculation
        ndcg = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            #print(f"positive item lens {len(pos_items)} all item lens {len(gt.data[uid])}")
            # logging.info(f"positive item lens {len(pos_items)} all item lens {len(gt.data[uid])}")
            # print(gt.data[uid])
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # Sort all items by their predicted ratings, taking only the topk value and its corresponding index (item_id).
            _, topk_items = list(zip(*rank_pair))
            idcg_len = min(len(topk_items), len(pos_items))
            idcg = sum([1.0 / math.log2(idx) for idx in range(2, idcg_len + 2)])
            dcg = sum(
                [1.0 / math.log2(idx2 + 2) if item in pos_items else 0.0 for idx2, item in enumerate(topk_items)])
            ndcg.append(dcg / idcg)
        return sum(ndcg) / len(ndcg)


class NDCG20():
    def __init__(self):
        self.topk = 20

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # Users with no recorded item interactions do not participate in the calculation
        ndcg = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # Sort all items by their predicted ratings, taking only the topk value and its corresponding index (item_id).
            _, topk_items = list(zip(*rank_pair))
            idcg_len = min(len(topk_items), len(pos_items))
            idcg = sum([1.0 / math.log2(idx) for idx in range(2, idcg_len + 2)])
            dcg = sum(
                [1.0 / math.log2(idx2 + 2) if item in pos_items else 0.0 for idx2, item in enumerate(topk_items)])
            ndcg.append(dcg / idcg)
        return sum(ndcg) / len(ndcg)


class RECALL5():
    """ Recall, which indicates how many of all positive samples are successfully identified by the model. It is calculated in the same way as Hit@K
    Probability of being recommended in a positive sample
      The larger the indicator, the better.
     eg:
        pred/ground_truth:
            [1,2,3,4,7] / [2,7,9];  命中item 2, 7; RECALL_1 = 2/3
            [1,3,4,8,9] / [4,7];    命中item 4;   RECALL_2 = 1/2
        RECALL@5 = 1/2 * (2/3 + 1/2)
    """

    def __init__(self):
        self.topk = 5

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # Users with no recorded item interactions do not participate in the calculation
        recall = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # Sort all items by their predicted ratings, taking only the topk value and its corresponding index (item_id).
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            recall.append(sum(hits) / len(pos_items))
        return sum(recall) / len(recall)


class RECALL10():
    def __init__(self):
        self.topk = 10

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # Users with no recorded item interactions do not participate in the calculation
        recall = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # Sort all items by their predicted ratings, taking only the topk value and its corresponding index (item_id).
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            recall.append(sum(hits) / len(pos_items))
        return sum(recall) / len(recall)


class RECALL20():
    def __init__(self):
        self.topk = 20

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # Users with no recorded item interactions do not participate in the calculation
        recall = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # Sort all items by their predicted ratings, taking only the topk value and its corresponding index (item_id).
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            recall.append(sum(hits) / len(pos_items))
        return sum(recall) / len(recall)

class MRR5():
    """Mean Reciprocal Rank (MRR@K), which reflects whether the recommended item is in a more visible position to the user, emphasising ‘sequentiality’.
         Countdown of the first correct answer's rank in the topk recommendation list
        Calculation formula: $$$$
        The larger the indicator, the better.
    eg:
        pred/ground_truth:
            [1,2,3,4,7] / [2,7];    First hit item 2;   mrr_1 = 1/2 = 0.5
            [1,3,4,7,9] / [4,7,10]; First hit item 4;   mrr_2 = 1/3 = 0.3333
        MRR@5 = 1/2*(0.5+0.3333)
    """

    def __init__(self):
        self.topk = 5

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # Users with no recorded item interactions do not participate in the calculation
        mrr = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue

            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # Sort all items by their predicted ratings, taking only the topk value and its corresponding index (item_id).
            _, topk_items = list(zip(*rank_pair))
            for idx, item in enumerate(topk_items):
                if item in pos_items:
                    mrr.append(1 / (idx + 1))
                    break
                # else:
                #     mrr.append(0)
            # mrr.append(sum([1/(idx+1) if item in pos_items else 0 for idx, item in enumerate(topk_items)])/len(topk_items)) # Similar to DCG, the denominator of DCG is math.log2(idx+2)
        return sum(mrr) / len(users)


class MRR10():
    def __init__(self):
        self.topk = 10

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  #  Users with no recorded item interactions do not participate in the calculation
        mrr = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue

            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # Sort all items by their predicted ratings, taking only the topk value and its corresponding index (item_id).
            _, topk_items = list(zip(*rank_pair))
            for idx, item in enumerate(topk_items):
                if item in pos_items:
                    mrr.append(1 / (idx + 1))
                    break
                # else:
                #     mrr.append(0)
            # mrr.append(sum([1/(idx+1) if item in pos_items else 0 for idx, item in enumerate(topk_items)])/len(topk_items)) # Similar to DCG, the denominator of DCG is math.log2(idx+2)
        return sum(mrr) / len(users)


class MRR20():
    def __init__(self):
        self.topk = 20

    def __str__(self):
        return self.__class__.__name__

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # Users with no recorded item interactions do not participate in the calculation
        mrr = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue

            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # Sort all items by their predicted ratings, taking only the topk value and its corresponding index (item_id).
            _, topk_items = list(zip(*rank_pair))
            for idx, item in enumerate(topk_items):
                if item in pos_items:
                    mrr.append(1 / (idx + 1))
                    break
                # else:
                #     mrr.append(0)
            # mrr.append(sum([1/(idx+1) if item in pos_items else 0 for idx, item in enumerate(topk_items)])/len(topk_items)) # Similar to DCG, the denominator of DCG is math.log2(idx+2)
        return sum(mrr) / len(users)



class Precision5():
    """Precision, which reflects the precision of the recommendation list in the item, emphasising the ‘accuracy’ of the prediction.
    Probability of a positive sample in the recommended results
    Calculation formula: $1/N $
    The larger the indicator, the better.
    eg:
        pred/ground_truth:
            [1,2,3,4,7] / [2,7,9];  hits item 2, 7; precision_1 = 2/5
            [1,3,4,8,9] / [4,7];    hits item 4;   precision_2 = 1/5
        precision@5 = 1/2 * (2/5 + 1/5)
    """

    def __init__(self):
        self.topk = 5

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # Users with no recorded item interactions do not participate in the calculation
        precision = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue

            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # Sort all items by their predicted ratings, taking only the topk value and its corresponding index (item_id).
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            precision.append(sum(hits) / len(topk_items))  # Difference with recall/hr
        return sum(precision) / len(precision)


class Precision10():
    def __init__(self):
        self.topk = 10

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))  # Users with no recorded item interactions do not participate in the calculation
        precision = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))  # Sort all items by their predicted ratings, taking only the topk value and its corresponding index (item_id).
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            precision.append(sum(hits) / len(topk_items))
        return sum(precision) / len(precision)


class Precision15():
    def __init__(self):
        self.topk = 15

    def __call__(self, ground_truth: sp.isspmatrix_csr, pred: sp.isspmatrix_csr) -> float:
        gt = ground_truth.tolil()
        pred = pred.tolil()
        users = list(set(ground_truth.nonzero()[0]))
        precision = []
        for uid in users:
            pos_items = [gt.rows[uid][i] for i, x in enumerate(gt.data[uid]) if x > rating_gate]
            if len(pos_items) == 0: continue
            rank_pair = heapq.nlargest(self.topk,
                                       zip(pred.data[uid], pred.rows[uid]))
            _, topk_items = list(zip(*rank_pair))
            hits = [1 if item in pos_items else 0 for item in topk_items]
            precision.append(sum(hits) / len(topk_items))
        return sum(precision) / len(precision)


