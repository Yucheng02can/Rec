import numpy as np
import scipy.sparse as sp
import os
from torch import nn,Tensor
import torch.utils.data as Data
import torch
import Debias.utils.earlystopping as earlystopping
import numpy
from tqdm import tqdm
import torch.nn.functional as F
import sys


from Debias.data_loader import RSImplicitData
from Debias.loss import BPRLoss

torch.manual_seed(1)
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'
use_gpu = torch.cuda.is_available()

EarlyStopping = earlystopping.EarlyStopping
class MatrixFactorization(nn.Module):     #

    def get_emb(self, ni, nf):
        assert isinstance(ni, int)
        e = nn.Embedding(ni, nf)
        e.weight.data.uniform_(-0.01, 0.01)
        return e

    def __init__(self, n_users, n_items, n_factors):
        super().__init__()
        self.u, self.m = self.get_emb(n_users, n_factors), self.get_emb(n_items, n_factors)
        self.bu, self.bi = self.get_emb(n_users, 1), self.get_emb(n_items, 1)
        self.mu = nn.parameter.Parameter(torch.zeros((1,)))
        # self.mu = self.get_emb(1, 1)

    def forward(self, users,items):
        tuser = users.long()
        titems= items.long()
        u, m = self.u(tuser), self.m(titems)
        pred = (u * m).sum(1, keepdim=True) + self.bu(tuser) + self.bi(titems) + self.mu.repeat((len(users),1))
        return pred.squeeze()


class MF:
    def __init__(self):
        self.d = 64
        self.lambd = 0.001
        self.lr=0.001
        self.batch_size = 2048
        self.n_itr = 1000

    def train(self, ds,valid_ds = None,valid_funcs=None,cb_progress=lambda x:None,patience=7):
        assert sp.isspmatrix_csr(ds)
        n,m = ds.shape
        np.random.seed(1)


        early_stopping = EarlyStopping(patience=patience, verbose=True)
        self.model = MatrixFactorization(n, m, self.d)
        loss_fun = BPRLoss
        ds = ds.tocoo()
        opt = torch.optim.Adam(self.model.parameters(), self.lr,weight_decay=self.lambd)
        loader = Data.DataLoader(
            dataset=RSImplicitData(ds),
            batch_size=self.batch_size,
            shuffle=True,
        )
        if use_gpu:
            self.model = self.model.cuda()
            # loss_fun = loss_fun.cuda()
        for t in range(self.n_itr):
            #cb_progress(float(t) / self.n_itr)
            for step, (batch_u, batch_i, batch_neg_i) in enumerate(tqdm(loader, desc="Processing", leave=True)):
                # pred = self.model(batch_u.numpy(),batch_i.numpy())
                if use_gpu:
                    batch_u=batch_u.cuda()
                    batch_i=batch_i.cuda()
                    batch_neg_i= batch_neg_i.cuda()
                pos_pred = self.model(batch_u, batch_i)
                neg_pred= self.model(batch_u, batch_neg_i)

                loss = loss_fun(pos_pred, neg_pred)

                loss.backward()
                opt.step()
                opt.zero_grad()
                if use_gpu:
                    loss = loss.cpu()
                cur_loss = float(loss.detach().numpy())
                # if step % 5 == 0:
                #     print("PID:%d\t n_itr:%s\t step:%d \t loss=%.2f  \t"%(os.getpid() ,t,step,cur_loss),file=sys.stderr)

            if t%1 == 0:
                if valid_funcs==None or valid_ds == None:
                    print("PID:%d\t loss=%.2f"%(os.getpid(),cur_loss),file=sys.stderr)
                else:
                    pred = self.predict(valid_ds)
                    scores = [valid_funcs(pred,valid_ds)]
                    fmt_scores = '\t'.join(["{0:0.4f}".format(s) for s in scores])
                    print("PID:%d\t t=%d\t loss=%.2f  \tNDCG@5:%s" % (os.getpid(),t, cur_loss, fmt_scores),file=sys.stderr)

                    early_stopping(scores[0], self.model)
                    if early_stopping.early_stop:
                        print("Early stopping",file=sys.stderr)
                        break
        if  valid_funcs!=None and valid_ds!=None :
            self.model.load_state_dict(early_stopping.get_best())

        # report new status
        cb_progress(1)

    # def predict(self,ds,cb_progress=lambda x:None):
    #     assert sp.isspmatrix_csr(ds)
    #     cb_progress(0)
    #     ds = ds.tocoo()
    #     # cats = np.array(list(zip(*(ds.row, ds.col))), dtype=np.int64)
    #     # cats = torch.from_numpy(cats)
    #     # row,col = torch.from_numpy(ds.row).long(),torch.from_numpy(ds.col).long()
    #     user_num, item_num = ds.shape
    #     uids = torch.Tensor(range(user_num))
    #     iids=range(item_num)
    #     if use_gpu:
    #         uids = uids.cuda()
    #     pred=self.model.full_sort_predict(uids).cpu() #全排列
    #     row=[] #创建user index
    #     for i in range(user_num):
    #         row.extend([iids]*user_num)
    #     col=[]
    #     for i in range(ds.shape[0]):
    #         col.extend(list(range(user_num))*item_num)
    #     data = pred.detach().numpy()
    #     return sp.csr_matrix((data,(row,col)),ds.shape)

    def predict(self,ds,cb_progress=lambda x:None):
        assert sp.isspmatrix_csr(ds)
        cb_progress(0)
        ds = ds.tocoo()
        # cats = np.array(list(zip(*(ds.row, ds.col))), dtype=np.int64)
        # cats = torch.from_numpy(cats)
        # row,col = torch.from_numpy(ds.row).long(),torch.from_numpy(ds.col).long()
        uids = torch.from_numpy(ds.row)
        iids = torch.from_numpy(ds.col)
        if use_gpu:
            uids = uids.cuda()
            iids = iids.cuda()

        #pred=self.model.full_sort_predict(uids)

        pred = self.model(uids,iids)
        cb_progress(1.0) # report progress
        if use_gpu:
            pred = pred.cpu()
        data = pred.detach().numpy()
        return sp.csr_matrix((data,(ds.row,ds.col)),ds.shape)




def GENERATE_MATRIX(observed_ratings, inverse_propensities, normalization, verbose=False):

    inversePropensities = SET_PROPENSITIES(observed_ratings, inverse_propensities, False)

    numUsers, numItems = numpy.shape(observed_ratings)
    scale = numUsers * numItems
    numObservations = numpy.ma.count(observed_ratings)

    # N_u = sum_i(t_ij)
    perUserNormalizer = numpy.ma.sum(inversePropensities, axis=1, dtype=numpy.longdouble)
    perUserNormalizer = numpy.ma.masked_less_equal(perUserNormalizer, 0.0, copy=False)

    # N_i = sum_j(t_ij)
    perItemNormalizer = numpy.ma.sum(inversePropensities, axis=0, dtype=numpy.longdouble)
    perItemNormalizer = numpy.ma.masked_less_equal(perItemNormalizer, 0.0, copy=False)

    # N_s = sum(t_ij)
    globalNormalizer = numpy.ma.sum(inversePropensities, dtype=numpy.longdouble)

    normalizedPropensities = None
    # if normalization == 'Vanilla':
    #     # W_ij = t_ij
    #     normalizedPropensities = inversePropensities
    # elif normalization == 'SelfNormalized':
    #     # W_ij = m*n*t_ij/N_s
    #     normalizedPropensities = scale * numpy.ma.divide(inversePropensities, globalNormalizer)
    # elif normalization == 'UserNormalized':
    #     # W_ij = n*t_ij/N_u[j]
    #     normalizedPropensities = numItems * numpy.ma.divide(inversePropensities, perUserNormalizer[:, None])
    # elif normalization == 'ItemNormalized':
    #     # W_ij = m*t_ij/N_u[i]
    #     normalizedPropensities = numUsers * numpy.ma.divide(inversePropensities, perItemNormalizer[None, :])
    # else:
    #     print("MF.GENERATE_MATRIX: [ERR]\t Normalization not supported:", normalization)
    #     sys.exit(0)
    #
    # # W_ij
    # return normalizedPropensities
    if normalization.startswith('Vanilla'):
        normalizedPropensities = inversePropensities
    elif normalization.startswith('SelfNormalized'):
        normalizedPropensities = scale * numpy.ma.divide(inversePropensities, globalNormalizer)
    elif normalization.startswith('UserNormalized'):
        normalizedPropensities = numItems * numpy.ma.divide(inversePropensities, perUserNormalizer[:, None])
    elif normalization.startswith('ItemNormalized'):
        # W_ui = n*(m*n/N_observed)/n_i)
        normalizedPropensities = numUsers * numpy.ma.divide(inversePropensities, perItemNormalizer[None, :])
    else:
        print("MF.GENERATE_MATRIX: [ERR]\t Normalization not supported:", normalization)
        sys.exit(0)

    return normalizedPropensities




def SET_PROPENSITIES(observed_ratings, inverse_propensities, verbose=False):
    numObservations = numpy.ma.count(observed_ratings)
    numUsers, numItems = numpy.shape(observed_ratings)
    scale = numUsers * numItems
    inversePropensities = None
    if inverse_propensities is None:
        # t_ij = m*n/N_observed
        inversePropensities = numpy.ones((numUsers, numItems), dtype=numpy.longdouble) * scale / \
                              numObservations
    else:
        inversePropensities = numpy.array(inverse_propensities, dtype=numpy.longdouble, copy=True)

    inversePropensities = numpy.ma.array(inversePropensities, dtype=numpy.longdouble, copy=False,
                                         mask=numpy.ma.getmask(observed_ratings), fill_value=0, hard_mask=True)

    # t_ij
    return inversePropensities