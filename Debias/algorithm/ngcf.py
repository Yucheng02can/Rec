import torch
from types import SimpleNamespace
from time import time
import numpy as np
import scipy.sparse as sp
import os
import torch.nn.functional as F
from scipy.sparse import csr_matrix
from torch import nn,Tensor
import torch.utils.data as Data
import torch
from tqdm import tqdm
import sys
import numpy
import Debias.utils.earlystopping as earlystopping
from Debias.data_loader import RSImplicitData
from Debias.loss import BPRLoss
import scipy.sparse as sp
import numpy as np
from torch.utils.data import DataLoader
import torch.nn as nn
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
torch.manual_seed(1)
use_gpu = torch.cuda.is_available()

EarlyStopping = earlystopping.EarlyStopping
class SparseDropout(nn.Module):
    """
    This is a Module that execute Dropout on Pytorch sparse tensor.
    """

    def __init__(self, p=0.5):
        super(SparseDropout, self).__init__()
        # p is ratio of dropout
        # convert to keep probability
        self.kprob = 1 - p

    def forward(self, x):
        if not self.training:
            return x

        mask = ((torch.rand(x._values().size()) + self.kprob).floor()).type(torch.bool)
        rc = x._indices()[:, mask]
        val = x._values()[mask] * (1.0 / self.kprob)
        return torch.sparse.FloatTensor(rc, val, x.shape)

class BiGNNLayer(nn.Module):
    r"""Propagate a layer of Bi-interaction GNN

    .. math::
        output = (L+I)EW_1 + LE \otimes EW_2
    """

    def __init__(self, in_dim, out_dim):
        super(BiGNNLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.linear = torch.nn.Linear(in_features=in_dim, out_features=out_dim)
        self.interActTransform = torch.nn.Linear(
            in_features=in_dim, out_features=out_dim
        )

    def forward(self, lap_matrix, eye_matrix, features):
        # for GCF ajdMat is a (N+M) by (N+M) mat
        # lap_matrix L = D^-1(A)D^-1
        x = torch.sparse.mm(lap_matrix, features)

        inter_part1 = self.linear(features + x)
        inter_feature = torch.mul(x, features)
        inter_part2 = self.interActTransform(inter_feature)

        return inter_part1 + inter_part2

class ngcf(nn.Module):
    r"""NGCF is a model that incorporate GNN for recommendation.
    We implement the model following the original author with a pairwise training mode.
    """

    def __init__(self, n_users,n_items,config, dataset):
        super().__init__()

        # load dataset info
        self.interaction_matrix = dataset
        self.n_users=n_users
        self.n_items=n_items
        if torch.cuda.is_available():
            self.device = 'cuda'

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.hidden_size_list = config["hidden_size_list"]
        self.hidden_size_list = [self.embedding_size] + self.hidden_size_list
        self.node_dropout = config["node_dropout"]
        self.message_dropout = config["message_dropout"]
        self.reg_weight = config["reg_weight"]
        self.is_mlp=config["is_mlp"]

        # define layers and loss
        self.sparse_dropout = SparseDropout(self.node_dropout)
        self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embedding = nn.Embedding(self.n_items, self.embedding_size)
        self.GNNlayers = torch.nn.ModuleList()
        for idx, (input_size, output_size) in enumerate(
            zip(self.hidden_size_list[:-1], self.hidden_size_list[1:])
        ):
            self.GNNlayers.append(BiGNNLayer(input_size, output_size))
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None

        # generate intermediate data
        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)
        self.eye_matrix = self.get_eye_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]

        self.F1 = nn.Linear(in_features=sum(self.hidden_size_list * 2), out_features=64)
        self.F2 = nn.Linear(in_features=64, out_features=32)
        self.F3 = nn.Linear(in_features=32, out_features=1)

    def get_norm_adj_mat(self):
        r"""Get the normalized interaction matrix of users and items.

        Construct the square matrix from the training data and normalize it
        using the laplace matrix.

        .. math::
            A_{hat} = D^{-0.5} \times A \times D^{-0.5}

        Returns:
            Sparse tensor of the normalized interaction matrix.
        """
        # build adj matrix
        A = sp.dok_matrix(
            (self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32
        )
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(
            zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz)
        )
        data_dict.update(
            dict(
                zip(
                    zip(inter_M_t.row + self.n_users, inter_M_t.col),
                    [1] * inter_M_t.nnz,
                )
            )
        )
        A._update(data_dict)
        # norm adj matrix
        sumArr = (A > 0).sum(axis=1)
        diag = (
            np.array(sumArr.flatten())[0] + 1e-7
        )  # add epsilon to avoid divide by zero Warning
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_eye_mat(self):
        r"""Construct the identity matrix with the size of  n_items+n_users.

        Returns:
            Sparse tensor of the identity matrix. Shape of (n_items+n_users, n_items+n_users)
        """
        num = self.n_items + self.n_users  # number of column of the square matrix
        i = torch.LongTensor([range(0, num), range(0, num)])
        val = torch.FloatTensor([1] * num)  # identity matrix
        return torch.sparse.FloatTensor(i, val)

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of (n_items+n_users, embedding_dim)
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        A_hat = (
            self.sparse_dropout(self.norm_adj_matrix)
            if self.node_dropout != 0
            else self.norm_adj_matrix
        )
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]
        for gnn in self.GNNlayers:
            all_embeddings = gnn(A_hat, self.eye_matrix, all_embeddings)
            all_embeddings = nn.LeakyReLU(negative_slope=0.2)(all_embeddings)
            all_embeddings = nn.Dropout(self.message_dropout)(all_embeddings)
            all_embeddings = F.normalize(all_embeddings, p=2, dim=1)
            embeddings_list += [
                all_embeddings
            ]  # storage output embedding of each layer
        ngcf_all_embeddings = torch.cat(embeddings_list, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            ngcf_all_embeddings, [self.n_users, self.n_items]
        )

        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self,  uids, iids,ratings):
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        user = uids.long()
        pos_item =  iids.long()


        user_all_embeddings, item_all_embeddings = self.forward()
        u_embeddings = user_all_embeddings[user]
        pos_embeddings = item_all_embeddings[pos_item]

        pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)

        mf_loss = self.mf_loss(pos_scores, ratings)  # calculate BPR Loss

        reg_loss = self.reg_loss(
            u_embeddings, pos_embeddings,
        )  # L2 regularization of embeddings

        return mf_loss + self.reg_weight * reg_loss

    def predict(self, uids, iids):
        user = uids.long()
        item = iids.long()


        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[user]
        i_embeddings = item_all_embeddings[item]

        pos_scores = torch.mul(u_embeddings, i_embeddings).sum(dim=1)

        return pos_scores

class EmbLoss(nn.Module):
    """EmbLoss, regularization on embeddings"""

    def __init__(self, norm=2):
        super(EmbLoss, self).__init__()
        self.norm = norm

    def forward(self, *embeddings, require_pow=False):
        if require_pow:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.pow(
                    input=torch.norm(embedding, p=self.norm), exponent=self.norm
                )
            emb_loss /= embeddings[-1].shape[0]
            emb_loss /= self.norm
            return emb_loss
        else:
            emb_loss = torch.zeros(1).to(embeddings[-1].device)
            for embedding in embeddings:
                emb_loss += torch.norm(embedding, p=self.norm)
            emb_loss /= embeddings[-1].shape[0]
            return emb_loss

class NGCF:
    def __init__(self):
        self.lr = 0.001
        self.batch_size = 2048
        self.embedding_size = 64
        self.nepochs = 10000
        self.lambd = 0.0001
        self.reg_weight = 0
        self.node_dropout = 0
        self.message_dropout = 0
        self.hidden_size_list = [64, 64, 64]
        self.is_mlp = False
        if torch.cuda.is_available():
            self.device = 'cuda'




    def train(self, ds,valid_ds = None,valid_funcs=None,cb_progress=lambda x:None,patience=7):
        assert sp.isspmatrix_csr(ds)

        self.n, self.m = ds.shape
        pos_ds = del_neg(ds)
        early_stopping = EarlyStopping(patience=patience, verbose=True)
        train_coo = pos_ds.tocoo()
        train_coo.data[:] = 1
        config = self.__dict__
        self.model = ngcf(self.n, self.m, config, train_coo)
        self.model.to(self.device)
        loss_fun = BPRLoss
        ds = ds.tocoo()

        opt = torch.optim.Adam(self.model.parameters(), self.lr,weight_decay=0)
        loader = Data.DataLoader(
            dataset=RSImplicitData(ds),
            batch_size=self.batch_size,
            shuffle=True,
        )
        if use_gpu:
            self.model = self.model.cuda()
            # loss_fun = loss_fun.cuda()
        for t in range(self.nepochs):
            for step, (batch_u, batch_i, batch_neg_i) in enumerate(tqdm(loader, desc="Processing", leave=True)):
                if use_gpu:
                    batch_u = batch_u.cuda()
                    batch_i = batch_i.cuda()
                    batch_neg_i = batch_neg_i.cuda()
                pos_pred = self.model.predict(batch_u, batch_i)
                neg_pred = self.model.predict(batch_u, batch_neg_i)

                loss = loss_fun(pos_pred, neg_pred)
                loss.backward()
                opt.step()
                opt.zero_grad()
                if use_gpu:
                    loss = loss.cpu()
                cur_loss = float(loss.detach().numpy())

            if t%1 == 0:
                if valid_funcs==None or valid_ds == None:
                    print("PID:%d\t loss=%.2f"%(os.getpid(),cur_loss),file=sys.stderr)
                else:
                    pred = self.predict(valid_ds)
                    scores = [valid_funcs(pred, valid_ds)]
                    fmt_scores = '\t'.join(["{0:0.4f}".format(s) for s in scores])
                    print("PID:%d\t t=%d\t loss=%.2f  \tNDCG@5:%s" % (os.getpid(), t, cur_loss, fmt_scores),
                          file=sys.stderr)

                    early_stopping(scores[0], self.model)
                    if early_stopping.early_stop:
                        print("Early stopping", file=sys.stderr)
                        break

        if  valid_funcs!=None and valid_ds!=None :
            self.model.load_state_dict(early_stopping.get_best())

        # report new status
        cb_progress(1)



    def predict(self,ds,cb_progress=lambda x:None):
        assert sp.isspmatrix_csr(ds)
        cb_progress(0)
        ds = ds.tocoo()
        uids = torch.from_numpy(ds.row)
        iids = torch.from_numpy(ds.col)
        if use_gpu:
            uids = uids.cuda()
            iids = iids.cuda()

        pred = self.model.predict(uids, iids)

        cb_progress(1.0) # report progress
        if use_gpu:
            pred = pred.cpu()
        data = pred.detach().numpy()
        #data = pred.cpu().detach().numpy()
        return sp.csr_matrix((data,(ds.row,ds.col)),ds.shape)

    def getSparseGraph(self,ds):
        print("generating adjacency matrix")
        s = time()
        adj_mat = sp.dok_matrix((self.n + self.m, self.n + self.m), dtype=np.float32)
        adj_mat = adj_mat.tolil()
        # R = self.UserItemNet.tolil()
        #R = ds.tolil()
        R = ds.tocoo()
        R.data[:] = 1
        adj_mat[:self.n, self.n:] = R
        adj_mat[self.n:, :self.n] = R.T
        adj_mat = adj_mat.todok()
        #adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

        rowsum = np.array(adj_mat.sum(axis=1))
        d_inv = np.power(rowsum, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.
        d_mat = sp.diags(d_inv)

        norm_adj = d_mat.dot(adj_mat)
        norm_adj = norm_adj.dot(d_mat)
        #norm_adj = adj_mat.tocsr()
        norm_adj = norm_adj.tocsr()
        end = time()
        print(f"costing {end - s}s, saved norm_mat...")

        if self.split == True:
            self.Graph = self._split_A_hat(norm_adj)
            print("done split matrix")
        else:
            self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
            self.Graph = self.Graph.coalesce().to(self.device)
            print("don't split the matrix")
        return self.Graph

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

def del_neg(csr):

    coo = csr.tocoo()

    mask = coo.data != -1
    filtered_coo = csr_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=csr.shape)

    filtered_csr = filtered_coo.tocsr()
    return filtered_csr

def xavier_normal_initialization(module):
    r"""using `xavier_normal_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_normal_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_normal_#torch.nn.init.xavier_normal_

    Examples:
        >>> self.apply(xavier_normal_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_normal_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_normal_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)