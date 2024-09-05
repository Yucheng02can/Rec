
from types import SimpleNamespace
from time import time
import numpy as np
import scipy.sparse as sp
import os,io
from torch.nn.init import xavier_normal_, xavier_uniform_, constant_
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

torch.manual_seed(1)
use_gpu = torch.cuda.is_available()

EarlyStopping = earlystopping.EarlyStopping

class lightgcn(nn.Module):
    def __init__(self,config):
        super().__init__()
        self.config=config
        self.n_layers = config.n_layers
        self.keep_prob = config.keep_prob
        self.A_split = config.A_split
        self.num_users=config.n
        self.num_items=config.m
        self.dropout=config.dropout

        self.embedding_user = torch.nn.Embedding(
            num_embeddings=config.n, embedding_dim=config.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=config.m, embedding_dim=config.latent_dim)
        # self.embedding_rating = torch.nn.Embedding(
        #     num_embeddings=num_users*num_items, embedding_dim=latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.01)
        nn.init.normal_(self.embedding_item.weight, std=0.01)
        #nn.init.normal_(self.embedding_rating.weight, std=0.1)
            #world.cprint('use NORMAL distribution initilizer')

        self.f = nn.Sigmoid()
        self.Graph = config.graph
        # print(f"lgn is already to go(dropout:{self.config['dropout']})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config.dropout:
            if self.training:
                #print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb,
         userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(), neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]
        # inner_pro = torch.mul(users_emb, items_emb)
        # gamma = torch.sum(inner_pro, dim=1)
        prediction = (users_emb * items_emb).sum(1)#embd.flatten()

        return prediction



class LightGCN:
    def __init__(self):
        self.n_layers = 3
        self.keep_prob = 0.6
        self.A_split = False
        self.latent_dim = 64
        self.batch_size = 2048
        self.lr = 0.0001
        self.lambd = 0.01
        self.dropout = 0
        self.nepochs = 10000
        self.split = False
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'




    def train(self, ds,valid_ds = None,valid_funcs=None,cb_progress=lambda x:None,patience=7):
        assert sp.isspmatrix_csr(ds)

        pos_ds=del_neg(ds)
        self.n, self.m = ds.shape

        early_stopping = EarlyStopping(patience=patience, verbose=True)
        self.graph = self.getSparseGraph(ds)
        config = self.__dict__
        config = SimpleNamespace(**config)

        self.model = lightgcn(config)
        # self.model = lightgcn(config)
        self.model.to(self.device)
        loss_fun = BPRLoss
        ds = ds.tocoo()

        opt = torch.optim.Adam(self.model.parameters(), self.lr,weight_decay=0.0001)
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
                pos_pred = self.model(batch_u, batch_i)
                neg_pred = self.model(batch_u, batch_neg_i)

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

        pred = self.model(uids, iids)

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