
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
def xavier_uniform_initialization(module):
    r"""using `xavier_uniform_`_ in PyTorch to initialize the parameters in
    nn.Embedding and nn.Linear layers. For bias in nn.Linear layers,
    using constant 0 to initialize.

    .. _`xavier_uniform_`:
        https://pytorch.org/docs/stable/nn.init.html?highlight=xavier_uniform_#torch.nn.init.xavier_uniform_

    Examples:
        >>> self.apply(xavier_uniform_initialization)
    """
    if isinstance(module, nn.Embedding):
        xavier_uniform_(module.weight.data)
    elif isinstance(module, nn.Linear):
        xavier_uniform_(module.weight.data)
        if module.bias is not None:
            constant_(module.bias.data, 0)

class lightgcn(nn.Module):
    def __init__(self, n_users,n_items,config, dataset):
        super().__init__()

        # load dataset info
        self.interaction_matrix = dataset

        # load parameters info
        self.latent_dim = config[
            "embedding_size"
        ]  # int type:the embedding size of lightGCN
        self.n_layers = config["n_layers"]  # int type:the layer num of lightGCN
        self.reg_weight = config[
            "reg_weight"
        ]  # float32 type: the weight decay for l2 normalization
        self.require_pow = config["require_pow"]

        # define layers and loss
        self.n_users=n_users
        self.n_items=n_items
        self.user_embedding = torch.nn.Embedding(
            num_embeddings=self.n_users, embedding_dim=self.latent_dim
        )
        self.item_embedding = torch.nn.Embedding(
            num_embeddings=self.n_items, embedding_dim=self.latent_dim
        )
        # self.mf_loss = Weighted_MSELoss()
        self.reg_loss = EmbLoss()

        # storage variables for full sort evaluation acceleration
        self.restore_user_e = None
        self.restore_item_e = None
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'

        # generate intermediate data

        self.norm_adj_matrix = self.get_norm_adj_mat().to(self.device)

        # parameters initialization
        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_item_e"]
        self.transForm1 = nn.Linear(in_features=self.latent_dim * 2, out_features=64)
        self.transForm2 = nn.Linear(in_features=64, out_features=32)
        self.transForm3 = nn.Linear(in_features=32, out_features=1)

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
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        # covert norm_adj matrix to tensor
        L = sp.coo_matrix(L)
        row = L.row
        col = L.col
        i = torch.LongTensor(np.array([row, col]))
        #A=sp.coo_matrix(A)
        data = torch.FloatTensor(L.data)
        SparseL = torch.sparse.FloatTensor(i, data, torch.Size(L.shape))
        return SparseL

    def get_ego_embeddings(self):
        r"""Get the embedding of users and items and combine to an embedding matrix.

        Returns:
            Tensor of the embedding matrix. Shape of [n_items+n_users, embedding_dim]
        """
        user_embeddings = self.user_embedding.weight
        item_embeddings = self.item_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        return ego_embeddings

    def forward(self):
        all_embeddings = self.get_ego_embeddings()
        embeddings_list = [all_embeddings]

        for layer_idx in range(self.n_layers):
            all_embeddings = torch.sparse.mm(self.norm_adj_matrix, all_embeddings)
            embeddings_list.append(all_embeddings)
        lightgcn_all_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_all_embeddings = torch.mean(lightgcn_all_embeddings, dim=1)

        user_all_embeddings, item_all_embeddings = torch.split(
            lightgcn_all_embeddings, [self.n_users, self.n_items]
        )
        return user_all_embeddings, item_all_embeddings

    def calculate_loss(self, users,items,ratings): #check out
        # clear the storage variable when training
        if self.restore_user_e is not None or self.restore_item_e is not None:
            self.restore_user_e, self.restore_item_e = None, None

        # u_embeddings = user_all_embeddings[users]
        # pos_embeddings = item_all_embeddings[items]


        # calculate BPR Loss
        pred = self.predict(users,items)


        # calculate BPR Loss
        u_ego_embeddings = self.user_embedding(users)
        pos_ego_embeddings = self.item_embedding(items)

        mf_loss=self.mf_loss(pred, ratings,torch.ones(users.shape).to(self.device))

        reg_loss = self.reg_loss(
            u_ego_embeddings,
            pos_ego_embeddings,
            require_pow=self.require_pow,
        )

        loss = mf_loss + self.reg_weight * reg_loss

        return loss

    def predict(self, users,items):

        user_all_embeddings, item_all_embeddings = self.forward()

        u_embeddings = user_all_embeddings[users.long()]
        i_embeddings = item_all_embeddings[items.long()]
        embd = torch.cat([u_embeddings, i_embeddings], dim=1)
        embd = nn.ReLU()(self.transForm1(embd))
        embd = self.transForm2(embd)
        embd = self.transForm3(embd)
        prediction = embd.flatten()

        return prediction

    def full_sort_predict(self, interaction):
        user = interaction[self.USER_ID]
        if self.restore_user_e is None or self.restore_item_e is None:
            self.restore_user_e, self.restore_item_e = self.forward()
        # get user embedding from storage variable
        u_embeddings = self.restore_user_e[user]

        # dot with all item embedding to accelerate
        scores = torch.matmul(u_embeddings, self.restore_item_e.transpose(0, 1))

        return scores.view(-1)

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



class LightGCN:
    def __init__(self):
        self.lr = 0.001
        self.embedding_size = 64
        self.reg_weight = 0.0001
        self.require_pow = False
        self.batch_size = 2048
        self.lambda_ = 0.1
        self.nepochs = 10000
        self.n_layers = 1
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'




    def train(self, ds,valid_ds = None,valid_funcs=None,cb_progress=lambda x:None,patience=7):
        assert sp.isspmatrix_csr(ds)

        pos_ds=del_neg(ds)
        self.user_num, self.item_num = ds.shape

        early_stopping = EarlyStopping(patience=patience, verbose=True)
        #self.graph = self.getSparseGraph(pos_ds)
        config = self.__dict__
        self.model = lightgcn(self.user_num, self.item_num, config, pos_ds.tocoo())
        # self.model = lightgcn(config)
        self.model.to(self.device)
        loss_fun = BPRLoss#weighted sum
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

                loss = loss_fun(pos_pred, neg_pred)  # training loss
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
    #Convert to COO format
    coo = csr.tocoo()
    # Filter out elements with a value of -1
    mask = coo.data != -1
    filtered_coo = csr_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=csr.shape)
    # Convert back to CSR format
    filtered_csr = filtered_coo.tocsr()
    return filtered_csr