
from types import SimpleNamespace
from time import time
import numpy as np
import scipy.sparse as sp
import os,io

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

r"""
NeuMF
################################################
Reference:
    Xiangnan He et al. "Neural Collaborative Filtering." in WWW 2017.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.nn.init import normal_
import scipy.sparse as sp
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


def activation_layer(activation_name="relu", emb_dim=None):
    """Construct activation layers

    Args:
        activation_name: str, name of activation function
        emb_dim: int, used for Dice activation

    Return:
        activation: activation layer
    """
    if activation_name is None:
        activation = None
    elif isinstance(activation_name, str):
        if activation_name.lower() == "sigmoid":
            activation = nn.Sigmoid()
        elif activation_name.lower() == "tanh":
            activation = nn.Tanh()
        elif activation_name.lower() == "relu":
            activation = nn.ReLU()
        elif activation_name.lower() == "leakyrelu":
            activation = nn.LeakyReLU()
        elif activation_name.lower() == "dice":
            activation = Dice(emb_dim)
        elif activation_name.lower() == "none":
            activation = None
    elif issubclass(activation_name, nn.Module):
        activation = activation_name()
    else:
        raise NotImplementedError(
            "activation function {} is not implemented".format(activation_name)
        )

    return activation

class Dice(nn.Module):
    r"""Dice activation function

    .. math::
        f(s)=p(s) \cdot s+(1-p(s)) \cdot \alpha s

    .. math::
        p(s)=\frac{1} {1 + e^{-\frac{s-E[s]} {\sqrt {Var[s] + \epsilon}}}}
    """

    def __init__(self, emb_size):
        super(Dice, self).__init__()

        self.sigmoid = nn.Sigmoid()
        self.alpha = torch.zeros((emb_size,))

    def forward(self, score):
        self.alpha = self.alpha.to(score.device)
        score_p = self.sigmoid(score)

        return self.alpha * (1 - score_p) * score + score_p * score

class MLPLayers(nn.Module):
    r"""MLPLayers

    Args:
        - layers(list): a list contains the size of each layer in mlp layers
        - dropout(float): probability of an element to be zeroed. Default: 0
        - activation(str): activation function after each layer in mlp layers. Default: 'relu'.
                           candidates: 'sigmoid', 'tanh', 'relu', 'leekyrelu', 'none'

    Shape:

        - Input: (:math:`N`, \*, :math:`H_{in}`) where \* means any number of additional dimensions
          :math:`H_{in}` must equal to the first value in `layers`
        - Output: (:math:`N`, \*, :math:`H_{out}`) where :math:`H_{out}` equals to the last value in `layers`

    Examples::

        >>> m = MLPLayers([64, 32, 16], 0.2, 'relu')
        >>> input = torch.randn(128, 64)
        >>> output = m(input)
        >>> print(output.size())
        >>> torch.Size([128, 16])
    """

    def __init__(
        self, layers, dropout=0.0, activation="relu", bn=False, init_method=None
    ):
        super(MLPLayers, self).__init__()
        self.layers = layers
        self.dropout = dropout
        self.activation = activation
        self.use_bn = bn
        self.init_method = init_method

        mlp_modules = []
        for idx, (input_size, output_size) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            mlp_modules.append(nn.Dropout(p=self.dropout))
            mlp_modules.append(nn.Linear(input_size, output_size))
            if self.use_bn:
                mlp_modules.append(nn.BatchNorm1d(num_features=output_size))
            activation_func = activation_layer(self.activation, output_size)
            if activation_func is not None:
                mlp_modules.append(activation_func)

        self.mlp_layers = nn.Sequential(*mlp_modules)
        if self.init_method is not None:
            self.apply(self.init_weights)

    def init_weights(self, module):
        # We just initialize the module with normal distribution as the paper said
        if isinstance(module, nn.Linear):
            if self.init_method == "norm":
                normal_(module.weight.data, 0, 0.01)
            if module.bias is not None:
                module.bias.data.fill_(0.0)

    def forward(self, input_feature):
        return self.mlp_layers(input_feature)





class neumf(nn.Module):
    r"""NeuMF is an neural network enhanced matrix factorization model.
    It replace the dot product to mlp for a more precise user-item interaction.

    Note:

        Our implementation only contains a rough pretraining function.

    """

    def __init__(self, n,m,config):
        super(neumf, self).__init__()

        # load dataset inf

        # load parameters info
        self.mf_embedding_size = config["mf_embedding_size"]
        self.mlp_embedding_size = config["mlp_embedding_size"]
        self.mlp_hidden_size = config["mlp_hidden_size"]
        self.dropout_prob = config["dropout_prob"]
        self.mf_train = config["mf_train"]
        self.mlp_train = config["mlp_train"]
        self.use_pretrain = config["use_pretrain"]
        self.mf_pretrain_path = config["mf_pretrain_path"]
        self.mlp_pretrain_path = config["mlp_pretrain_path"]
        self.n_users=n
        self.n_items=m

        # define layers and loss
        self.user_mf_embedding = nn.Embedding(self.n_users, self.mf_embedding_size)
        self.item_mf_embedding = nn.Embedding(self.n_items, self.mf_embedding_size)
        self.user_mlp_embedding = nn.Embedding(self.n_users, self.mlp_embedding_size)
        self.item_mlp_embedding = nn.Embedding(self.n_items, self.mlp_embedding_size)
        self.mlp_layers = MLPLayers(
            [2 * self.mlp_embedding_size] + self.mlp_hidden_size, self.dropout_prob
        )
        self.mlp_layers.logger = None  # remove logger to use torch.save()
        if self.mf_train and self.mlp_train:
            self.predict_layer = nn.Linear(
                self.mf_embedding_size + self.mlp_hidden_size[-1], 1
            )
        elif self.mf_train:
            self.predict_layer = nn.Linear(self.mf_embedding_size, 1)
        elif self.mlp_train:
            self.predict_layer = nn.Linear(self.mlp_hidden_size[-1], 1)
        self.sigmoid = nn.Sigmoid()
        self.loss = nn.MSELoss()

        # parameters initialization
        if self.use_pretrain:
            self.load_pretrain()
        else:
            self.apply(self._init_weights)

    def load_pretrain(self):
        r"""A simple implementation of loading pretrained parameters."""
        mf = torch.load(self.mf_pretrain_path, map_location="cpu")
        mlp = torch.load(self.mlp_pretrain_path, map_location="cpu")
        mf = mf if "state_dict" not in mf else mf["state_dict"]
        mlp = mlp if "state_dict" not in mlp else mlp["state_dict"]
        self.user_mf_embedding.weight.data.copy_(mf["user_mf_embedding.weight"])
        self.item_mf_embedding.weight.data.copy_(mf["item_mf_embedding.weight"])
        self.user_mlp_embedding.weight.data.copy_(mlp["user_mlp_embedding.weight"])
        self.item_mlp_embedding.weight.data.copy_(mlp["item_mlp_embedding.weight"])

        mlp_layers = list(self.mlp_layers.state_dict().keys())
        index = 0
        for layer in self.mlp_layers.mlp_layers:
            if isinstance(layer, nn.Linear):
                weight_key = "mlp_layers." + mlp_layers[index]
                bias_key = "mlp_layers." + mlp_layers[index + 1]
                assert (
                    layer.weight.shape == mlp[weight_key].shape
                ), f"mlp layer parameter shape mismatch"
                assert (
                    layer.bias.shape == mlp[bias_key].shape
                ), f"mlp layer parameter shape mismatch"
                layer.weight.data.copy_(mlp[weight_key])
                layer.bias.data.copy_(mlp[bias_key])
                index += 2

        predict_weight = torch.cat(
            [mf["predict_layer.weight"], mlp["predict_layer.weight"]], dim=1
        )
        predict_bias = mf["predict_layer.bias"] + mlp["predict_layer.bias"]

        self.predict_layer.weight.data.copy_(predict_weight)
        self.predict_layer.bias.data.copy_(0.5 * predict_bias)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            normal_(module.weight.data, mean=0.0, std=0.01)

    def forward(self, user, item):
        user_mf_e = self.user_mf_embedding(user)
        item_mf_e = self.item_mf_embedding(item)
        user_mlp_e = self.user_mlp_embedding(user)
        item_mlp_e = self.item_mlp_embedding(item)
        if self.mf_train:
            mf_output = torch.mul(user_mf_e, item_mf_e)  # [batch_size, embedding_size]
        if self.mlp_train:
            mlp_output = self.mlp_layers(
                torch.cat((user_mlp_e, item_mlp_e), -1)
            )  # [batch_size, layers[-1]]
        if self.mf_train and self.mlp_train:
            output = self.predict_layer(torch.cat((mf_output, mlp_output), -1))
        elif self.mf_train:
            output = self.predict_layer(mf_output)
        elif self.mlp_train:
            output = self.predict_layer(mlp_output)
        else:
            raise RuntimeError(
                "mf_train and mlp_train can not be False at the same time"
            )
        return output.squeeze(-1)

    def calculate_loss(self, user,item,rating):
        user = user.long()
        item = item.long()
        label = rating

        output = self.forward(user, item)
        return self.loss(output, label)

    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        predict = self.sigmoid(self.forward(user, item))
        return predict


class NeuMF:
    def __init__(self):
        self.nepochs = 1000
        self.lr = 0.001
        self.batch_size = 2048
        self.mf_embedding_size = 64
        self.mlp_embedding_size = 64
        self.mlp_hidden_size = [128, 64]
        self.dropout_prob = 0
        self.mlp_train = False
        self.mf_train = True
        self.use_pretrain = False
        self.mlp_pretrain_path = None
        self.mf_pretrain_path = None
        self.lambd = 0.0001
        self.device = 'cpu'
        if torch.cuda.is_available():
            self.device = 'cuda'




    def train(self, ds,valid_ds = None,valid_funcs=None,cb_progress=lambda x:None,patience=7):
        assert sp.isspmatrix_csr(ds)

        pos_ds=del_neg(ds)
        self.n, self.m = ds.shape

        early_stopping = EarlyStopping(patience=patience, verbose=True)
        config = self.__dict__
        self.model = neumf(self.n,self.m,config)
        self.model.to(self.device)
        loss_fun = BPRLoss#
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
        for t in range(self.nepochs):
            for step, (batch_u, batch_i, batch_neg_i) in enumerate(tqdm(loader, desc="Processing", leave=True)):
                if use_gpu:
                    batch_u = batch_u.cuda()
                    batch_i = batch_i.cuda()
                    batch_neg_i = batch_neg_i.cuda()
                pos_pred = self.model(batch_u, batch_i)
                neg_pred = self.model(batch_u, batch_neg_i)

                loss = loss_fun(pos_pred, neg_pred)  #
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



def del_neg(csr):

    coo = csr.tocoo()

    mask = coo.data != -1
    filtered_coo = csr_matrix((coo.data[mask], (coo.row[mask], coo.col[mask])), shape=csr.shape)

    filtered_csr = filtered_coo.tocsr()
    return filtered_csr