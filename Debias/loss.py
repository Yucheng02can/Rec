# -*- coding: utf-8 -*-
# @Time    : 2024/8/19 17:12
# @Author  : colagold
# @FileName: loss.py

def BPRLoss(pos_scores,neg_scores):
    return -(pos_scores - neg_scores).sigmoid().log().mean()