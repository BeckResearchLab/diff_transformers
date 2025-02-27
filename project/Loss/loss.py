import torch
import sys
import os
import pandas as pd


parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(parent_dir)

import HelperFunctions.definitions as df
import HelperFunctions.msd as msd
import HelperFunctions.features as ft

def masked_loss(pred, tgt, mask, loss_fn):
    """
    Computes the masked loss, where loss is only computed for non-padded tokens.

    Args:
    - pred: Predictions from the model, shape [batch_size, seq_len, ...].
    - tgt: Target values, shape [batch_size, seq_len, ...].
    - mask: Padding mask, shape [batch_size, seq_len], where 1 indicates valid tokens and 0 indicates padding.
    - loss_fn: The loss function to apply (e.g., nn.MSELoss, nn.CrossEntropyLoss) with reduction='none'.

    Returns:
    - loss: Scalar value of the masked loss.
    """
    raw_loss = loss_fn(pred, tgt)
    mask = mask.unsqueeze(-1).expand_as(raw_loss)
    masked_loss = raw_loss * ~mask
    total_loss = masked_loss.sum() / (~mask).sum()

    return total_loss

def DiffusionLoss(Predictions, padding_mask, target):

    pred_track = df.unpad(Predictions, padding_mask);
    tgt_track = df.unpad(target, padding_mask);

    dataForMSD_PRED = df.dataFixer(pred_track, False)
    dataForMSD_TGT = df.dataFixer(tgt_track, False)

    output_list = []
    tgt_list = []
    for data in dataForMSD_PRED:
        dframe = pd.DataFrame(data=data)
        dframe = msd.msd_calc(dframe, len(data))
        res, trash = ft.alpha_calc(dframe)
        output_list.append(res)
    for data1 in dataForMSD_TGT:
        dframe1 = pd.DataFrame(data=data1)
        dframe1 = msd.msd_calc(dframe1, len(data))
        res1, trash1 = ft.alpha_calc(dframe1)
        tgt_list.append(res1)


    print(output_list)
    print(tgt_list)




