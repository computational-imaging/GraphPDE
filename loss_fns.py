import torch
import math
import numpy as np

def mse_prior(regularize, mode,sig, x1, x2):
    if regularize:
        if mode =="l2":
            return {"mse": ((x1['model_out'] - x2['img'])**2).mean(),
                    "batch_vecs":sig*torch.mean((x1['batch_vecs']**2))} #0.05
        elif mode =="l1":
            return {"mse": ((x1['model_out'] - x2['img']).abs()).mean(),
                    "batch_vecs":sig*torch.mean((x1['batch_vecs']**2))} #0.05
    else:
        if mode =="l2":
            return {"mse": ((x1['model_out'] - x2['img'])**2).mean()}
        elif mode =="l1":
            return {"mse": ((x1['model_out'] - x2['img']).abs()).mean()}


def loss_gt_mse(output_type,output,data_copy):
    if data_copy.var!=0:
        gnn_prdiction = output.eval[:,:]
        gt = output.v_gt[:,:].cuda()-output.noise[:,0]
    else:
        gnn_prdiction = output.eval[:,:]
        if output_type=='v':
            gt = output.v_gt[:,:].cuda()
        elif output_type=='a':
            gt = output.a_gt[:,:].cuda()
    return {"mse":((gnn_prdiction[:,0]-gt[:,0])**2).mean()}


