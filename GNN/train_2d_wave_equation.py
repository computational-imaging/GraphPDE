import torch
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
np.random.seed(seed=121)
torch.manual_seed(121)
from torch_geometric.data import DataLoader
import gnn_module
import dataio
import wandb
import loss_fns
import utils
import train_GNN
from glob import glob
from functools import partial

import configargparse
p = configargparse.ArgumentParser()
p.add('-c', '--config', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--log',action='store_true',  default=False,
               help='whether log the training data')
p.add_argument('--file', type=str, default='./data/training/', help='root for logging')
p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default="solver_irregular",
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')

#setup dataloader
p.add_argument('--dataset_size', type=int, default=1000,help="dataset size")
p.add_argument('--step_size', type=int, default=5,help="step size")
p.add_argument('--endtime', type=int, default=250,help="trajectory step max step number in dataset")
p.add_argument('--val_dataset', type=int, default=5,help="validation dataset size")
p.add_argument('--train_dataset', type=int, default=5,help="ploted training dataset size")

p.add_argument('--batch_size', type=int, default=1,help="batch size")
p.add_argument('--dataset_update',action='store_true',  default=False,
               help='whether update dataset throughout training')

p.add_argument('--random',action='store_true',  default=False,
               help='whether use random dataset')

p.add_argument('--normalize',action='store_true',  default=False,
               help='whether use random dataset')

#setup GNN
p.add_argument('--diffMLP', action='store_true', default=False,
               help='use different MLP in each message passing steps')
p.add_argument('--time_steps', type=int, default=1)
p.add_argument('--out_features', type=int, default=1)

p.add_argument('--node_features', type=str, default=['u','v','density','type'])
p.add_argument('--edge_features', nargs='+', type=str, default=['dist','direction'])
p.add_argument('--output', type=str, default='v',
               choices=['x', 'v', 'a'],
               help='output function')

p.add_argument('--features', type=int, default=256)
p.add_argument('--layers', type=int, default=2)
p.add_argument('--steps', type=int, default=10, help='num message passing steps')
p.add_argument('--nl', type=str, default='relu',
               choices=['sine', 'relu', 'requ', 'gelu', 'selu', 'softplus', 'tanh', 'swish'],
               help='nonlinear activation to use')
p.add_argument('--encoder_nl', type=str, default='relu',
               choices=['sine', 'relu', 'requ', 'gelu', 'selu', 'softplus', 'tanh', 'swish'],
               help='nonlinear activation to use')
p.add_argument('--layer_norm', type=bool, default=True)
p.add_argument('--lr', type=float, default=1e-4, help='learning rate. default=1e-4')
p.add_argument('--decay_factor', type=float, default=1e-4, 
               help='learning rate decay by this factor after num_epoch epochs, exponetial decay . default=1e-4')
p.add_argument('--noise_var',type=float, default=0, help='input noise scale')
p.add_argument('--batchnorm', action='store_true', default=False,
               help='whether use batchnorm')
p.add_argument('--double_precision',action='store_true',default=False)
#setup loss function
p.add_argument('--loss_fn', type=str, default='mse',
               choices=['fem_x','fem_v','fd_x', 'fd_v', 'mse'],
               help='loss function')

#setup training parameters
p.add_argument('--epochs_til_ckpt', type=int, default=1,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--num_epochs', type=int, default=500,
               help='Number of epochs to train for.')
p.add_argument('--steps_til_summary', type=int, default=1,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--wandb', action='store_true', default=False,
               help='whether use wandb')
p.add_argument('--lr_schedule', action='store_true', default=False,
               help='whether use lr_schedule')

p.add_argument('--gpu', type=int, default=2, help='which gpu to use')
opt = p.parse_args()
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.cuda.set_device(opt.gpu)
device = "cuda:{}".format(opt.gpu)
print('--- Run Configuration ---')


if opt.config is None:
    fname_vars = ['loss_fn', 'dataset_size','batch_size','layers', 'features', 'steps', 'nl',
                  'lr','node_features','edge_features','normalize']
    opt.experiment_name = ''.join([f'{k}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]

num_experiments = len(glob(os.path.join(opt.logging_root, opt.experiment_name) + '*'))
if num_experiments > 0:
    opt.experiment_name += f'_{num_experiments}'

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)
    
# Define the model.
print('--- Load GNN Model ---')
model = gnn_module.mesh_PDE(edge_dim=3,node_dim=4, latent_dim = 256,num_steps=10,layer_norm=True,
                                nl='relu',var=0,batch_norm=False,normalize=True,encoder_nl='relu',diffMLP=True).cuda()

print('--- Load Dataset ---')
train_datalist = dataio.wave_data_2D_irrgular(opt.dataset_size,node_features=opt.node_features,edge_features = opt.edge_features,
                                        file=opt.file,step_size=opt.step_size,endtime=opt.endtime,train=True, device=device, var=opt.noise_var)
print(len(train_datalist))
dataset = DataLoader(train_datalist, batch_size=opt.batch_size,shuffle=True)
graph_update_fn = partial(dataio.wave_data_update,opt.node_features)

# validation dataset sample
val_loaders = []
for j in range(opt.val_dataset):
    train_datalist =dataio.wave_data_2D_irrgular(1,node_features=opt.node_features,edge_features = opt.edge_features,
                                        file=opt.file,step_size=opt.step_size,endtime=opt.endtime,index=j, train=False, device="cpu", var=opt.noise_var)
    val_loader = DataLoader(train_datalist, batch_size=1,shuffle=False)
    val_loaders.append(val_loader)

#define loss function
if opt.loss_fn == "mse":
    loss_fn = partial(loss_fns.loss_gt_mse,opt.output)
else:
    NotImplemented


#setup wandb if one want 
wandb = None

log_iter = partial(utils.log_iter_2d_irregular,opt.time_steps)
train_GNN.train(model=model, train_dataloader=dataset, val_loader=val_loaders,epochs=opt.num_epochs, lr=opt.lr,
                epochs_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, wandb=wandb,output_type=opt.output,graph_update_fn=graph_update_fn,
               lr_schedule=opt.lr_schedule,decay_factor=opt.decay_factor,log_iter=log_iter)
