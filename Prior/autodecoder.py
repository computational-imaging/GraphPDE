# Enable import from parent package
import sys
import os
import contextlib
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append("..")
import dataio
import utils
import train
import loss_fns
import modules
import numpy as np
import torch
from glob import glob
import wandb
from torch.utils.data import DataLoader
import configargparse
from functools import partial

p = configargparse.ArgumentParser()
p.add('-c', '--config', required=False, is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default='./logs', help='root for logging')
p.add_argument('--experiment_name', type=str, default="initial_states",
               help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--prior', type=str, default="init_state",
               help='type of prior')

# General model options
p.add_argument('--latent_size', type=int, default=64)
p.add_argument('--skip_connect', nargs='+', type=int, default=None)
p.add_argument('--layers', type=int, default=6)
p.add_argument('--features', type=int, default=256)
p.add_argument('--num_pe_fns', type=int, default=3)
p.add_argument('--w0', type=int, default=60)
p.add_argument('--outmost_nl', action='store_true', default=False)
p.add_argument('--outmost_nonlinearity',type=str, default='relu',
                choices=['sine', 'relu', 'requ', 'gelu', 'selu', 'softplus', 'tanh', 'swish','sigmoid'])
p.add_argument('--nl', type=str, default='relu',
               choices=['sine', 'relu', 'requ', 'gelu', 'selu', 'softplus', 'tanh', 'swish'],
               help='nonlinear activation to use (all except sine use positional encoding)')
p.add_argument('--use_pe', action='store_true', default=False,
               help='whether use pe')
p.add_argument('--regularize', action='store_true', default=False,
               help='whether regularize latent code for guassian prior')
p.add_argument('--sig', type=int, default=0.1)
p.add_argument('--loss_mode',type=str, default='l2') 
            
# General training options
p.add_argument('--dataset_size', type=int, default=100)
p.add_argument('--sampled_points', type=int, default=900)
p.add_argument('--batch_size', type=int, default=20)
p.add_argument('--lr', type=float, default=5e-4, help='learning rate. default=5e-4')
p.add_argument('--lr_latent', type=float, default=1e-3, help='learning rate. default=1e-3')

p.add_argument('--num_epochs', type=int, default=1500,
               help='Number of epochs to train for.')
p.add_argument('--epochs_til_ckpt', type=int, default=10,
               help='Time interval in seconds until checkpoint is saved.')
p.add_argument('--steps_til_summary', type=int, default=500,
               help='Time interval in seconds until tensorboard summary is saved.')
p.add_argument('--num_workers', type=int, default=10,
               help='Number of dataloader workers.')
p.add_argument('--jitter', action='store_true', default=False,
               help='whether add jitter to coords in training')
p.add_argument('--irregular_mesh', action='store_true', default=False)

p.add_argument('--wandb', action='store_true', default=False,
               help='whether use wandb')
p.add_argument('--checkpoint_path', default=None, help='Checkpoint to trained model.')
p.add_argument('--gpu', type=int, default=0, help='which gpu to use')

opt = p.parse_args()
# opt.use_pe = True
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.cuda.set_device(opt.gpu)

np.random.seed(seed=121)
torch.manual_seed(121)
# torch.set_num_threads(1)

print('--- Run Configuration ---')

# make experiment name, create directory
if opt.config is None:
    fname_vars = ['experiment_name', 'layers', 'features', 'nl', 'latent_size', 
                  'lr', 'w0', 'num_pe_fns','skip_connect']
    opt.experiment_name = ''.join([f'{k}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]

num_experiments = len(glob(os.path.join(opt.logging_root, opt.experiment_name) + '*'))
if num_experiments > 0:
    opt.experiment_name += f'_{num_experiments}'

root_path = os.path.join(opt.logging_root, opt.experiment_name)
utils.cond_mkdir(root_path)

# if opt.wandb:
    # wandb.init(project='prior', entity='username',name='deepsdf{}_{}_{}'.format(opt.dataset_size,opt.use_pe,opt.experiment_name))

density_dataset = dataio.density(dataset_size=opt.dataset_size,sampled_points = opt.sampled_points, jitter=opt.jitter,type=opt.prior)
dataloader = DataLoader(density_dataset, shuffle=False, batch_size=opt.batch_size, pin_memory=True, num_workers=opt.num_workers)

# Define the model.
model = modules.CoordinateNet_autodecoder(latent_size=opt.latent_size, nl=opt.nl, in_features=opt.latent_size+2, out_features=1,
                                  hidden_features=opt.features,
                                  num_hidden_layers=opt.layers, num_pe_fns=opt.num_pe_fns,
                                  w0=opt.w0,use_pe=opt.use_pe,skip_connect=opt.skip_connect,dataset_size=opt.dataset_size,
                                  outmost_nonlinearity=opt.outmost_nonlinearity,outermost_linear=not(opt.outmost_nl))

model.cuda()
print(f'Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
loss_fn = partial(loss_fns.mse_prior,opt.regularize,opt.loss_mode,opt.sig)
summary_fn = partial(utils.summary_autodecoder,True,opt.latent_size,opt.irregular_mesh)

# Save command-line parameters log directory.
p.write_config_file(opt, [os.path.join(root_path, 'config.ini')])

# Save text summary of model into log directory.
with open(os.path.join(root_path, "model.txt"), "w") as out_file:
    out_file.write(str(model))

if opt.wandb:
    config = wandb.config
    config.folder = root_path
    config.dataset_size = opt.dataset_size
    config.problem = opt.experiment_name
    wandb.watch(model,log_freq=50)
else:
    wandb = None

train.train(model=model, train_dataloader=dataloader, epochs=opt.num_epochs, lr=opt.lr,
               steps_til_summary=opt.steps_til_summary, epochs_til_checkpoint=opt.epochs_til_ckpt,
               model_dir=root_path, loss_fn=loss_fn, summary_fn=summary_fn, lr_latent=opt.lr_latent,wandb=wandb)