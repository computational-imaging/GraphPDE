import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"inverse_scripts"))

import utils
import torch
from glob import glob
import torch
import inverse_gnn
import modules
import gnn_module
import dataio
import numpy as np
from functools import partial
import configargparse
p = configargparse.ArgumentParser()

p.add('-c', '--config', required=False, default='InverseProblem/config/init_state_gnn_p.ini', is_config_file=True, help='Path to config file.')

p.add_argument('--logging_root', type=str, default="./log/" , help='root for logging')
p.add_argument('--note', type=str, default=None, help='root for logging')

p.add_argument('--experiment_name', type=str, default="GNN",
            help='Name of subdirectory in logging_root where summaries and checkpoints will be saved.')
p.add_argument('--path_to_data', type=str, default='./data', help='root for logging')
p.add_argument('--start_observation_index',type=int,default=1)
p.add_argument('--repeat_time',type=int,default=120)


#solver setup
p.add_argument('--solver_path', type=str, default='./data/model_zoo/gnn_solver_3136000.pth')
p.add_argument('--edge_features', nargs='+', type=str, default=['dist','direction'])

#inverse problem
p.add_argument('--time_steps', nargs='+', type=int, default=[16])
p.add_argument('--mask_type',type=str,default="random")
p.add_argument('--sensor_num',type=int,default=None)
p.add_argument('--obversation_step',type=int,default=1)
p.add_argument('--sensor_percent',type=float,default=None)
#optimizer setup:
p.add_argument('--lr',type=float,default=1e-2)
p.add_argument('--reg',type=float,default=0)

p.add_argument('--lr_decay',type=float,default=0.9)
p.add_argument('--lr_decay_type',type=str,default="per_iter")
p.add_argument('--gradient_clip',type=float,default=0)
p.add_argument('--lr_decay_steps',type=int,default=200)
p.add_argument('--num_iter',type=int,default=2000)
p.add_argument('--loss_type', type=str,default="l2")
p.add_argument('--progressive', action='store_true', default=False)

p.add_argument('--convergence_stop', action='store_true', default=False)

#prior setup
p.add_argument('--noprior', action='store_true', default=False)
p.add_argument('--prior_type',type=str,default="init_state")
p.add_argument('--prior_path', type=str, default='./data/model_zoo/prior')
p.add_argument('--outmost_nl', action='store_false', default=True)
p.add_argument('--outmost_nonlinearity',type=str, default='sigmoid',
                choices=['sine', 'relu', 'requ', 'gelu', 'selu', 'softplus', 'tanh', 'swish','sigmoid'])
#dataset setup
p.add_argument('--mesh_type', type=str, default='coarse')
p.add_argument('--resolution',type=int,default=64)
p.add_argument('--start_index',type=int,default=0)
p.add_argument('--dataset_size',type=int,default=1)

p.add_argument('--store',action='store_true',default=False)
p.add_argument('--gpu', type=int, default=3, help='which gpu to use')

opt = p.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpu)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
torch.cuda.set_device(opt.gpu)
print('--- Load Configuration ---')

if opt.sensor_num==None:
    opt.sensor_num = opt.sensor_percent

device = 'cuda:{}'.format(opt.gpu)

fname_vars = ['experiment_name','repeat_time','loss_type','progressive','sensor_num','time_steps','obversation_step','start_observation_index','gradient_clip','lr',"reg","lr_decay","lr_decay_steps"]
opt.experiment_name = ''.join([f'{k}_{vars(opt)[k]}_|_'.replace('[', '(').replace(']',')') for k in fname_vars])[16:-1]
print(opt.experiment_name)

root_path = os.path.join(opt.logging_root, opt.experiment_name)
mse_vs_time = os.path.join(root_path, "mse_vs_time")
run_time = os.path.join(root_path, "run_time")
individual_npy = os.path.join(root_path, "individual_npy")
gradient_norms = os.path.join(root_path, "gradient_norms")

utils.cond_mkdir(root_path)
utils.cond_mkdir(mse_vs_time)
utils.cond_mkdir(run_time)
utils.cond_mkdir(individual_npy)
utils.cond_mkdir(gradient_norms)

if opt.mesh_type=="coarse":
    data_file = '{}/data/coarse_valid'.format(opt.path_to_data)
elif opt.mesh_type=="fine":
    data_file =  '{}/data/fine_valid'.format(opt.path_to_data)

print('--- Load Prior ---')
if opt.noprior:
    prior = partial(inverse_gnn.identity)
else:
    if opt.prior_type!="both":
        prior = modules.CoordinateNet_autodecoder(latent_size=64, nl='relu', in_features=64+2, out_features=1,
                                        hidden_features=256,
                                        num_hidden_layers=6, num_pe_fns=3,
                                        w0=60,use_pe=True,skip_connect=None,dataset_size=10000,
                                        outmost_nonlinearity=opt.outmost_nonlinearity,outermost_linear=not(opt.outmost_nl)).to(device)
        opt.prior_path = "{}/{}".format(opt.path_to_data,opt.prior_path)
        if opt.prior_type=="density":
            opt.prior_path = "{}.pth".format(opt.prior_path)
        elif opt.prior_type=="init_state":
            opt.prior_path = "{}_field.pth".format(opt.prior_path)
        checkpoint_prior = torch.load(opt.prior_path,map_location='cuda:{}'.format(opt.gpu))
        prior.load_state_dict(checkpoint_prior)
        prior.lat_vecs = None
        prior = prior.to(device)
    else:
        prior_density = modules.CoordinateNet_autodecoder(latent_size=64, nl='relu', in_features=64+2, out_features=1,
                                        hidden_features=256,
                                        num_hidden_layers=6, num_pe_fns=3,
                                        w0=60,use_pe=True,skip_connect=None,dataset_size=10000,
                                        outmost_nonlinearity=opt.outmost_nonlinearity,outermost_linear=not(opt.outmost_nl)).to(device)
        opt.prior_path_d = "{}/{}".format(opt.path_to_data,opt.prior_path)
        opt.prior_path_d = "{}.pth".format(opt.prior_path_d)
        checkpoint_prior = torch.load(opt.prior_path_d,map_location='cuda:{}'.format(opt.gpu))
        prior_density.load_state_dict(checkpoint_prior)
        prior_density.lat_vecs = None
        prior_density = prior_density.to(device)

        prior_init_state = modules.CoordinateNet_autodecoder(latent_size=64, nl='relu', in_features=64+2, out_features=1,
                                        hidden_features=256,
                                        num_hidden_layers=6, num_pe_fns=3,
                                        w0=60,use_pe=True,skip_connect=None,dataset_size=10000,
                                        outmost_nonlinearity=opt.outmost_nonlinearity,outermost_linear=not(opt.outmost_nl)).to(device)
        opt.prior_path_i = "{}/{}".format(opt.path_to_data,opt.prior_path)
        opt.prior_path_i = "{}_field.pth".format(opt.prior_path_i)
        checkpoint_prior = torch.load(opt.prior_path_i,map_location='cuda:{}'.format(opt.gpu))
        prior_init_state.load_state_dict(checkpoint_prior)
        prior_init_state.lat_vecs = None
        prior_init_state = prior_init_state.to(device)


print('--- Load Solver ---')
opt.solver_path = "{}/{}".format(opt.path_to_data,opt.solver_path)
gnn_solver = gnn_module.mesh_PDE(edge_dim=3,node_dim=4, latent_dim = 256,num_steps=10,layer_norm=True,
                                nl='relu',var=0,batch_norm=False,normalize=True,encoder_nl='relu',
                                diffMLP=True).to(device)
checkpoint_gnn = torch.load(opt.solver_path,map_location='cuda:{}'.format(opt.gpu))
gnn_solver.load_state_dict(checkpoint_gnn['model_state_dict'])
gnn_solver = gnn_solver.to(device)
graph_update_fn = partial(dataio.wave_data_update,('u', 'v', 'density','type'))

num_experiments = len(glob(os.path.join(opt.logging_root, opt.experiment_name) + 'config*'))
p.write_config_file(opt, [os.path.join(root_path, 'config_{}.ini'.format(num_experiments))])

if opt.loss_type=='l2':
    loss_fn = partial(inverse_gnn.l2)
elif opt.loss_type=="l1":
    loss_fn = partial(inverse_gnn.l1)

print('--- Solve Inverse Problem ---')
inverse_gnn.test_inverse(prior=prior,gnn_solver=gnn_solver,graph_update_fn=graph_update_fn,mask_type=opt.mask_type,start_index=opt.start_index,
                        dataset_size=opt.dataset_size,time_steps=opt.time_steps,
                        sensor_num=opt.sensor_num,num_iter=opt.num_iter,lr=opt.lr,lr_decay=opt.lr_decay,lr_decay_steps=opt.lr_decay_steps,
                        log_path=root_path,store=opt.store,data_file=data_file,edge_features=opt.edge_features,prior_type=opt.prior_type,
                        noprior = opt.noprior,loss_fn=loss_fn,progressive=opt.progressive,convergence_stop=opt.convergence_stop,
                        path_to_data=opt.path_to_data,obversation_step=opt.obversation_step,start_observation_index=opt.start_observation_index,
                        gradient_clip=opt.gradient_clip,lr_decay_type=opt.lr_decay_type,reg=opt.reg,device=device,
                        repeat_time=opt.repeat_time)
