
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import dataio
import utils
import numpy as np
import torch
from glob import glob
import matplotlib.pyplot as plt
np.random.seed(seed=121)
import torch
from scipy import spatial
import h5py
from functools import partial

from utils_inverse import *
gpu_num = 0

torch.cuda.set_device(gpu_num)
from glob import glob
os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_num)
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_LAUNCH_BLOCKING"] = str(1)
PATH = './Inverse_Problem/data_validation'


def l2(prediction,sparse_abservation,mask,steps,obversation_step=1,start_observation_index=1):
    number_of_observation = sparse_abservation[start_observation_index:steps+1:obversation_step].shape[0]
    return (((prediction[start_observation_index:steps+1:obversation_step,:]*mask-sparse_abservation[start_observation_index:steps+1:obversation_step,:])**2).sum(dim=1)).sum()/mask.sum()/number_of_observation

def l1(prediction,sparse_abservation,mask,steps,obversation_step=1,start_observation_index=1):
    number_of_observation = sparse_abservation[start_observation_index:steps+1:obversation_step].shape[0]
    return (((prediction[start_observation_index:steps+1:obversation_step,:]*mask-sparse_abservation[start_observation_index:steps+1:obversation_step,:]).abs()).sum(dim=1)).sum()/mask.sum()/number_of_observation

def identity(model_input,latent):
    return {'model_out':latent}

def gen_dataset(file,mask=None,time_steps=2,start_with_0=False,edge_features=None,prior_type=None,gnn_solver=None,graph_update_fn=None):
    
    train_datalist = dataio.file_dataloader(file=file,node_features=('u', 'v', 'density', 'type'),edge_features = edge_features,
                                            step_size=5,endtime=250)
    if prior_type=='density':
        observed_fields = torch.zeros(time_steps+1,train_datalist[0].x.shape[0])
        gt_fields = torch.zeros(time_steps+1,train_datalist[0].x.shape[0])
        for idx,graph in enumerate(train_datalist):
            observed_g = graph.clone()
            gt_parameter = observed_g.x[:,2].clone()
            observed_g.x[:,2] = 0 #set density to 0
            if idx==0:
                init_graph=observed_g.clone()    # if idx==0, init graph
                observed_fields[0,:] = observed_g.x[:,[0]].permute(1,0).float()
                gt_fields[0,:] = observed_g.x[:,[0]].permute(1,0).float()
            observed_g.gt[:,0]= observed_g.gt[:,0]*mask  # mask out observation points
            observed_fields[idx+1,:] = (observed_g.gt).permute(1,0).float() #store it in observed_fields
            gt_fields[idx+1,:] = (graph.gt).permute(1,0).float()
            if idx>=time_steps-1: 
                break
    elif prior_type=='init_state':
        observed_fields = torch.zeros(time_steps+1,train_datalist[0].x.shape[0])
        gt_fields = torch.zeros(time_steps+1,train_datalist[0].x.shape[0])
        for idx,graph in enumerate(train_datalist):
            observed_g = graph.clone()
            if idx==0:
                gt_parameter = observed_g.x[:,0].clone()
                gt_fields[0,:] = observed_g.x[:,[0]].permute(1,0).float()
                observed_g.x[:,0] = (observed_g.x[:,0]*mask) #observed points
                observed_fields[0,:] = observed_g.x[:,[0]].permute(1,0).float()
                observed_g.x[:,0] = 0            # remove init states
                init_graph=observed_g.clone()    # if idx==0, init graph
            gt_fields[idx+1,:] = (graph.gt).permute(1,0).float()
            observed_g.gt[:,0]= observed_g.gt[:,0]*mask  # mask out observation points
            observed_fields[idx+1,:] = (observed_g.gt).permute(1,0).float() #store it in observed_fields
            if idx>=time_steps-1: 
                break
    return init_graph, observed_fields,gt_parameter,gt_fields

def gen_mask(coords,mask_type="random",mask_percent=0,resolution=128,bdd_nodes=None,index=0,path_to_data=None):
    if mask_type=="random_nodes":
        n = coords.shape[0]
        if isinstance(mask_percent,int):
            masked_num = mask_percent
        else:
            masked_num = int(mask_percent*n)
        masked_index=torch.randperm(n)[0:masked_num]
        mask = torch.zeros(n)
        mask.view(-1)[masked_index]=1
    elif mask_type=="load":
        sensor_coords = np.load('{}/data_validation/data/random_coords/{}/{}.npy'.format(path_to_data,mask_percent,index),allow_pickle=True)
        assert sensor_coords.shape[0]==mask_percent
        tree = spatial.KDTree(coords)
        a = tree.query(sensor_coords)
        mask = np.zeros(coords.shape[0])
        mask[a[1][:]]=1
        masked_index = a[1][:]
        mask = torch.tensor(mask)
    return mask,masked_index,coords[masked_index]

def gnn_update_density(prior,gnn_solver,model_input,latent,time_steps,init_graph,n,graph_update_fn,noprior=False,device="cuda"):
    for t in range(time_steps):
        if t==0:
            input_graph = init_graph.clone().to(device)
            if noprior:
                structure = latent*1+(1-latent)*3
                structure = torch.clamp(structure,1,3)
            else:
                structure = (prior(model_input=model_input, latent=latent)['model_out']*2+1).squeeze()
            input_graph.x[:,2] = structure.clone()
            prediction = torch.zeros([time_steps+1,n]).to(device)
            prediction[0,:] = input_graph.x[:,0]
            ploted_graph = input_graph.clone()
        input_graph_clone = input_graph.clone()
        u_old = input_graph.x[:,0].clone()
        output_graph = gnn_solver(input_graph)
        prediction[[t+1],:] = (u_old.unsqueeze(-1)+output_graph.x[:,:].clone()).permute(1, 0)
        input_graph = graph_update_fn(output_graph,input_graph_clone,train=False,keep_grad=True)
    return prediction,ploted_graph

def gnn_update_init_state(prior,gnn_solver,model_input,latent,time_steps,init_graph,n,graph_update_fn,bdd_mask,noprior=False,device="cuda"):
    for t in range(time_steps):
        if t==0:
            input_graph = init_graph.clone().to(device)
            if noprior:
                init_state = latent*bdd_mask
            else:
                holder = (prior(model_input=model_input, latent=latent)['model_out']*1.097-0.5622).squeeze()
                init_state = (holder/holder.abs().max())*bdd_mask
            input_graph.x[:,0] = init_state
            ploted_graph = input_graph.clone()
            prediction = torch.zeros([time_steps+1,n]).to(device)
            prediction[0,:] = input_graph.x[:,0]
        input_graph_clone = input_graph.clone()
        u_old = input_graph.x[:,0].clone()
        output_graph = gnn_solver(input_graph)
        prediction[[t+1],:] = (u_old.unsqueeze(-1)+output_graph.x[:,:].clone()).permute(1, 0)
        input_graph = graph_update_fn(output_graph,input_graph_clone,train=False,keep_grad=True)
    return  prediction, ploted_graph

def setup_inverseProblem(time_steps,data_file=None,mask_percent=0,plot=False,mask_type="random",index=0,start_with_0=False,path_to_data=None,
                         gpu_num=0,edge_features=None,prior_type=None,resolution=128,gnn_solver=None,graph_update_fn=None,device="cuda"):
    data_file = sorted(glob('{}/*.npy'.format(data_file)))[index]
    
    coords = np.load(data_file,allow_pickle=True)[0]['nodes_low']
    bdd_nodes = np.load(data_file,allow_pickle=True)[0]['is_boundary_low']
    mask,mask_idx,sensor_coords = gen_mask(coords,mask_percent=mask_percent,mask_type=mask_type,resolution=resolution,bdd_nodes=bdd_nodes,index=index,path_to_data=path_to_data)
    
    coords = torch.tensor(coords).unsqueeze(0).float().to(device)
    model_input = {'coords':coords}

    
    init_graph,sparse_abservation,gt_parameter,gt_fields= gen_dataset(file=data_file,time_steps=time_steps,mask=mask,
                                                                    start_with_0=start_with_0,edge_features=edge_features,
                                                                    prior_type=prior_type,gnn_solver=gnn_solver,graph_update_fn=graph_update_fn)

    if plot:
        plot_input_graph(init_graph)
        coords = init_graph.coords
        cells = init_graph.cell
        plt.tricontourf(coords[:,0],coords[:,1],cells,sparse_abservation[1,:])
        plt.triplot(coords[:,0],coords[:,1],cells,linewidth=0.1)
        plt.title("input_u")
        
    return init_graph,sparse_abservation, mask,mask_idx, gt_parameter,model_input,gt_fields,sensor_coords


def solver_invserproblem(prior,gnn_solver,graph_update_fn,time_steps,mask_percent=1,num_iter=1000,lr=4e-3,mask_type="load",seed=42,index=0,lr_decay=0.8,lr_decay_steps=100,
                         log_path=None,data_file=None,store=False,start_with_0=False,edge_features=None,prior_type=None,resolution=128,
                         noprior=False,loss_weight=None,loss_fn=None,progressive=False,convergence_stop=False,path_to_data=None,obversation_step=1,
                         start_observation_index=1,gradient_clip=0,lr_decay_type="per_iter",reg = 0,device="cuda",repeat_time=120):
    
    init_graph,sparse_abservation, mask,mask_idx, gt_parameter,model_input,gt_fields,sensor_coords= setup_inverseProblem(time_steps,
                                                    data_file=data_file,mask_percent=mask_percent,mask_type=mask_type,index=index,
                                                    start_with_0=start_with_0,edge_features=edge_features,prior_type=prior_type
                                                    ,resolution=resolution,gnn_solver=gnn_solver,graph_update_fn=graph_update_fn,path_to_data=path_to_data,device=device)
    n = sparse_abservation.shape[1]
    mask = mask.to(device)
    sparse_abservation = sparse_abservation.to(device)
    if prior_type == "init_state":
            vert = init_graph.coords
            tri = init_graph.cell
            mesh,_,_= gen_mesh(vert,tri)
            bdd_mask = solve_Eikonal(mesh)
            bdd_mask = (bdd_mask.compute_vertex_values()**0.8)/((bdd_mask.compute_vertex_values()**0.8).max())
            bdd_mask = torch.tensor(bdd_mask).to(device)
            
    image_path = ("{}/images/{}/sensor_num_{}_timesteps{}_{}_{}".format(log_path,index,mask_percent,time_steps,mask_type,seed))
    utils.cond_mkdir(image_path)
    
    #init latent code
    if noprior:
        if prior_type=="density": init_val = 0.5
        elif prior_type=="init_state": init_val = 0.
        class density_parameter(torch.nn.Module):
            def __init__(self):
                super(density_parameter, self).__init__()
                self.density_parameter = torch.nn.Parameter((torch.ones(mask.shape[0]).to(device))*init_val,requires_grad=True).to(device)
    else:
        class density_parameter(torch.nn.Module):
            def __init__(self):
                super(density_parameter, self).__init__()
                self.density_parameter = torch.nn.Parameter(0*((torch.randn(1,prior.latent_size).to(device))*torch.sqrt(prior.varience).to(device)+prior.mean.to(device)).float(),requires_grad=True).to(device)
                
    density = density_parameter().to(device)
    optim = torch.optim.Adam(params=density.parameters(), lr=lr)
    if lr_decay_type=="per_iter":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optim, gamma=lr_decay, last_epoch=-1)
    elif lr_decay_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, mode='min', factor=0.1, patience=100,verbose=False)

    mse = []
    losses = []
    predictions = []
    estimated_parameters = []
    observation_steps = np.arange(time_steps+1)[start_observation_index:time_steps:obversation_step]
    if progressive:
        observation_steps = np.arange(time_steps+1)[start_observation_index:time_steps:obversation_step]
        observation_timestep_count = len(observation_steps)
        steps_progressive = observation_steps.repeat(repeat_time)
        steps = (np.ones(num_iter).astype(np.int32))*time_steps
        steps[:repeat_time*observation_timestep_count]=steps_progressive.astype(np.int32)
    else:
        observation_timestep_count = time_steps
        steps = (np.ones(num_iter)*time_steps).astype(np.int32)
    print("steps",steps[::repeat_time])

    if loss_fn=="l2":
        loss_fn=partial(l2)
    elif loss_fn=="l1":
        loss_fn=partial(l1)
        

    for i in range(num_iter):
        if prior_type=="density":
            prediction,ploted_graph = gnn_update_density(prior,gnn_solver,model_input,density.density_parameter,steps[i],init_graph,n,graph_update_fn,noprior=noprior,device=device)
            target_index = 2
        elif prior_type=="init_state":
            prediction,ploted_graph = gnn_update_init_state(prior,gnn_solver,model_input,density.density_parameter,steps[i],init_graph,n,graph_update_fn,bdd_mask,noprior=noprior,device=device)
            target_index = 0
            
        predictions.append(prediction)
        loss = loss_fn(prediction,sparse_abservation,mask,steps[i],obversation_step,start_observation_index=start_observation_index)
        loss = loss + reg*((density.density_parameter)**2).mean()

        assert loss!=0

        optim.zero_grad()
        loss.backward()
        if gradient_clip>0:
            torch.nn.utils.clip_grad_norm_(density.parameters(),gradient_clip)

        optim.step()

        mse.append(((ploted_graph.x[:,target_index].detach().cpu()-gt_parameter)**2).mean().detach().cpu().numpy())
        losses.append(loss.detach().cpu().numpy())
        estimated_parameters.append(ploted_graph.x[:,target_index].detach().cpu())
        
        if convergence_stop and (i>repeat_time*observation_timestep_count or not(progressive)) and i>500:
            if not(if_continue(sum(losses[-50:])/len(losses[-50:]),losses[-2])):
                break
        
        if i%500==0:
            print('iter:',i)
            print('loss:{}'.format(loss))
            print('mse:{}'.format(((ploted_graph.x[:,target_index].detach().cpu()-gt_parameter)**2).mean()))
            flat = [sparse_abservation[i,:] for i in np.arange(steps[-1])[start_observation_index:steps[i]+1:obversation_step]]
            flat_gt = [gt_fields[i,:] for i in np.arange(steps[-1])[start_observation_index:steps[i]+1:obversation_step]]
            flat_prdict = [prediction[i,:] for i in np.arange(steps[-1])[start_observation_index:steps[i]+1:obversation_step]]
            if store:
                plot_input_graph(ploted_graph,[gt_parameter],mask_idx=mask_idx)
                plt.savefig("{}/{}.jpg".format(image_path,i))

                plt.show()
        if lr_decay_type == "per_iter":
            if i%lr_decay_steps==0 and i!=0:
                scheduler.step()
        elif lr_decay_type == "plateau":
            scheduler.step(loss)

    if store:
        print(' iter: ',i)
        print(' masked:{} '.format(mask.sum()/mask.shape[0]))
        print('loss:{} '.format(loss))
        print('mse:{} '.format(((ploted_graph.x[:,target_index].detach().cpu()-gt_parameter)**2).mean()))
        flat = [sparse_abservation[i,:] for i in np.arange(steps[-1])[start_observation_index:steps[i]+1:obversation_step]]
        flat_gt = [gt_fields[i,:] for i in np.arange(steps[-1])[start_observation_index:steps[i]+1:obversation_step]]
        flat_prdict = [prediction[i,:] for i in np.arange(steps[-1])[start_observation_index:steps[i]+1:obversation_step]]
        plot_input_graph(ploted_graph,[gt_parameter],mask_idx=mask_idx)
        plt.savefig("{}/{}.jpg".format(image_path,i))
        plt.show()
        
        fig = plt.figure(figsize=[10,3])
        fig.add_subplot(1,2,1)
        plt.plot(losses);plt.yscale('log');plt.title("last loss:{}".format(losses[-1]))
        fig.add_subplot(1,2,2)
        plt.plot(mse);plt.yscale('log');plt.title("last mse:{}".format(mse[-1]))
        plt.savefig("{}/summary.png".format(image_path))
        plt.show()
    dict = {"index":index,
            "gt_observation":sparse_abservation,
            "gt_fields":gt_fields,
            "gt_parameter":gt_parameter,
            'estimated_parameter':estimated_parameters[-1],
            'estimated_parameters':estimated_parameters,
            'predictions':predictions,
            
            "losses":losses,
    
            "mse":mse,
            "mask":mask_percent,
            "timesteps":time_steps,
            "steps":steps,
            "predict_field":ploted_graph,
            'sensor_coords':sensor_coords,
            'loss_weight':loss_weight,
            'nodes':init_graph.coords,
            'cells':init_graph.cell,
            'lr':lr,
            'start_with_0':start_with_0}
    return mse[-1],dict,sensor_coords,mse
                

def test_inverse(start_index=0, dataset_size=1, prior=None,gnn_solver=None,graph_update_fn=None,mask_type=None,time_steps=None,
                 sensor_num=None,num_iter=1000,lr=1e-2,lr_decay=0.9,lr_decay_steps=500,log_path=None,store=False,
                 data_file=None,start_with_0=False,edge_features=None,prior_type=None,resolution=128,noprior=False,loss_fn=None,progressive=False,
                 convergence_stop=False,path_to_data=None,obversation_step=1,start_observation_index=1,gradient_clip=0,
                 lr_decay_type = "per_iter",reg=0,device="cuda",repeat_time=0):
    
    mse_vs_time = []
    seed = 1
    for count_sample,index in enumerate(range(start_index,start_index+dataset_size)):
        for idx,t in enumerate(time_steps):
            seed_everything(seed)
            loss_value, dict,sensor_coords, mse = solver_invserproblem(prior,gnn_solver,graph_update_fn,t,mask_percent=sensor_num,lr=lr,num_iter=num_iter,mask_type=mask_type,
                                                                    seed=seed,index=index,lr_decay=lr_decay,lr_decay_steps=lr_decay_steps,store=store,
                                                                    log_path=log_path,data_file=data_file,start_with_0=start_with_0,progressive=progressive,
                                                                    edge_features=edge_features,prior_type=prior_type,resolution=resolution,noprior=noprior,
                                                                    loss_fn=loss_fn,convergence_stop=convergence_stop,path_to_data=path_to_data,obversation_step=obversation_step,
                                                                    start_observation_index = start_observation_index,gradient_clip=gradient_clip,lr_decay_type=lr_decay_type,
                                                                    reg=reg,device=device,repeat_time=repeat_time)            
            mse_vs_time.append({'mse': mse, "time_steps": time_steps})
            
            if store:
                hf = h5py.File('{}/losses_{}_{}_{}.h5'.format(log_path,sensor_num,t,noprior), 'a')
                g1 = hf.create_group('{}'.format(index))
                g1.create_dataset("loss", data=loss_value)       
                hf.close()
                np.save('{}/individual_npy/dict_{}.npy'.format(log_path,index),dict)
                f= open("{}/summary.txt".format(log_path),"a+")
                content = str([index,loss_value])
                f.write(content)
                f.write(" \n")

def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    