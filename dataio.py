from torch_geometric.data import Data, DataLoader, Dataset
import torch
import numpy as np

class wave_data_2D_irrgular(Dataset):
    def __init__(self, num_trajectory=1,node_features=['u','v','density','type'],
                    edge_features = ['dist','direction'],file=None,train=True,endtime=-1,
                    step_size=1,index=0, device="cuda:0",var=0):
        super(wave_data_2D_irrgular, self).__init__()
        """ 
        Parameters
        ----------
        num_trajectory: number of trajectories
        node_features: input node features of GNN
        edge_features: input edge features of GNN
        file: dataset file folder 
        train: if training or validation 
        endtime: length for each trajectory
        step_size: gnn_stepsize = stepsize*class_solver_stepsize
        index: starting index for dataset


        Returns
        -------
        datalist: a list of constructed graphs with:
            graph.x - node features [n,f_nodes], f_nodes = size(node_features)
            graph.edge_list - list of edges [2, m], m - number of edges
            graph.edge_attr - list of edge features [m, f_edges] - f_edges = size(edge_features)
            graph.current_u - current field value
            graph.h - history field value
            graph.gt - ground truth field value
        """

        self.num_trajectory = num_trajectory
        if train:
            trajectory = np.load("{}/train.npy".format(file),allow_pickle=True)
        else:
            trajectory = np.load("{}/valid.npy".format(file),allow_pickle=True)
        self.endtime = endtime
        self.step_size = step_size
        self.trajectory_dataset = trajectory[index:num_trajectory+index]
        self.node_features = node_features
        self.edge_features = edge_features
        self.num_timesteps_pertraj = self.trajectory_dataset[0]['solution_low'][0:self.endtime:self.step_size].shape[0]-1        
        self.device = device
        self.var = var

    def len(self):
        return (self.num_timesteps_pertraj*self.num_trajectory)

    def get(self, idx):
        trajnum = int(np.floor(idx/self.num_timesteps_pertraj))
        traj = self.trajectory_dataset[trajnum]
        U_solution = torch.tensor(traj['solution_low'][0:self.endtime:self.step_size])
        _,num_nodes = U_solution.shape
        U_solution = U_solution.view(-1,num_nodes)
        U_eval = U_solution
        edges = traj['cell_low']
        a1 = np.concatenate([edges[:,[0]],edges[:,[1]]],axis=-1)
        a2 = np.concatenate([edges[:,[1]],edges[:,[0]]],axis=-1)
        a3 = np.concatenate([edges[:,[1]],edges[:,[2]]],axis=-1)
        a4 = np.concatenate([edges[:,[2]],edges[:,[1]]],axis=-1)
        a5 = np.concatenate([edges[:,[0]],edges[:,[2]]],axis=-1)
        a6 = np.concatenate([edges[:,[2]],edges[:,[0]]],axis=-1)
        edge_lists_with_duplication = np.concatenate([a1,a2,a3,a4,a5,a6],axis=0)
        edge_lists = [tuple(row) for row in edge_lists_with_duplication]
        edge_lists = np.array(np.unique(edge_lists, axis=0),dtype=np.int64)
        edge_lists = torch.tensor(edge_lists)
        edge_lists = edge_lists.permute(1,0)
        coords = np.array(traj['nodes_low'])
    
            
        edge_nodes = np.where((coords[:, 0] == 0) |
                     (coords[:, 0] == 1) |
                     (coords[:, 1] == 0) |
                     (coords[:, 1] == 1), True, False)[:, None]
        edge_nodes = torch.tensor(edge_nodes)*1
        node_type = torch.tensor(np.array(traj['is_boundary_low'])).reshape(-1,1)
        
        i = int(idx-trajnum*self.num_timesteps_pertraj)

        input_node_features = {}
        input_edge_features = {}
        input_edge_features['dist'] = torch.tensor(np.sqrt(np.sum((coords[edge_lists[1,:]]-coords[edge_lists[0,:]])**2,axis=-1))).unsqueeze(-1)
        input_edge_features['direction'] = torch.tensor((coords[edge_lists[0,:]]-coords[edge_lists[1,:]]))
        
        if 'density' in self.node_features:
            input_node_features['density'] = torch.tensor(traj['density_mesh']).view(-1,num_nodes).squeeze().unsqueeze(-1)
            input_node_features['density_eval'] = torch.tensor(traj['density_mesh']).view(-1,num_nodes).squeeze().unsqueeze(-1)

        input_node_features['coords'] = torch.tensor(coords)
        input_node_features['coords_eval'] = torch.tensor(coords)
        input_node_features['u'] =  U_solution[i,:].unsqueeze(-1)
        input_node_features['u_eval'] =  U_eval[i,:].unsqueeze(-1)
        input_node_features['type'] =  node_type
        input_node_features['type_eval'] =  node_type
        if i>0:
            input_node_features['u_h']  = U_solution[i-1,:].unsqueeze(-1)
            input_node_features['u_h_eval']  = U_eval[i-1,:].unsqueeze(-1)
        else:
            input_node_features['u_h']  = U_solution[i,:].unsqueeze(-1)
            input_node_features['u_h_eval']  = U_eval[i,:].unsqueeze(-1)
            
        #unroll now only support timestep=1,
        unroll_v_gt =  (U_solution[i+1:i+1+1,:]-U_solution[i:i+1,:]).permute(1,0)
        unroll_u_gt = (U_solution[i+1:i+1+1,:]).permute(1,0)
        input_node_features['u_gt'] = U_solution[i+1:i+1+1,:].permute(1,0)
        input_node_features['u_gt_eval'] = U_eval[i+1:i+1+1,:].permute(1,0)
        input_node_features['v'] = input_node_features['u']-input_node_features['u_h']
        input_node_features['v_eval'] = input_node_features['u_eval']-input_node_features['u_h_eval']
        input_node_features['v_gt'] = (U_solution[i+1:i+1+1,:]-U_solution[i:i+1,:]).permute(1,0)
        input_node_features['v_gt_eval'] = (U_eval[i+1:i+1+1,:]-U_eval[i:i+1,:]).permute(1,0)
        input_node_features['a_gt'] = input_node_features['v_gt'] - input_node_features['v'] 
        input_node_features['a_gt_eval'] = input_node_features['v_gt_eval'] - input_node_features['v_eval'] 

        x = input_node_features[self.node_features[0]].float().to(self.device)
        for feature in self.node_features[1:]:
            x = torch.cat([x,input_node_features[feature].to(self.device)],dim=-1).float()

        x_eval = input_node_features[self.node_features[0]+'_eval'].float().to(self.device)
        for feature in self.node_features[1:]:
            x_eval = torch.cat([x_eval,input_node_features[feature+'_eval'].to(self.device)],dim=-1).float()

        edge_attr = input_edge_features[self.edge_features[0]].float().to(self.device)
        for feature in self.edge_features[1:]:
            edge_attr = torch.cat([edge_attr,input_edge_features[feature].to(self.device)],dim=-1).float()

        data = Data(x=x, edge_index=edge_lists, edge_attr=edge_attr, current_u=input_node_features['u'],
                        h=input_node_features['u_h'],gt=input_node_features['u_gt_eval'],v_gt=input_node_features['v_gt_eval'], 
                        x_eval = x_eval,cell=edges,num_nodes=num_nodes,coords=coords,unroll_v_gt=unroll_v_gt,unroll_u_gt=unroll_u_gt,
                        a_gt=input_node_features['a_gt_eval'],fishcount=traj['fishcount'],var=self.var)
        return data

def file_dataloader(file, node_features=['u','v','density','type'],
                    edge_features = ['dist','direction'], step_size=5, endtime=-1,prefix="_low"):

    
    """
        this function takes a file and construct graph data from loaded data

        Parameters
        ----------
        file: path to data file
        node_features: input node features of GNN
        edge_features: input edge features of GNN
        step_size: gnn_stepsize = stepsize*class_solver_stepsize
        endtime: length for each trajectory


        Returns
        -------
        datalist: a list of constructed graphs with:
            graph.x - node features [n,f_nodes], f_nodes = size(node_features)
            graph.edge_list - list of edges [2, m], m - number of edges
            graph.edge_attr - list of edge features [m, f_edges] - f_edges = size(edge_features)
            graph.current_u - current field value
            graph.h - history field value
            graph.gt - ground truth field value
        """

    trajectory  = np.load(file,allow_pickle=True)

    trajectory_dataset = []
    trajectory_dataset.append(trajectory[0])

    datalist = []
    for idx,traj in enumerate(trajectory_dataset):
        if prefix=='high':
            U_solution = torch.tensor(traj['solution'][0:endtime:step_size])
        else:
            U_solution = torch.tensor(traj['solution{}'.format(prefix)][0:endtime:step_size])
        
        time_steps,num_nodes = U_solution.shape
        U_solution = U_solution.view(-1,num_nodes)
        U_eval = U_solution

        #construct adjacency matrix
        if prefix=='high':
            edges = traj['cell']
        else:
            edges = traj['cell{}'.format(prefix)]
        a1 = np.concatenate([edges[:,[0]],edges[:,[1]]],axis=-1)
        a2 = np.concatenate([edges[:,[1]],edges[:,[0]]],axis=-1)
        a3 = np.concatenate([edges[:,[1]],edges[:,[2]]],axis=-1)
        a4 = np.concatenate([edges[:,[2]],edges[:,[1]]],axis=-1)
        a5 = np.concatenate([edges[:,[0]],edges[:,[2]]],axis=-1)
        a6 = np.concatenate([edges[:,[2]],edges[:,[0]]],axis=-1)
        edge_lists_with_duplication = np.concatenate([a1,a2,a3,a4,a5,a6],axis=0)
        edge_lists = [tuple(row) for row in edge_lists_with_duplication]
        edge_lists = np.array(np.unique(edge_lists, axis=0),dtype=np.int64)
        edge_lists = torch.tensor(edge_lists)
        edge_lists = edge_lists.permute(1,0)
        
        if prefix=='high':
            coords = np.array(traj['nodes'])
        else:
            coords = np.array(traj['nodes{}'.format(prefix)])
            
        edge_nodes = np.where((coords[:, 0] == 0) |
                     (coords[:, 0] == 1) |
                     (coords[:, 1] == 0) |
                     (coords[:, 1] == 1), True, False)[:, None]
        edge_nodes = torch.tensor(edge_nodes)*1

        if prefix=='high':
            node_type = torch.tensor(np.array(traj['is_boundary'])).reshape(-1,1)
        else:
            node_type = torch.tensor(np.array(traj['is_boundary{}'.format(prefix)])).reshape(-1,1)

        inital_state_list = range(0,time_steps-1,1)
                
        for i in inital_state_list:
            input_node_features = {}
            input_edge_features = {}
            input_edge_features['dist'] = torch.tensor(np.sqrt(np.sum((coords[edge_lists[1,:]]-coords[edge_lists[0,:]])**2,axis=-1))).unsqueeze(-1)
            input_edge_features['direction'] = torch.tensor((coords[edge_lists[0,:]]-coords[edge_lists[1,:]]))
            
            if 'density' in node_features:
                if 'density_low' in traj:
                    input_node_features['density'] = torch.tensor(traj['density_low']).view(-1,num_nodes).squeeze().unsqueeze(-1)
                    input_node_features['density_eval'] = torch.tensor(traj['density_low']).view(-1,num_nodes).squeeze().unsqueeze(-1)
                elif 'density_mesh' in traj:
                    input_node_features['density'] = torch.tensor(traj['density_mesh']).view(-1,num_nodes).squeeze().unsqueeze(-1)
                    input_node_features['density_eval'] = torch.tensor(traj['density_mesh']).view(-1,num_nodes).squeeze().unsqueeze(-1)

            input_node_features['coords'] = torch.tensor(coords)
            input_node_features['coords_eval'] = torch.tensor(coords)
            input_node_features['u'] =  U_solution[i,:].unsqueeze(-1)
            input_node_features['u_eval'] =  U_eval[i,:].unsqueeze(-1)
            input_node_features['type'] =  node_type
            input_node_features['type_eval'] =  node_type
            if i>0:
                input_node_features['u_h']  = U_solution[i-1,:].unsqueeze(-1)
                input_node_features['u_h_eval']  = U_eval[i-1,:].unsqueeze(-1)
            else:
                input_node_features['u_h']  = U_solution[i,:].unsqueeze(-1)
                input_node_features['u_h_eval']  = U_eval[i,:].unsqueeze(-1)
                
            #unroll now only support 1=1, ouputsteps=1
            unroll_v_gt =  (U_solution[i+1:i+1+1,:]-U_solution[i:i+1,:]).permute(1,0)
            unroll_u_gt = (U_solution[i+1:i+1+1,:]).permute(1,0)
            input_node_features['u_gt'] = U_solution[i+1:i+1+1,:].permute(1,0)
            input_node_features['u_gt_eval'] = U_eval[i+1:i+1+1,:].permute(1,0)
            input_node_features['v'] = input_node_features['u']-input_node_features['u_h']
            input_node_features['v_eval'] = input_node_features['u_eval']-input_node_features['u_h_eval']
            input_node_features['v_gt'] = (U_solution[i+1:i+1+1,:]-U_solution[i:i+1,:]).permute(1,0)
            input_node_features['v_gt_eval'] = (U_eval[i+1:i+1+1,:]-U_eval[i:i+1,:]).permute(1,0)
            input_node_features['a_gt'] = input_node_features['v_gt'] - input_node_features['v'] 
            input_node_features['a_gt_eval'] = input_node_features['v_gt_eval'] - input_node_features['v_eval'] 
    
            x = input_node_features[node_features[0]].float()
            for feature in node_features[1:]:
                x = torch.cat([x,input_node_features[feature].float()],dim=-1)

            x_eval = input_node_features[node_features[0]+'_eval'].float()
            for feature in node_features[1:]:
                x_eval = torch.cat([x_eval,input_node_features[feature+'_eval'].float()],dim=-1)

            edge_attr = input_edge_features[edge_features[0]].float()
            for feature in edge_features[1:]:
                edge_attr = torch.cat([edge_attr,input_edge_features[feature].float()],dim=-1)
            
            traj['fishcount'] = None
            data = Data(x=x, edge_index=edge_lists, edge_attr=edge_attr, current_u=input_node_features['u'],
                            h=input_node_features['u_h'],gt=input_node_features['u_gt_eval'],v_gt=input_node_features['v_gt_eval'], 
                            noise=None, eval=None,x_eval = x_eval,cell=edges,coords=coords)
            datalist.append(data)  
    return datalist



def wave_data_update(node_features, output_graph, old_graph, output_type='v', train=True, keep_grad=False):
    """ 
    Parameters
    ----------
    node_features: input node features, i.e. ['u','v','type']
    output_graph: graph output from model
    old_graph: input graph to the model
    output_type: physics quantiy of output_graph.x

    Returns
    -------
    new_graph: updated graph with 
        new_graph.x[:,0] - amplitude
        new_graph.x[:,1] - velocity
    that satisfy boundary condition
    """

    if not keep_grad:
        output_graph.x = output_graph.x.detach()
        old_graph.x = old_graph.x.detach()

    new_graph = old_graph.clone()
    node_feature_list = {}
    if output_type=="x":
        node_feature_list['u'] =  output_graph.x[:,-1]
        node_feature_list['u_eval'] =  output_graph.eval[:,-1]
        node_feature_list['v'] = output_graph.x[:,-1] - old_graph.x[:,0]
        node_feature_list['v_eval'] =  output_graph.eval[:,-1] - old_graph.x_eval[:,0]

    elif output_type=="v":
        node_feature_list['u'] =  output_graph.x[:,-1] + old_graph.x[:,0] # v*1+x
        node_feature_list['u_eval'] =  output_graph.eval[:,-1] + old_graph.x_eval[:,0] # v*1+x
        node_feature_list['v'] =  output_graph.x[:,-1] #v
        node_feature_list['v_eval'] =  output_graph.eval[:,-1] #v
    
    elif output_type=="a":
        node_feature_list['u'] =  output_graph.x[:,-1] + old_graph.x[:,1] + old_graph.x[:,0] # a*1+v+x
        node_feature_list['u_eval'] =  output_graph.eval[:,-1] + old_graph.x_eval[:,1] + old_graph.x_eval[:,0] # a*1+v+x
        node_feature_list['v'] =  output_graph.x[:,-1] + old_graph.x[:,1] #a+v
        node_feature_list['v_eval'] =  output_graph.eval[:,-1] + old_graph.x[:,1]#a+v

    if 'density' in node_features:
        node_feature_list['density'] = old_graph.x[:,2]
        node_feature_list['density_eval'] = old_graph.x_eval[:,2]
        
    for (idx,feature) in enumerate(node_features[:-1]):
        new_graph.x[:,idx] = node_feature_list[feature]
        
    for (idx,feature) in enumerate(node_features[:-1]):
        new_graph.x_eval[:,idx] = node_feature_list[feature+'_eval']
    
    new_graph.h[:,0] = old_graph.x_eval[:,0]
    new_graph.current_u[:,0] = node_feature_list['u']
    # set bdd point x, v to be 0, node type is always at last column
    if train:
        new_graph.x[:,:-1]= -1*(new_graph.x[:,[-1]]-1)*new_graph.x[:,:-1]

    return new_graph

def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.'''
    if isinstance(sidelen, int):
        sidelen = dim * (sidelen,)

    if dim == 2:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[0, :, :, 0] = pixel_coords[0, :, :, 0] / (sidelen[0] - 1)
        pixel_coords[0, :, :, 1] = pixel_coords[0, :, :, 1] / (sidelen[1] - 1)
    elif dim == 3:
        pixel_coords = np.stack(np.mgrid[:sidelen[0], :sidelen[1], :sidelen[2]], axis=-1)[None, ...].astype(np.float32)
        pixel_coords[..., 0] = pixel_coords[..., 0] / max(sidelen[0] - 1, 1)
        pixel_coords[..., 1] = pixel_coords[..., 1] / (sidelen[1] - 1)
        pixel_coords[..., 2] = pixel_coords[..., 2] / (sidelen[2] - 1)
    else:
        raise NotImplementedError('Not implemented for dim=%d' % dim)

    pixel_coords -= 0.5
    pixel_coords *= 2.
    pixel_coords = torch.Tensor(pixel_coords).view(-1, dim)
    return pixel_coords

def lin2img(tensor, image_resolution=None):
    batch_size, num_samples, channels = tensor.shape
    if image_resolution is None:
        width = np.sqrt(num_samples).astype(int)
        height = width
    else:
        height = image_resolution[0]
        width = image_resolution[1]

    return tensor.permute(0, 2, 1).view(batch_size, channels, height, width)


class density(torch.utils.data.Dataset):
    def __init__(self, sidelength=300,dataset_size=1000,sampled_points=900, jitter=False, type="density",wandb=None):
        
        if type=="density":
            self.density  = torch.tensor(np.load('./data/prior/density_10000_{}_5_10.npy'.format(int(sidelength/2)),allow_pickle=True))
            self.density = self.density[0:dataset_size,:,:]
            self.density = (self.density-1)/2 #normalzie to be between 0 and 1
        elif type=="init_state":
            self.density  = torch.tensor(np.load('./data/prior/initial_states_10000_{}_6_10.npy'.format(int(sidelength/2)),allow_pickle=True))
            self.density = self.density[0:dataset_size,:,:]
            self.offset = self.density.min()
            self.density = self.density-self.offset
            self.rescale = self.density.max()
            self.density = (self.density)/self.rescale
            print(self.offset,self.rescale)

        self.mgrid = get_mgrid([sidelength,sidelength])
        self.sidelength = sidelength
        self.sampled_points = sampled_points
        self.jitter = jitter
    def __len__(self):
        return self.density.shape[0]

    def __getitem__(self, idx):
        img = self.density[[idx],:,:]
        img = img.permute(1, 2, 0).view(-1, 1)
        
        if self.jitter:
            coords_jitter = torch.randn(self.mgrid.shape)*(1/self.sidelength/10)
            mgrid_jitter = coords_jitter+self.mgrid

        mask = torch.randperm(self.sidelength**2)[0:self.sampled_points]
        coords = mgrid_jitter[mask]
        img_coarse = img[mask]
        
        in_dict = {'idx': idx, 'coords': coords, 'coords_fine': self.mgrid}
        gt_dict = {'img': img_coarse, 'img_fine': img, 'mask':mask}

        return in_dict, gt_dict


