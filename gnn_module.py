from torch_geometric.nn import MessagePassing
import modules
import torch
import utils
from torch import nn

class processor(MessagePassing):
    def __init__(self, in_channels, out_channels,layer_norm=False,nl='relu'):
        super(processor, self).__init__(aggr='add')  # "Add" aggregation.
        self.edge_encoder = modules.FCBlock(in_features=in_channels*3, 
                               out_features=out_channels,
                               num_hidden_layers=2,
                               hidden_features=in_channels,
                               outermost_linear=True,
                               nonlinearity=nl,layer_norm=layer_norm)
        self.node_encoder = modules.FCBlock(in_features=in_channels*2,
                               out_features=out_channels,
                               num_hidden_layers=2,
                               hidden_features=in_channels,
                               outermost_linear=True,
                               nonlinearity=nl,layer_norm=layer_norm)
        self.latent_dim = out_channels

    def forward(self, graph):
        edge_index = graph.edge_index
        # cat features together (eij,vi,ei)
        x_receiver = torch.gather(graph.x,0,edge_index[0,:].unsqueeze(-1).repeat(1,graph.x.shape[1]))
        x_sender = torch.gather(graph.x,0,edge_index[1,:].unsqueeze(-1).repeat(1,graph.x.shape[1]))
        edge_features = torch.cat([x_receiver,x_sender,graph.edge_attr],dim=-1)
        # edge processor
        edge_features = self.edge_encoder(edge_features)
        
        # aggregate edge_features
        node_features = self.propagate(edge_index, x=graph.x, edge_attr=edge_features)
        # cat features for node processor (vi,\sum_eij)
        features = torch.cat([graph.x,node_features[:,self.latent_dim:]],dim=-1)
        # node processor and update graph
        graph.x = self.node_encoder(features) + graph.x
        graph.edge_attr = edge_features
        return graph

    def message(self, x_i, edge_attr):
        z = torch.cat([x_i,edge_attr],dim=-1)
        return z
    
class mesh_PDE(torch.nn.Module):
    def __init__(self, edge_dim, node_dim,latent_dim=32,num_steps=3,layer_norm=False,nl="relu",var=0,
                 batch_norm=False, normalize = False,encoder_nl='relu',diffMLP=False,checkpoints=None):
        super().__init__()
        self.encoder_edge = modules.FCBlock(in_features=edge_dim,
                               out_features=latent_dim,
                               num_hidden_layers=2,
                               hidden_features=latent_dim,
                               outermost_linear=True,
                               nonlinearity=nl,layer_norm=layer_norm)
        self.encoder_nodes = modules.FCBlock(in_features=node_dim,
                               out_features=latent_dim,
                               num_hidden_layers=2,
                               hidden_features=latent_dim,
                               outermost_linear=True,
                               nonlinearity=nl,layer_norm=layer_norm)
        
    
        self.num_steps = num_steps
        self.diffMLP = diffMLP
        if diffMLP:
            # message passing with different MLP for each steps
            self.processors = []
            for _ in range(num_steps):
                self.processors.append((processor(latent_dim, latent_dim,layer_norm=layer_norm)))
                if batch_norm:
                    self.processors.append(torch.nn.BatchNorm1d(latent_dim))
                    self.processors.append(torch.nn.BatchNorm1d(latent_dim))
            self.processors = torch.nn.Sequential(*self.processors)
        else:
            self.processors=((processor(latent_dim, latent_dim,layer_norm=layer_norm)))
        
       
        self.decoder_node = modules.FCBlock(in_features=latent_dim,
                                out_features=1,
                                num_hidden_layers=3,
                                hidden_features=latent_dim,
                                outermost_linear=True,
                                nonlinearity=encoder_nl,layer_norm=False)
       
        self.var = var

        self.batch_norm = batch_norm
        self.normalize = normalize
        if self.normalize:
            if checkpoints==None:
                self.normalizer_node_feature = normalizer(node_dim)
                self.normalizer_edge_feature = normalizer(edge_dim)
                self.normalizer_v_gt = normalizer(1)
            else:
                self.normalizer_node_feature = normalizer(node_dim,max_acc=0)
                self.normalizer_edge_feature = normalizer(edge_dim,max_acc=0)
                self.normalizer_v_gt = normalizer(1,max_acc=0)
        
            
    def encoder(self,graph):
        # add noise to input training sample
        noise = (torch.normal(0,1,size=(graph.x.shape[0],graph.x.shape[1]-1))*self.var).cuda()
        graph.x[:,:-1] = graph.x[:,:-1] + graph.x[:,[-1]]*noise
        graph.noise = graph.x[:,[-1]]*noise
        
        graph.x = self.encoder_nodes(graph.x)
        graph.edge_attr = self.encoder_edge(graph.edge_attr)
        return graph
    
    def decoder(self,graph):
        graph.x = self.decoder_node(graph.x)
        return graph
    
    def forward(self, graph, train=False):
        # normalize the dataset
        if self.normalize:
            graph.x = self.normalizer_node_feature.update(graph.x,train)
            graph.edge_attr = self.normalizer_edge_feature.update(graph.edge_attr,train)
            graph.v_gt = self.normalizer_v_gt.update(graph.v_gt,train)
        #ecode edges and nodes to latent dim
        graph_latent = self.encoder(graph.clone())
        
        if self.diffMLP:
            # message passing steps with different MLP each time
            for i in range(self.num_steps):
                if self.batch_norm:
                    graph_latent = self.processors[i*3](graph_latent)
                    graph_latent.x = self.processors[i*3+1](graph_latent.x)
                    graph_latent.edge_attr = self.processors[i*3+2](graph_latent.edge_attr)
                else:
                    graph_latent = self.processors[i](graph_latent)
        else:
            # message passing steps with same MLP each time
            for _ in range(self.num_steps):
                graph_latent = self.processors(graph_latent)
                    
                
        # decoding
        graph_latent = self.decoder(graph_latent)
        graph_latent.x = graph_latent.x/10 #div 10 for 46, div 100 for 31
        if self.normalize and not(train):
            graph_latent.x = self.normalizer_v_gt.reverse(graph_latent.x)
        graph_latent.eval = graph_latent.x
        return graph_latent
    

class normalizer(nn.Module):
    def __init__(self, dim, mean=0, std=1e-8, max_acc = 60*600):
        super().__init__()
        self.acc_sum = nn.Parameter(torch.zeros(dim).cuda(),requires_grad=False)
        self.acc_sum_squared = nn.Parameter(torch.zeros(dim).cuda(),requires_grad=False)
        self.mean = nn.Parameter(torch.zeros(dim).cuda(),requires_grad=False)
        self.std = nn.Parameter(torch.ones(dim).cuda(),requires_grad=False)
        
        self.total_acc = 0
        self.max_acc = max_acc
    
    def update(self,value,train):
        if self.total_acc<self.max_acc*value.shape[0] and train:
            self.total_acc += value.shape[0]
            self.acc_sum += torch.sum(value,0).data
            self.acc_sum_squared += torch.sum(value**2,0).data
            safe_count = max(1,self.total_acc)
            self.mean = nn.Parameter(self.acc_sum/safe_count)
            self.std = nn.Parameter(torch.maximum(torch.sqrt(self.acc_sum_squared / safe_count - self.mean**2),torch.tensor(1e-5).cuda()))
        return (value-self.mean.data)/self.std.data
    
    def reverse(self,value):
        return value*self.std.data+self.mean.data