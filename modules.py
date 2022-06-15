import torch
import math
import numpy as np
from torch import nn
from torchmeta.modules import MetaModule
from collections import OrderedDict
import copy
import torch.nn.functional as F


def init_weights_requ(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_in')


def init_weights_normal(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, a=0.0, nonlinearity='relu', mode='fan_out')


def init_weights_selu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=1/math.sqrt(num_input))


def init_weights_elu(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            nn.init.normal_(m.weight, std=math.sqrt(1.5505188080679277)/math.sqrt(num_input))


def init_weights_xavier(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            nn.init.xavier_normal_(m.weight)


def init_weights_uniform(m):
    if type(m) == BatchLinear or type(m) == nn.Linear:
        if hasattr(m, 'weight'):
            torch.nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')


def sine_init(m, w0=60):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-np.sqrt(6/num_input)/w0, np.sqrt(6/num_input)/w0)


def first_layer_sine_init(m):
    with torch.no_grad():
        if hasattr(m, 'weight'):
            num_input = m.weight.size(-1)
            m.weight.uniform_(-1/num_input, 1/num_input)


class BatchLinear(nn.Linear, MetaModule):
    '''A linear meta-layer that can deal with batched weight matrices and biases, as for instance output by a
    hypernetwork.'''
    __doc__ = nn.Linear.__doc__

    def forward(self, input, params=None):
        if params is None:
            params = OrderedDict(self.named_parameters())

        bias = params.get('bias', None)
        weight = params['weight']

        output = input.matmul(weight.permute(*[i for i in range(len(weight.shape)-2)], -1, -2))
        output += bias.unsqueeze(-2)
        return output


class FirstSine(nn.Module):
    def __init__(self, w0=60):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0*input)


class Sine(nn.Module):
    def __init__(self, w0=60):
        super().__init__()
        self.w0 = torch.tensor(w0)

    def forward(self, input):
        return torch.sin(self.w0*input)


class ReQU(nn.Module):
    def __init__(self, inplace=True):
        super().__init__()
        self.relu = nn.ReLU(inplace)

    def forward(self, input):
        # return torch.sin(np.sqrt(256)*input)
        return .5*self.relu(input)**2


class MSoftplus(nn.Module):
    def __init__(self):
        super().__init__()
        self.softplus = nn.Softplus()
        self.cst = torch.log(torch.tensor(2.))

    def forward(self, input):
        return self.softplus(input)-self.cst


class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input):
        return input*torch.sigmoid(input)


def layer_factory(layer_type):
    layer_dict = \
        {
         'relu': (nn.ReLU(inplace=True), init_weights_normal),
         'requ': (ReQU(inplace=False), init_weights_requ),
         'sigmoid': (nn.Sigmoid(), None),
         'fsine': (Sine(), first_layer_sine_init),
         'sine': (Sine(), sine_init),
         'tanh': (nn.Tanh(), init_weights_xavier),
         'selu': (nn.SELU(inplace=True), init_weights_selu),
         'gelu': (nn.GELU(), init_weights_selu),
         'swish': (Swish(), init_weights_selu),
         'softplus': (nn.Softplus(), init_weights_normal),
         'msoftplus': (MSoftplus(), init_weights_normal),
         'elu': (nn.ELU(), init_weights_elu)
        }
    return layer_dict[layer_type]


class PositionalEncoding(nn.Module):
    def __init__(self, num_encoding_functions=6, include_input=True, log_sampling=True, normalize=False,
                 input_dim=2, gaussian_pe=False, gaussian_variance=0.1):
        super().__init__()
        self.num_encoding_functions = num_encoding_functions
        self.include_input = include_input
        self.log_sampling = log_sampling
        self.normalize = normalize
        self.gaussian_pe = gaussian_pe
        self.normalization = None

        if self.gaussian_pe:
            # this needs to be registered as a parameter so that it is saved in the model state dict
            # and so that it is converted using .cuda(). Doesn't need to be trained though
            self.gaussian_weights = nn.Parameter(2*np.pi*gaussian_variance * torch.randn((num_encoding_functions*2), input_dim),
                                                 requires_grad=False)

        else:
            self.frequency_bands = None
            if self.log_sampling:
                self.frequency_bands = 2.0 ** torch.linspace(
                    0.0,
                    self.num_encoding_functions - 1,
                    self.num_encoding_functions)
            else:
                self.frequency_bands = torch.linspace(
                    2.0 ** 0.0,
                    2.0 ** (self.num_encoding_functions - 1),
                    self.num_encoding_functions)

            if normalize:
                self.normalization = torch.tensor(1/self.frequency_bands)

    def forward(self, tensor) -> torch.Tensor:
        r"""Apply positional encoding to the input.
        Args:
            tensor (torch.Tensor): Input tensor to be positionally encoded.
            encoding_size (optional, int): Number of encoding functions used to compute
                a positional encoding (default: 6).
            include_input (optional, bool): Whether or not to include the input in the
                positional encoding (default: True).
        Returns:
        (torch.Tensor): Positional encoding of the input tensor.
        """

        encoding = [tensor] if self.include_input else []
        if self.gaussian_pe:
            for func in [torch.sin, torch.cos]:
                encoding.append(func(torch.matmul(tensor, self.gaussian_weights.T)))
        else:
            for idx, freq in enumerate(self.frequency_bands):
                for func in [torch.sin, torch.cos]:
                    if self.normalization is not None:
                        encoding.append(self.normalization[idx]*func(tensor * freq))
                    else:
                        encoding.append(func(tensor * freq))

        # Special case, for no positional encoding
        if len(encoding) == 1:
            return encoding[0]
        else:
            return torch.cat(encoding, dim=-1)



class FCBlock(nn.Module):
    '''A fully connected neural network that also allows swapping out the weights when used with a hypernetwork.
    Can be used just as a normal neural network though, as well.
    '''
    def __init__(self, in_features, out_features,
                 num_hidden_layers, hidden_features,
                 outermost_linear=False, outmost_nonlinearity=None, nonlinearity='relu',
                 weight_init=None, w0=60, set_bias=None,
                 dropout=0.0, layer_norm=False,latent_dim=64,skip_connect=None):
        super().__init__()

        self.skip_connect = skip_connect
        self.latent_dim = latent_dim
        self.first_layer_init = None
        self.dropout = dropout

        if outmost_nonlinearity==None:
            outmost_nonlinearity = nonlinearity

        # Create hidden features list
        if not isinstance(hidden_features, list):
            num_hidden_features = hidden_features
            hidden_features = []
            for i in range(num_hidden_layers+1):
                hidden_features.append(num_hidden_features)
        else:
            num_hidden_layers = len(hidden_features)-1
        #print(f"net_size={hidden_features}")

        # Create the net
        #print(f"num_layers={len(hidden_features)}")
        if isinstance(nonlinearity, list):
            print(f"num_non_lin={len(nonlinearity)}")
            assert len(hidden_features) == len(nonlinearity), "Num hidden layers needs to " \
                                                              "match the length of the list of non-linearities"

            self.net = []
            self.net.append(nn.Sequential(
                nn.Linear(in_features, hidden_features[0]),
                layer_factory(nonlinearity[0])[0]
            ))
            for i in range(num_hidden_layers):
                if self.skip_connect==None:
                    self.net.append(nn.Sequential(
                        nn.Linear(hidden_features[i], hidden_features[i+1]),
                        layer_factory(nonlinearity[i+1])[0]
                    ))
                else:
                    if i+1 in self.skip_connect:
                        self.net.append(nn.Sequential(
                        nn.Linear(hidden_features[i]+self.latent_dim, hidden_features[i+1]),
                        layer_factory(nonlinearity[i+1])[0]
                    ))
                    else:
                        self.net.append(nn.Sequential(
                            nn.Linear(hidden_features[i], hidden_features[i+1]),
                            layer_factory(nonlinearity[i+1])[0]
                        ))

            if outermost_linear:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                    layer_factory(nonlinearity[-1])[0]
                ))
        elif isinstance(nonlinearity, str):
            nl, weight_init = layer_factory(nonlinearity)
            outmost_nl, _ = layer_factory(outmost_nonlinearity)
            if(nonlinearity == 'sine'):
                first_nl = FirstSine()
                self.first_layer_init = first_layer_sine_init
            else:
                first_nl = nl

            if weight_init is not None:
                self.weight_init = weight_init

            self.net = []
            self.net.append(nn.Sequential(
                nn.Linear(in_features, hidden_features[0]),
                first_nl
            ))

            for i in range(num_hidden_layers):
                if(self.dropout > 0):
                    self.net.append(nn.Dropout(self.dropout))
                if self.skip_connect == None:
                    self.net.append(nn.Sequential(
                        nn.Linear(hidden_features[i], hidden_features[i+1]),
                        copy.deepcopy(nl)
                    ))
                else:
                    if i+1 in self.skip_connect:
                        self.net.append(nn.Sequential(
                        nn.Linear(hidden_features[i]+self.latent_dim, hidden_features[i+1]),
                        copy.deepcopy(nl)
                    ))
                    else:
                        self.net.append(nn.Sequential(
                            nn.Linear(hidden_features[i], hidden_features[i+1]),
                            copy.deepcopy(nl)
                        ))

            if (self.dropout > 0):
                self.net.append(nn.Dropout(self.dropout))
            if outermost_linear:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                ))
            else:
                self.net.append(nn.Sequential(
                    nn.Linear(hidden_features[-1], out_features),
                    copy.deepcopy(outmost_nl)
                ))
            if layer_norm:
                self.net.append(nn.LayerNorm([out_features]))

        self.net = nn.Sequential(*self.net)

        if isinstance(nonlinearity, list):
            for layer_num, layer_name in enumerate(nonlinearity):
                self.net[layer_num].apply(layer_factory(layer_name)[1])
        elif isinstance(nonlinearity, str):
            if self.weight_init is not None:
                self.net.apply(self.weight_init)

            if self.first_layer_init is not None:
                self.net[0].apply(self.first_layer_init)

        if set_bias is not None:
            self.net[-1][0].bias.data = set_bias * torch.ones_like(self.net[-1][0].bias.data)

    def forward(self, coords, batch_vecs=None):
        if self.skip_connect == None:
            output = self.net(coords)
        else:
            input = coords
            for i in range(len(self.net)):
                output = self.net[i](input)
                if i+1 in self.skip_connect:
                    input = torch.cat([batch_vecs, output], dim=-1)
                else:
                    input = output
        return output


class CoordinateNet_autodecoder(nn.Module):
    '''A autodecoder network'''
    def __init__(self, latent_size=64, out_features=1, nl='sine', in_features=64+2,
                 hidden_features=256, num_hidden_layers=3, num_pe_fns=6,
                 w0=60,use_pe=False,skip_connect=None,dataset_size=100,
                 outmost_nonlinearity=None,outermost_linear=True):
        super().__init__()

        self.nl = nl
        self.use_pe = use_pe
        self.latent_size = latent_size
        self.lat_vecs = torch.nn.Embedding(dataset_size, self.latent_size)
        torch.nn.init.normal_(self.lat_vecs.weight.data, 0.0, 1/ math.sqrt(self.latent_size))

        if self.nl != 'sine' and use_pe:
            in_features = 2 * (2*num_pe_fns + 1)+latent_size

        if self.use_pe:
            self.pe = PositionalEncoding(num_encoding_functions=num_pe_fns)
        self.decoder = FCBlock(in_features=in_features,
                           out_features=out_features,
                           num_hidden_layers=num_hidden_layers,
                           hidden_features=hidden_features,
                           outermost_linear=outermost_linear,
                           nonlinearity=nl,
                           w0=w0,skip_connect=skip_connect,latent_dim=latent_size,outmost_nonlinearity=outmost_nonlinearity)
        self.mean =  torch.mean(torch.mean(self.lat_vecs.weight.data.detach(), dim=1)).cuda()
        self.varience =  torch.mean(torch.var(self.lat_vecs.weight.data.detach(), dim=1)).cuda()
    

    def forward(self, model_input,latent=None):
        coords = model_input['coords'].clone().detach().requires_grad_(True)
        if latent==None:
            batch_vecs = self.lat_vecs(model_input['idx']).unsqueeze(1).repeat(1,coords.shape[1],1)
        else:
            batch_vecs = latent.unsqueeze(1).repeat(1,coords.shape[1],1)

        if self.nl != 'sine' and self.use_pe:
            coords_pe = self.pe(coords)
            input = torch.cat([batch_vecs, coords_pe], dim=-1)
            output = self.decoder(input,batch_vecs)
        else:
            input = torch.cat([batch_vecs, coords], dim=-1)
            output = self.decoder(input,batch_vecs)
     
        return {'model_in': coords, 'model_out': output,'batch_vecs': batch_vecs, 'meta': model_input}
