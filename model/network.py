import torch.nn as nn
import functools
import torch
import torch.nn.functional as F
from torch.nn.utils import weight_norm


def define_fast_vth_block(num_seq, num_inputs, num_channels, kernel_size, dropout):
    net = FastVthSearchBlock(num_seq, num_inputs, num_channels, kernel_size, dropout)
    # net.apply(weights_init)
    return net


###############################################################################
# Define function
###############################################################################
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('ConvTranspose2d') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0)
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


def print_network(net):
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print('Total number of parameters: %d' % num_params)


def get_module_names(model):
    names = []
    for key, val in model.state_dict().items():
        name = key.split('.')[0]
        if not name in names:
            names.append(name)
    return names

###############################################################################
# Basic TCN layers
###############################################################################
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv2d(n_inputs, n_outputs, (1, kernel_size), stride=stride, padding=0, dilation=dilation)
        self.pad = torch.nn.ZeroPad2d((padding, 0, 0, 0))
        self.relu = nn.ReLU()
        # self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(n_outputs, n_outputs, (1, kernel_size), stride=stride, padding=0, dilation=dilation)
        self.net = nn.Sequential(self.pad, self.conv1, self.relu, self.pad, self.conv2, self.relu)
        self.downsample = nn.Conv2d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

##############################################################################
# Model
##############################################################################
class FastVthSearchBlock(nn.Module):
    def __init__(self, num_seq, num_inputs, num_channels, kernel_size, dropout):
        super(FastVthSearchBlock, self).__init__()
        self.tcn = TemporalConvNet(num_inputs, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.num_inputs = num_inputs
        # self.dropout = nn.Dropout(dropout)
        self.flatten = Flatten()
        self.regressor = nn.Linear(num_channels[-1]*num_seq, 1)
    def forward(self, x):
        if self.num_inputs == 1:
            x = x.unsqueeze(1).unsqueeze(1) # [bs, vec] to [bs, 1, 1, vec] for conv2d,
        else:
            x = x.unsqueeze(2) #  [bs, ch, vec] to [bs, ch, 1, vec]
        x = self.tcn(x)
        x = self.flatten(x)
        out = self.regressor(x)
        return out