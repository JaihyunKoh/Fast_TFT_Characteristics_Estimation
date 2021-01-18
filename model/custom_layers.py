import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.signal import gaussian


# From DAVANet, for Context module  #
def conv(in_channels, out_channels, kernel_size=3, stride=1,dilation=1, bias=True):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, dilation=dilation, padding=((kernel_size-1)//2)*dilation, bias=bias),
        nn.LeakyReLU(0.2,  inplace=True)
    )

def ms_dilate_block(in_channels, kernel_size=3, dilation=[1,1,1,1], bias=True):
    return MSDilateBlock(in_channels, kernel_size, dilation, bias)

def cat_with_crop(target, input):
    output = []
    for item in input:
        if item.size()[2:] == target.size()[2:]:
            output.append(item)
        else:
            output.append(item[:, :, :target.size(2), :target.size(3)])
    output = torch.cat(output,1)
    return output

class MSDilateBlock(nn.Module):
    def __init__(self, in_channels, kernel_size, dilation, bias):
        super(MSDilateBlock, self).__init__()
        self.conv1 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[0], bias=bias)
        self.conv2 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[1], bias=bias)
        self.conv3 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[2], bias=bias)
        self.conv4 =  conv(in_channels, in_channels, kernel_size,dilation=dilation[3], bias=bias)
        self.convi =  nn.Conv2d(in_channels*4, in_channels, kernel_size=kernel_size, stride=1, padding=(kernel_size-1)//2, bias=bias)
    def forward(self, x):
        conv1 = self.conv1(x)
        conv2 = self.conv2(x)
        conv3 = self.conv3(x)
        conv4 = self.conv4(x)
        cat  = torch.cat([conv1, conv2, conv3, conv4], 1)
        out = self.convi(cat) + x
        return out


# Warping Layer #
def get_grid(x):
    torchHorizontal = torch.linspace(-1.0, 1.0, x.size(3)).view(1, 1, 1, x.size(3)).expand(x.size(0), 1, x.size(2), x.size(3))
    torchVertical = torch.linspace(-1.0, 1.0, x.size(2)).view(1, 1, x.size(2), 1).expand(x.size(0), 1, x.size(2), x.size(3))
    grid = torch.cat([torchHorizontal, torchVertical], 1)
    return grid

class WarpingLayer(nn.Module):
    def __init__(self):
        super(WarpingLayer, self).__init__()

    def forward(self, x, flow):
        # WarpingLayer uses F.grid_sample, which expects normalized grid
        # we still output unnormalized flow for the convenience of comparing EPEs with FlowNet2 and original code
        # so here we need to denormalize the flow
        flow_for_grip = torch.zeros_like(flow)
        flow_for_grip[:, 0, :, :] = (flow[:, 0, :, :] / ((flow.size(3) - 1.0) / 2.0))
        flow_for_grip[:, 1, :, :] = (flow[:, 1, :, :] / ((flow.size(2) - 1.0) / 2.0))

        # grid = (get_grid(x).to(args.device) + flow_for_grip).permute(0, 2, 3, 1)
        grid = (get_grid(x).cuda() + flow_for_grip).permute(0, 2, 3, 1)
        x_warp = F.grid_sample(x, grid)
        return x_warp


# Edge extraction layer
class CannyEdge(nn.Module):
    def __init__(self, threshold= 0.0, use_cuda=False):
        super(CannyEdge, self).__init__()

        self.threshold = threshold
        self.use_cuda = use_cuda

        filter_size = 5
        generated_filters = gaussian(filter_size, std=1.0).reshape([1,filter_size])
        generated_filters = generated_filters/np.sum(generated_filters)

        self.gaussian_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1,filter_size), padding=(0,filter_size//2))
        self.gaussian_filter_horizontal.weight.data.copy_(torch.from_numpy(generated_filters))

        self.gaussian_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.gaussian_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(filter_size,1), padding=(filter_size//2,0))
        self.gaussian_filter_vertical.weight.data.copy_(torch.from_numpy(generated_filters.T))
        self.gaussian_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        sobel_filter = np.array([[1, 0, -1],
                                 [2, 0, -2],
                                 [1, 0, -1]])/8

        self.sobel_filter_horizontal = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_horizontal.weight.data.copy_(torch.from_numpy(sobel_filter))
        self.sobel_filter_horizontal.bias.data.copy_(torch.from_numpy(np.array([0.0])))
        self.sobel_filter_vertical = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=sobel_filter.shape, padding=sobel_filter.shape[0]//2)
        self.sobel_filter_vertical.weight.data.copy_(torch.from_numpy(sobel_filter.T))
        self.sobel_filter_vertical.bias.data.copy_(torch.from_numpy(np.array([0.0])))

        # filters were flipped manually
        filter_0 = np.array([   [ 0, 0, 0],
                                [ 0, 1, -1],
                                [ 0, 0, 0]])

        filter_45 = np.array([  [0, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, -1]])

        filter_90 = np.array([  [ 0, 0, 0],
                                [ 0, 1, 0],
                                [ 0,-1, 0]])

        filter_135 = np.array([ [ 0, 0, 0],
                                [ 0, 1, 0],
                                [-1, 0, 0]])

        filter_180 = np.array([ [ 0, 0, 0],
                                [-1, 1, 0],
                                [ 0, 0, 0]])

        filter_225 = np.array([ [-1, 0, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_270 = np.array([ [ 0,-1, 0],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        filter_315 = np.array([ [ 0, 0, -1],
                                [ 0, 1, 0],
                                [ 0, 0, 0]])

        all_filters = np.stack([filter_0, filter_45, filter_90, filter_135, filter_180, filter_225, filter_270, filter_315])

        self.directional_filter = nn.Conv2d(in_channels=1, out_channels=8, kernel_size=filter_0.shape, padding=filter_0.shape[-1] // 2)
        self.directional_filter.weight.data.copy_(torch.from_numpy(all_filters[:, None, ...]))
        self.directional_filter.bias.data.copy_(torch.from_numpy(np.zeros(shape=(all_filters.shape[0],))))

    def forward(self, img):
        img_r = torch.div(torch.add(img[:,0:1], 1.0), 2.0)
        img_g = torch.div(torch.add(img[:,1:2], 1.0), 2.0)
        img_b = torch.div(torch.add(img[:,2:3], 1.0), 2.0)

        blur_horizontal = self.gaussian_filter_horizontal(img_r)
        blurred_img_r = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_g)
        blurred_img_g = self.gaussian_filter_vertical(blur_horizontal)
        blur_horizontal = self.gaussian_filter_horizontal(img_b)
        blurred_img_b = self.gaussian_filter_vertical(blur_horizontal)

        blurred_img = torch.cat([blurred_img_r,blurred_img_g,blurred_img_b], dim=1)
        # print(blurred_img.shape)
        # blurred_img = torch.stack([torch.squeeze(blurred_img)])

        grad_x_r = self.sobel_filter_horizontal(blurred_img_r)
        grad_y_r = self.sobel_filter_vertical(blurred_img_r)
        grad_x_g = self.sobel_filter_horizontal(blurred_img_g)
        grad_y_g = self.sobel_filter_vertical(blurred_img_g)
        grad_x_b = self.sobel_filter_horizontal(blurred_img_b)
        grad_y_b = self.sobel_filter_vertical(blurred_img_b)

        # COMPUTE THICK EDGES

        grad_mag = torch.sqrt(grad_x_r**2 + grad_y_r**2)
        grad_mag += torch.sqrt(grad_x_g**2 + grad_y_g**2)
        grad_mag += torch.sqrt(grad_x_b**2 + grad_y_b**2)
        grad_orientation = (torch.atan2(grad_y_r+grad_y_g+grad_y_b, grad_x_r+grad_x_g+grad_x_b) * (180.0/3.14159))
        grad_orientation += 180.0
        grad_orientation =  torch.round( grad_orientation / 45.0 ) * 45.0

        # THIN EDGES (NON-MAX SUPPRESSION)

        all_filtered = self.directional_filter(grad_mag)

        inidices_positive = (grad_orientation / 45) % 8
        inidices_negative = ((grad_orientation / 45) + 4) % 8

        height = inidices_positive.size()[2]
        width = inidices_positive.size()[3]
        pixel_count = height * width
        pixel_range = torch.FloatTensor([range(pixel_count)])
        if self.use_cuda:
            pixel_range = torch.cuda.FloatTensor([range(pixel_count)])

        indices = (inidices_positive.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_positive = all_filtered.view(-1)[indices.long()].view(1,height,width)

        indices = (inidices_negative.view(-1).data * pixel_count + pixel_range).squeeze()
        channel_select_filtered_negative = all_filtered.view(-1)[indices.long()].view(1,height,width)

        channel_select_filtered = torch.stack([channel_select_filtered_positive,channel_select_filtered_negative])

        is_max = channel_select_filtered.min(dim=0)[0] > 0.0
        is_max = torch.unsqueeze(is_max, dim=0)

        thin_edges = grad_mag.clone()
        thin_edges[is_max==0] = 0.0

        # THRESHOLD

        thresholded = thin_edges.clone()
        thresholded[thin_edges<self.threshold] = 0.0

        early_threshold = grad_mag.clone()
        early_threshold[grad_mag<self.threshold] = 0.0

        assert grad_mag.size() == grad_orientation.size() == thin_edges.size() == thresholded.size() == early_threshold.size()

        return blurred_img, grad_mag, grad_orientation, thin_edges, thresholded, early_threshold

class WidenAttention(nn.Module):
    def __init__(self, threshold):
        super(WidenAttention, self).__init__()

        self.threshold = threshold
        self.use_cuda = True

        widen_filter = np.array([[1, 1, 1],
                                 [1, 1, 1],
                                 [1, 1, 1]])

        self.widen_filter = nn.Conv2d(in_channels=1, out_channels=1, kernel_size=widen_filter.shape, padding=widen_filter.shape[0] // 2)
        self.widen_filter.weight.data.copy_(torch.from_numpy(widen_filter))
        self.widen_filter.bias.data.copy_(torch.from_numpy(np.array([0.0])))

    def forward(self, attention):
        widen = self.widen_filter(attention)
        # thresholding = widen.clone()
        # thresholding[widen > self.threshold] = 1.0
        thresholding = torch.clamp(widen, 0.0, 1.0)
        return thresholding


class AdaptiveInstanceNorm(nn.Module):
    def __init__(self):
        super(AdaptiveInstanceNorm, self).__init__()

        self.gradInput = None
        self.eps = 1e-5
        self.num_features = 128  # channel * batch

        self.register_buffer('running_mean', torch.zeros(self.num_features))
        self.register_buffer('running_var', torch.ones(self.num_features))

    def forward(self, content, style):
        """
        Implement the AdaIN layer. The layer applies batch normalization on the content input using the standard
        deviation and mean computed using the style input.
        :param input: a PyTorch tensor of 2x3x128x128 consisting of the content and the style.
        :return: the output of ? as the batch normalized content.
        """

        # content, style = input[0], input[1]
        # hc, wc = content.size()[1], content.size()[2]
        # hs, ws = style.size()[1], style.size()[2]

        batch, channel, height, width = content.size(0), content.size(1), content.size(2), content.size(3)
        running_mean = self.running_mean.type_as(content)    # [self.num_features]
        running_var = self.running_var.type_as(content)                               #[-1, 4096]
        content_reshaped = content.contiguous().view(1, batch*channel, height*width)

        weight = torch.cat([style[0, :128]], 0).view(-1)
        bias = torch.cat([style[0, 128:]], 0).view(-1)

        output = F.batch_norm(content_reshaped, running_mean, running_var, weight, bias, True, eps=self.eps)
        output = output.view(batch, channel, height, width)
        return output

class GaussianNoise(nn.Module):
    """Gaussian noise regularizer.

    Args:
        sigma (float, optional): relative standard deviation used to generate the
            noise. Relative means that it will be multiplied by the magnitude of
            the value your are adding the noise to. This means that sigma can be
            the same regardless of the scale of the vector.
        is_relative_detach (bool, optional): whether to detach the variable before
            computing the scale of the noise. If `False` then the scale of the noise
            won't be seen as a constant but something to optimize: this will bias the
            network to generate vectors with smaller values.
    """

    def __init__(self, sigma=0.1, is_relative_detach=True):
        super().__init__()
        self.sigma = sigma
        self.is_relative_detach = is_relative_detach

    def forward(self, x):
        scale = self.sigma * x.detach() if self.is_relative_detach else self.sigma * x
        sampled_noise = torch.randn(*x.size()).cuda() * scale
        x = x + sampled_noise
        return torch.clamp(x, -1.0, 1.0)


