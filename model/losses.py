import torch.nn as nn
import torch
import torchvision.models as models
from torch.nn import init
import functools
import torch.autograd as autograd
import numpy as np
from torch.autograd import Variable

###############################################################################
# Functions
###############################################################################

def init_loss(opt):
    content_loss = ContentLoss()
    content_loss.initialize(nn.L1Loss())

    # perceptual_loss = PerceptualLoss()
    # perceptual_loss.initialize(nn.MSELoss())

    # disc_loss = DiscLossWGANGP()
    # disc_loss.initialize()

    # disc_loss = DiscLoss()
    # disc_loss.initialize()


    # tv_loss.initialize()
    return content_loss

class TVLoss():
    """
    Total variation loss.
    """
    def initialize(self):
        pass
    def get_loss(self, flow):
        bsize, chan, height, width = flow.size()
        tvhs = []
        tvws = []
        for h in range(height-1):
            dy = torch.abs(flow[:,:,h+1,:] - flow[:,:,h,:])
            tvh = torch.norm(dy, 1)
            tvhs.append(tvh)
        for w in range(width-1):
            dx = torch.abs(flow[:,:,:,w+1] - flow[:,:,:,w])
            tvw = torch.norm(dx, 1)
            tvws.append(tvw)
        return sum(tvhs + tvws) / (height + width)


class ClassificationLoss():
    def initialize(self, loss):
        self.criterion = loss
    def get_loss(self, label_h, label_y):
        return self.criterion(label_h, label_y)


class ContentLoss():
    def initialize(self, loss):
        self.criterion = loss
    def get_loss(self, fakeIm, realIm):
        return self.criterion(fakeIm, realIm)


class PerceptualLoss():
    def contentFunc(self):
        conv_3_3_layer = 14
        cnn = models.vgg19(pretrained=True).features
        cnn = cnn.cuda()
        model = nn.Sequential()
        model = model.cuda()
        for i, layer in enumerate(list(cnn)):
            model.add_module(str(i), layer)
            if i == conv_3_3_layer:
                break
        return model

    def initialize(self, loss):
        self.criterion = loss
        self.contentFunc = self.contentFunc()

    def get_loss(self, fakeIm, realIm):
        f_fake = self.contentFunc.forward(fakeIm)
        f_real = self.contentFunc.forward(realIm)
        f_real_no_grad = f_real.detach()
        loss = self.criterion(f_fake, f_real_no_grad)
        return loss


class GANLoss(nn.Module):
    def __init__(self, use_l1=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor
        if use_l1:
            self.loss = nn.L1Loss()
        else:
            self.loss = nn.BCELoss()
            # self.loss = nn.BCEWithLogitsLoss()

    def get_target_tensor(self, input, target_is_real):
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)
                self.real_label_var = Variable(real_tensor, requires_grad=False)
            target_tensor = self.real_label_var
        else:
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real).cuda()
        return self.loss(input, target_tensor)

class DiscLoss():
    def name(self):
        return 'DiscLoss'

    def initialize(self):
        self.criterionGAN = GANLoss(use_l1=False)

    def get_g_loss(self, net, fake):
        # First, G(A) should fake the discriminator
        pred_fake = net.forward(fake)
        return self.criterionGAN(pred_fake, 1)

    def get_loss(self, net, real, fake):
        # Fake
        # stop backprop to the generator by detaching fake_B
        # Generated Image Disc Output should be close to zero
        self.pred_fake = net.forward(fake.detach())
        self.loss_D_fake = self.criterionGAN(self.pred_fake, 0)

        # Real
        self.pred_real = net.forward(real)
        self.loss_D_real = self.criterionGAN(self.pred_real, 1)

        # Combined loss
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        return self.loss_D


class DiscLossLS(DiscLoss):
    def name(self):
        return 'DiscLossLS'

    def initialize(self):
        DiscLoss.initialize(self)
        self.criterionGAN = GANLoss(use_l1=False)

    def get_g_loss(self, net, fake):
        return DiscLoss.get_g_loss(self, net, fake)

    def get_loss(self, net, real, fake):
        return DiscLoss.get_loss(self, net, real, fake)


class DiscLossWGANGP(DiscLossLS):
    def name(self):
        return 'DiscLossWGAN-GP'

    def initialize(self):
        DiscLossLS.initialize(self)
        self.LAMBDA = 10

    def get_g_loss(self, net, fake):
        # First, G(A) should fake the discriminator
        self.D_fake = net.forward(fake)
        return -self.D_fake.mean()

    def calc_gradient_penalty(self, netD, real_data, fake_data):
        alpha = torch.rand(1, 1)
        alpha = alpha.expand(real_data.size())
        alpha = alpha.cuda()

        interpolates = alpha * real_data + ((1 - alpha) * fake_data)

        interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        disc_interpolates = netD.forward(interpolates)

        gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                                  grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
                                  create_graph=True, retain_graph=True, only_inputs=True)[0]

        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.LAMBDA
        return gradient_penalty

    def get_loss(self, net, real, fake):
        self.D_fake = net.forward(fake.detach())
        self.D_fake = self.D_fake.mean()

        # Real
        self.D_real = net.forward(real)
        self.D_real = self.D_real.mean()
        # Combined loss
        self.loss_D = self.D_fake - self.D_real
        gradient_penalty = self.calc_gradient_penalty(net, real.data, fake.data)
        return self.loss_D + gradient_penalty

