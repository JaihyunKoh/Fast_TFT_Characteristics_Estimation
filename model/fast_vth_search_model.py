# from util.util import tensor2im
from collections import OrderedDict
from .base_model import BaseModel
from . import network
from .losses import init_loss
import torch

class FastVthSearch(BaseModel):
    def name(self):
        return 'FastVthSearch'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain
        self.opt = opt

        # load/define network
        self.model_k = network.define_fast_vth_block(int(opt.num_seq/opt.num_sample), 2, opt.num_channels, opt.kernel_size, opt.dropout)
        self.model_th = network.define_fast_vth_block(int(opt.num_seq/opt.num_sample), 2, opt.num_channels, opt.kernel_size, opt.dropout)
        # self.model_k_sec = network.define_fast_vth_block(int(opt.num_seq/opt.num_sample), 2, opt.num_channels, opt.kernel_size, opt.dropout)

        if len(opt.gpu_ids) > 0:
            self.model_k = self.model_k.cuda()
            self.model_th = self.model_th.cuda()
            # self.model_k_sec = self.model_k_sec.cuda()

        if self.isTrain:
            # initialize optimizers
            self.old_lr = opt.lr
            self.optimizer_k = torch.optim.Adam(self.model_k.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            self.optimizer_th = torch.optim.Adam(self.model_th.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))
            # self.optimizer_k_sec = torch.optim.Adam(self.model_k_sec.parameters(), lr=opt.lr, betas=(opt.beta1, opt.beta2))


            # define loss function
            self.contLoss = init_loss(opt)


        print('--- Vth Networks initialized ---')
        print(self.model_k)
        print(self.model_th)
        # print(self.model_k_sec)
        network.print_network(self.model_k)
        network.print_network(self.model_th)
        # network.print_network(self.model_k_sec)
        print('-------------------------------------------------')

        if not self.isTrain or opt.continue_train or opt.phase == 'test':
            self.model_k.eval()
            self.model_th.eval()
            # self.model_k_sec.eval()
            # self.load_network(self.model_k, 'model_k', opt.which_epoch)
            # self.load_network(self.model_th, 'model_th', opt.which_epoch)
            # self.load_network(self.model_k_sec, 'model_k_sec', opt.which_epoch)
            self.model_k.load_state_dict(torch.load('./checkpoint/2300_net_model_k.pth'))
            self.model_th.load_state_dict(torch.load('./checkpoint/2300_net_model_th.pth'))
            print('Network was successfully loaded!!')


    def set_input(self, input):
        self.input = input['input'].cuda() # [bs, 5]
        self.k_single = input['k'].unsqueeze(1).cuda()  # [bs, 1]
        self.k = self.k_single.expand_as(self.input).cuda()
        self.gt = input['gt'].cuda()
        self.data_path = input['data_path']

    def forward(self):
        self.pre_k_vec = self.pre_k_single.expand_as(self.input).cuda() # [bs, 5]
        self.vs_pre_k = torch.cat([torch.unsqueeze(self.input, dim=1), torch.unsqueeze(self.pre_k_vec.data, dim=1)], dim=1) # [bs, 2, 5]
        self.output_th_single = self.model_th.forward(self.vs_pre_k)

        self.pre_th_vec = self.pre_th_single.expand_as(self.input).cuda() # [bs, 5]
        self.vs_pre_th = torch.cat([torch.unsqueeze(self.input, dim=1), torch.unsqueeze(self.pre_th_vec.data, dim=1)], dim=1) # [bs, 2, 5]
        self.output_k_single = self.model_k(self.vs_pre_th) # [bs, 1]

    def backward(self):
        self.cont_loss_k = self.contLoss.get_loss(self.output_k_single, self.k_single)
        self.cont_loss_th = self.contLoss.get_loss(self.output_th_single, self.gt)
        self.total_loss = self.cont_loss_k + self.cont_loss_th

        self.total_loss.backward()

    def optimize_parameters(self):
        n_iter = self.opt.n_iter
        self.pre_k_single = torch.zeros_like(self.k_single)
        self.pre_th_single = torch.zeros_like(self.gt)
        for iter in range(n_iter):
            self.forward()
            if iter is not 0:
                self.optimizer_k.zero_grad()
                self.optimizer_th.zero_grad()
                self.backward()
                self.optimizer_k.step()
                self.optimizer_th.step()
            self.pre_k_single = self.output_k_single
            self.pre_th_single = self.output_th_single

    # no backprop gradients
    def test(self):
        with torch.no_grad():
            n_iter = self.opt.n_iter
            self.pre_k_single = torch.zeros_like(self.k_single)
            self.pre_th_single = torch.zeros_like(self.gt)
            for iter in range(n_iter):
                self.forward()
                self.pre_k_single = self.output_k_single
                self.pre_th_single = self.output_th_single

    # get image paths
    def get_data_paths(self):
        return [self.data_path]

    def get_current_errors(self):
        return OrderedDict([('k_loss', self.cont_loss_k.data),
                            ('th_loss', self.cont_loss_th.data)
                            ])

    def get_current_visuals(self):
        gt_k = self.k_single.cpu().numpy()
        output_k = self.output_k_single.cpu().numpy()
        output_th = self.output_th_single.cpu().numpy()
        gt_th = self.gt.cpu().numpy()
        return OrderedDict(
                [('output_th', output_th), ('gt_th', gt_th),
                 ('output_k', output_k), ('gt_k', gt_k)
                ])


    def save(self, label):
        self.save_network(self.model_k, 'model_k', label, self.gpu_ids)
        self.save_network(self.model_th, 'model_th', label, self.gpu_ids)
        # self.save_network(self.model_k_sec, 'model_k_sec', label, self.gpu_ids)

    def update_learning_rate(self):
        # lrd = self.opt.lr / self.opt.niter_decay
        # lr = self.old_lr - lrd
        lr = self.old_lr/2
        for param_group in self.optimizer_k.param_groups:
            param_group['lr'] = lr
        for param_group in self.optimizer_th.param_groups:
            param_group['lr'] = lr
        # for param_group in self.optimizer_k_sec.param_groups:
        #     param_group['lr'] = lr
        print('update learning rate G: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr

