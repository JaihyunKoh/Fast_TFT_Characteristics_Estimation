import argparse
import os
# from util import util
import torch


class BaseOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--dataroot', type=str, default='/media/ssd/satyricon00/Fast_Vth_k3_label/', help='path to images')
        # self.parser.add_argument('--dataroot', type=str, default='/media/ssd/satyricon00/Fast_Vth_10000/', help='path to images')
        self.parser.add_argument('--batchSize', type=int, default=48, help='input batch size')
        self.parser.add_argument('--gpu_ids', type=str, default='3', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='Vth_and_k_est_1ms_10spl_80_80_80_80ch_L1_bs_46_20iter_lr500_half', help='name of the experiment. It decides where to store samples and models')
        self.parser.add_argument('--nThreads', default=3, type=int, help='# threads for loading data')
        self.parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints', help='models are saved here')
        self.parser.add_argument('--n_iter', default=20, type=int, help='the number of recurrent iterations')

        self.parser.add_argument('--num_seq', type=int, default=100, help='the number of Vs input sequence, ts')
        self.parser.add_argument('--num_sample', type=int, default=10, help='data sampling period tf')
        self.parser.add_argument('--num_inputs', type=int, default=1)
        self.parser.add_argument('--num_channels', type=int, default=[80, 80, 80, 80])
        # self.parser.add_argument('--num_channels', type=int, default=[160, 80, 40, 20])
        # self.parser.add_argument('--num_channels', type=int, default=[80, 80, 80, 80, 80, 80, 80, 80])
        self.parser.add_argument('--kernel_size', type=int, default=3)
        self.parser.add_argument('--dropout', type=float, default=0.0)

        self.parser.add_argument('--display_winsize', type=int, default=256, help='display window size')
        self.parser.add_argument('--display_id', type=int, default=1, help='window id of the web display')
        self.parser.add_argument('--display_port', type=int, default=8103, help='visdom port of the web display')
        self.parser.add_argument('--display_single_pane_ncols', type=int, default=0, help='if positive, display all images in a single visdom web panel with certain number of images per row.')
        self.parser.add_argument('--max_dataset_size', type=int, default=float("inf"), help='Maximum number of samples allowed per dataset. If the dataset directory contains more than max_dataset_size, only a subset is loaded.')

        self.initialized = True

    def parse(self):
        if not self.initialized:
            self.initialize()
        self.opt = self.parser.parse_args()
        self.opt.isTrain = self.isTrain  # train or test

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        # save to the disk
        # expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.phase)
        # util.mkdirs(expr_dir)
        # file_name = os.path.join(expr_dir, 'opt.txt')
        # with open(file_name, 'wt') as opt_file:
        #     opt_file.write('------------ Options -------------\n')
        #     for k, v in sorted(args.items()):
        #         opt_file.write('%s: %s\n' % (str(k), str(v)))
        #     opt_file.write('-------------- End ----------------\n')
        return self.opt
