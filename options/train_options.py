from .base_options import BaseOptions

class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--display_freq', type=int, default=3840, help='frequency of showing training results on screen')
		self.parser.add_argument('--print_freq', type=int, default=3840, help='frequency of showing training results on console and save log')
		self.parser.add_argument('--save_latest_freq', type=int, default=19200, help='frequency of saving the latest results')
		self.parser.add_argument('--save_epoch_freq', type=int, default=20, help='frequency of saving checkpoints at the end of epochs')
		self.parser.add_argument('--continue_train', type=bool, default=False, help='continue training: load the latest model')
		self.parser.add_argument('--epoch_count', type=int, default=1, help='the starting epoch count, we save the model by <epoch_count>, <epoch_count>+<save_latest_freq>, ...')
		self.parser.add_argument('--phase', type=str, default='train', help='train, val, test, etc')
		self.parser.add_argument('--which_epoch', type=str, default='100', help='which epoch to load? set to latest to use latest cached model')
		self.parser.add_argument('--niter', type=int, default=4000, help='total # of iter')
		# self.parser.add_argument('--niter_decay', type=int, default=100, help='# of iter to halp decay learning rate')
		self.parser.add_argument('--decay_period', type=int, default=500, help='# of iter to halp decay learning rate')


		self.parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate for adam')
		self.parser.add_argument('--no_html', type=bool, default=True, help='do not save intermediate training results to [opt.checkpoints_dir]/[opt.name]/web/')
		self.parser.add_argument('--beta1', type=float, default=0.9, help ='Adam optimizer paremeter 1')
		self.parser.add_argument('--beta2', type=float, default=0.999, help = 'Adam optimizer parameter 2')

		self.isTrain = True