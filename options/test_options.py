from .base_options import BaseOptions

class ValOptions(BaseOptions):
    def initialize(self, parser):
        parser=BaseOptions.initialize(self,parser)
        parser.add_argument('--results_dir', type=str, default=parser.get_default('store_root')+'results/', help='saves results here.')
        parser.add_argument('--phase', type=str, default='val', help='train, val, test, etc')
        parser.add_argument('--saveimg', type=bool, default=False,help='whether save image during each evaluation')
        parser.add_argument('--saveimg_epoch', type=int, default=50,help='epoch to save image of validation set')
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain=False
        return parser

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--results_dir', type=str, default=parser.get_default('store_root')+'results/', help='saves results here.')
        parser.add_argument('--aspect_ratio', type=float, default=1.0, help='aspect ratio of result images')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--num_test', type=int, default=2000, help='how many test images to run')
        parser.set_defaults(model='seaunet')
        # To avoid cropping, the loadSize should be the same as fineSize
        parser.set_defaults(loadSize=parser.get_default('fineSize'))
        self.isTrain = False
        return parser