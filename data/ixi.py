import os, glob
from data import srdata

class IXI(srdata.SRData):
    def __init__(self, args, name='IXI', image_type='PD', train=True, benchmark=False):
        self.norm = 32767.0
        self.name = args.downsample_type
        self.downsample_type = args.downsample_type
        self.image_type = image_type
        super(IXI, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_lr = [[] for _ in self.scale]
        t = "train" if self.train else "test"
        d = "B" if self.downsample_type == "bicubic" else "T"
        names_hr = glob.glob(os.path.join(self.dir_hr, self.image_type, t, "*"+self.ext))

        for i in names_hr:
            for si, s in enumerate(self.scale):
                filenum = os.path.basename(i).split('.')[0].split('_')[-1:]
                names_lr[si].append(os.path.join(
                    self.dir_lr,
                    '{}_{}x/{}/{}/LR_{}_{}_{}X_{}{}'.format(
                        self.downsample_type,
                        s,
                        self.image_type,
                        t,
                        self.image_type, d, s,
                        filenum[0], #filenum[1],
                        self.ext)
                ))
        # print(names_hr[0])
        # print(names_lr[0][0])
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(IXI, self)._set_filesystem(dir_data)

        self.dir_hr = os.path.join(self.apath, 'IXI_HR')
        self.dir_lr = os.path.join(self.apath, 'IXI_LR')
        if self.args.ext == 'npy':
            self.ext = '.npy'
        else:
            self.ext = '.png'
        
        if self.input_large: self.dir_lr += 'L'

