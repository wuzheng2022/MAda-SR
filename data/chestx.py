import os, glob
from data import srdata

class ChestX(srdata.SRData):
    def __init__(self, args, name='ChestX', image_type="ChestX",  train=True, benchmark=False):
        self.norm = 255.0
        self.name = name
        self.downsample_type = args.downsample_type
        self.image_type = image_type
        super(ChestX, self).__init__(
            args, name=name, train=train, benchmark=benchmark
        )

    def _scan(self):
        names_lr = [[] for _ in self.scale]
        t = "train" if self.train else "test"
        d = "B" if self.downsample_type == "bicubic" else "T"
        names_hr = glob.glob(os.path.join(self.dir_hr, t, "*"+self.ext))

        for i in names_hr:
            for si, s in enumerate(self.scale):
                filenum = os.path.basename(i).split('.')[0]
                names_lr[si].append(os.path.join(
                    self.dir_lr,
                    '{}_{}x/{}/{}{}'.format(
                        self.downsample_type,
                        s,
                        t,
                        filenum, 
                        self.ext)
                ))
        # print(names_hr[0])
        # print(names_lr[0][0])
        return names_hr, names_lr

    def _set_filesystem(self, dir_data):
        super(ChestX, self)._set_filesystem(dir_data)
        self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_lr = os.path.join(self.apath, 'LR')
        self.ext = '.png'
        if self.input_large: self.dir_lr += 'L'

