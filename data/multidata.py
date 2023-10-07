from importlib import import_module
from torch.utils.data import DataLoader
import os

class Data:
    def __init__(self, args):
        self.loader_train = []
        self.loader_test = []

        data_dir = args.dir_data

    
        # 制作多任务数据集
        print('===> Loading datasets')
        
        for tdir in args.task:  # 获取该目录下的子文件
            
            if tdir in ['PD', 'T1', 'T2']:
                data_train = 'IXI'
            else:
                data_train = tdir
            module_train = import_module('data.' + data_train.lower())
            module_test = import_module('data.' +  data_train.lower())

            trainset = getattr(module_train, data_train)(args, image_type=tdir)
            training_data_loader = DataLoader(
                trainset,
                batch_size=args.batch_size,
                num_workers=args.n_threads,
                shuffle=True,
                pin_memory=not args.cpu
            )
            # append the dataloaders of these tasks
            self.loader_train.append(training_data_loader)

            # ----------------------------test--------------------
            
            testset = getattr(module_test, data_train)(args, name=data_train, image_type=tdir, train=False)

            testing_data_loader = DataLoader(
                testset,
                batch_size=1,
                num_workers=0,
                shuffle=False,
                pin_memory=not args.cpu
            )
            self.loader_test.append(testing_data_loader)
        print('===> Load finish!')
        # get the number of tasks in the sequence
        print('the number of task are {}' .format(len(self.loader_train)))

