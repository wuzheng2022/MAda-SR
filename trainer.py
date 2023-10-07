from fileinput import filename
import os
import math
from decimal import Decimal
from statistics import median

import utility

import numpy as np
import torch
import torch.nn.utils as utils
from tqdm import tqdm
import pdb
from skimage.transform import resize
from loss import DilLoss
from optimizer_adam import *
from mas_utils import *



def batch_bicubic_resize(inp_batch, scale):
    [bSize, h, w, c] = inp_batch.shape
    out_batch = np.zeros([bSize, h*scale, w*scale, c], dtype=np.float32)
    for i in range(bSize):
        slc = inp_batch[i, :, :, :]
        slc = resize(slc, [h*scale, w*scale, c], order = 3, 
                     mode = 'symmetric', preserve_range = True)
        # slc = slc.astype(inp_batch.dtype)
        out_batch[i, :, :, :] = slc[:, :, :]
    
    return out_batch




class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale
        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.model = my_model
        self.loss = my_loss
        self.device = torch.device('cpu' if self.args.cpu else 'cuda')
        self.reg_lambda = args.reg_lambda
        if self.args.load != '':
            self.optimizer.load(ckp.dir, epoch=len(ckp.log))

        if self.args.lml == 'our': #采用adam
            self.optimizer = utility.make_optimizer(args, self.model)
        else:   # ewc, si, icarl: 采用sgd
            self.optimizer = utility.make_base_optimizer(args, self.model)
        self.error_last = 1e8

        if self.args.lml == 'si':
            self.si_init()

    def train(self):
        self.loss.step()
        #print(self.optimizer.get_last_epoch())
        epoch = self.optimizer.get_last_epoch() + 1
        #pdb.set_trace
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()

        timer_data, timer_model = utility.timer(), utility.timer()
        # TEMP
        self.loader_train.dataset.set_scale(0)
        for batch, (lr, hr, _,) in enumerate(self.loader_train):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)
            loss = self.loss(sr, hr)
            loss.backward()

            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(self.loader_train.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(self.loader_train))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    def test(self, loader_test, task, tname):
        torch.set_grad_enabled(False)

        epoch = self.optimizer.get_last_epoch()
        #print(epoch)
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(
            torch.zeros(1, len(loader_test), len(self.scale))
        )
        self.model.eval()

        timer_test = utility.timer()
        
        if self.args.save_results: self.ckp.begin_background()
        idx_data = 0
        for idx_scale, scale in enumerate(self.scale):
            loader_test.dataset.set_scale(idx_scale)
            mean_ssim = 0
            for lr, hr, filename in tqdm(loader_test, ncols=80):
                lr, hr = self.prepare(lr, hr)
                
                '''
                # 单通道测试 (由于GPU限制, batchsize为96, 无法满足)
                sr_img = np.zeros(hr.cpu().numpy().shape)
                hr_img = np.zeros(hr.cpu().numpy().shape)
                mean_psnr = 0.0
                for b in range(lr.size(1)):
                    tmp_lr = lr[:,b,:,:].unsqueeze(1)
                    tmp_hr = hr[:,b,:,:].unsqueeze(1)
                    a = self.model(tmp_lr, idx_scale)
                    
                    # print('lr max{}, min{}'.format(torch.max(tmp_lr), torch.min(tmp_lr)))
                    # print('hr max{}, min{}'.format(torch.max(tmp_hr), torch.min(tmp_hr)))
                    # print('quantize before max{}, min{}'.format(torch.max(sr), torch.min(sr)))
                    a = utility.quantize(a, self.args.rgb_range)
                    # print('quantize after max{}, min{}'.format(torch.max(sr), torch.min(sr)))

                    hr_a = (tmp_hr.cpu().numpy()*32767.0).astype(np.int16)
                    lr_a = (tmp_lr.cpu().numpy()*32767.0).astype(np.int16)
                    sr_a = (a.cpu().numpy()*32767.0).astype(np.int16)

                    sr_img[0,b,:,:] = sr_a[0]
                    hr_img[0,b,:,:] = hr_a[0]

                # print('name {}\n, hr max:{}\t, min{}\t, type{}\t'.format(filename, np.max(hr_img), np.min(hr_img), hr_a.dtype))
                # print('sr max:{}\t, min{}\t, type{}\t'.format(np.max(sr_img), np.min(sr_img), sr_a.dtype))
                # print('lr max{}, min{}, type{}'.format(np.max(lr_a), np.min(lr_a), lr_a.dtype))

                mean_psnr = utility.calc_psnr_npy(
                    hr_img, sr_img, 
                    data_range=(np.max(hr_img)-np.min(hr_img))
                )
                
                
                '''
                # 多通道转为batch方式测试
                # torch transpose (1,96,h,w) -> (96,1,h,w)
                if tname == 'ChestX':
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range, norm=loader_test.dataset.norm)
                    mean_psnr = utility.calc_psnr(
                        hr[0,0,:,:], sr[0,0,:,:], 
                        scale, self.args.rgb_range
                    )
                    hr_img = (hr.cpu().numpy()*255.0).astype(np.int16)
                    sr_img = (sr.cpu().numpy()*255.0).astype(np.int16)
                else:
                    lr = torch.transpose(lr, 1, 0)
                    sr = self.model(lr, idx_scale)
                    sr = utility.quantize(sr, self.args.rgb_range, norm=loader_test.dataset.norm)
                    sr = torch.transpose(sr, 1, 0)

                    hr_img = (hr.cpu().numpy()*32767.0).astype(np.int16)
                    sr_img = (sr.cpu().numpy()*32767.0).astype(np.int16)

                    mean_psnr = utility.calc_psnr_npy(
                        hr_img, sr_img, 
                        data_range=(np.max(hr_img)-np.min(hr_img))
                    )
                
                s = utility.calc_batch_ssim(
                    hr_img, sr_img
                )
                mean_ssim += s
                self.ckp.write_log('{}\tPSNR: {:.3f}, SSIM:{:.4f}'.format(filename, mean_psnr, s))
                
                self.ckp.log[-1, idx_data, idx_scale] += mean_psnr

                # sr_img and hr_img are the ndarray type
                save_list = [sr_img]
                if self.args.save_gt:
                    save_list.extend([hr_img])
                
                if self.args.save_results:
                    self.ckp.save_results(self.args.data_test[0], filename[0], save_list, scale)

            self.ckp.log[-1, idx_data, idx_scale] /= len(loader_test)
            mean_ssim /= len(loader_test)

            best = self.ckp.log.max(0)
            self.ckp.write_log(
                '[Task: {}-{}][{} x{}]\tPSNR: {:.3f} (Best: {:.3f} @epoch {}), SSIM:{:.4f}'.format(
                    task, tname,
                    loader_test.dataset.name,
                    scale,
                    self.ckp.log[-1, idx_data, idx_scale],
                    best[0][idx_data, idx_scale],
                    best[1][idx_data, idx_scale],
                    mean_ssim
                )
            )


        self.ckp.write_log('Forward: {:.2f}s\n'.format(timer_test.toc()))
        self.ckp.write_log('Saving...')

        if self.args.save_results:
            self.ckp.end_background()

        if not self.args.test_only:
            self.ckp.save(self, epoch, is_best=(best[1][0, 0]+1 == epoch))

        self.ckp.write_log(
            'Total: {:.2f}s\n'.format(timer_test.toc()), refresh=True
        )

        torch.set_grad_enabled(True)

    def prepare(self, *args):
        device = torch.device('cpu' if self.args.cpu else 'cuda')
        def _prepare(tensor):
            # if self.args.precision == 'half': tensor = tensor.half()
            return tensor.to(device)

        return [_prepare(a) for a in args]

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.optimizer.get_last_epoch() + 1
            return epoch >= self.args.epochs


    # ----------------------------------------------------------------------------------------------------------------------
    # -------------------------------------------------  multi task   ------------------------------------------------------
    # ----------------------------------------------------------------------------------------------------------------------

    def ewc_train(self, epocha, train_loader):
        '''训练'''
        self.loss.step()

        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()

        images, labels = None, None
        init_parameters = []
        ewc_loss = 0
        for batch, (lr, hr, _) in enumerate(train_loader):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)

            # ewc 新加的损失
            if (epocha == 1 or batch == 0):
                ewc_loss = 0
                images = lr
                labels = hr
                init_parameters = self.model.model.init_params()
            else:
                fisher_information = self._calc_fisher_information(images, labels)
                ewc_penalty_list = []
                for f, p1, p2 in zip(fisher_information, init_parameters, self.model.model.get_params()):
                    a = (p1 - p2) ** 2 + 5e-4
                    ewc_penalty_list.append(torch.sum(torch.mul(f, a)))
                ewc_loss = torch.stack(ewc_penalty_list).sum()

            # compute primary loss
            loss = self.loss(sr, hr) + self.reg_lambda * ewc_loss
            loss.backward()

            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(train_loader.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(train_loader))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()


    def si_init(self):
        self.task_count = 0
        self.damping_factor = 0.1
        self.params = {n: p for n, p in self.model.named_parameters() if p.requires_grad}  # For convenience
        self.regularization_terms = {}
        self.w = {}
        for n, p in self.params.items():
            self.w[n] = p.clone().detach().zero_()

        # The initial_params will only be used in the first task (when the regularization_terms is empty)
        self.initial_params = {}
        for n, p in self.params.items():
            self.initial_params[n] = p.clone().detach()
        
    def si_update_model(self, lr, hr, task, batch):
        unreg_gradients = {}
        
        # 1.Save current parameters
        old_params = {}
        for n, p in self.params.items():
            old_params[n] = p.clone().detach()

        # 2. Collect the gradients without regularization term
        # forward
        sr = self.model(lr, 0)

        # Collect the gradients without regularization term
        loss = self.loss(sr, hr)
        
        self.optimizer.zero_grad()
        loss.backward(retain_graph=True)

        for n, p in self.params.items():
            if p.grad is not None:
                unreg_gradients[n] = p.grad.clone().detach()

         # 3. Normal update with regularization
        sr = self.model(lr, 0)
        loss = self.loss(sr, hr)
        
        # Normal update with regularization
        # Calculate the reg_loss only when the regularization_terms exists
        reg_loss = 0
        if len(self.regularization_terms)>0:
            for i,reg_term in self.regularization_terms.items():
                task_reg_loss = 0
                importance = reg_term['importance']
                task_param = reg_term['task_param']
                for n, p in self.params.items():
                    task_reg_loss += (importance[n] * (p - task_param[n]) ** 2).sum()
                reg_loss += task_reg_loss
            if(reg_loss>10 and reg_loss<=100):
                self.reg_lambda = 1e-1
            elif(reg_loss>100 and reg_loss<=1e3):
                self.reg_lambda = 1e-2
            elif(reg_loss>1e3):
                self.reg_lambda = 1e-5
            loss += self.reg_lambda * reg_loss

        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )

        # 4. Accumulate the w
        for n, p in self.params.items():
            delta = p.detach() - old_params[n]
            if n in unreg_gradients.keys():  # In multi-head network, some head could have no grad (lazy) since no loss go through it.
                self.w[n] -= unreg_gradients[n] * delta  # w[n] is >=0
        return loss, sr


    def si_train(self, task, train_loader):
        '''训练'''
        self.loss.step()

        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()
        
        for batch, (lr, hr, _) in enumerate(train_loader):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            loss, sr = self.si_update_model(lr, hr, task, batch)
            
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(train_loader.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(train_loader))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()


    def icarl_train(self, train_loader):
        '''训练'''
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (lr, hr, _) in enumerate(train_loader):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)

            # compute primary loss
            loss_primary = self.loss(sr, hr)
            
            # icarl loss
            self.criterion_dil = DilLoss(self.scale, lr.size(0))
            dilLoss_primary = self.criterion_dil(sr, hr)

            # compute total loss
            loss = loss_primary + self.reg_lambda * dilLoss_primary
            loss.backward()
            if self.args.gclip > 0:
                utils.clip_grad_value_(
                    self.model.parameters(),
                    self.args.gclip
                )
            self.optimizer.step()

            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(train_loader.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(train_loader))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()

    
    def lwu_train(self, train_loader):
        '''训练'''
        self.loss.step()
        epoch = self.optimizer.get_last_epoch() + 1
        lr = self.optimizer.get_lr()

        self.ckp.write_log(
            '[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr))
        )
        self.loss.start_log()
        self.model.train()
        timer_data, timer_model = utility.timer(), utility.timer()

        for batch, (lr, hr, _) in enumerate(train_loader):
            lr, hr = self.prepare(lr, hr)
            timer_data.hold()
            timer_model.tic()

            self.optimizer.zero_grad()
            sr = self.model(lr, 0)

            # compute primary loss
            loss = self.loss(sr, hr)
            loss.backward()
            self.optimizer.step(self.model.model.reg_params, self.device)
            
            timer_model.hold()

            if (batch + 1) % self.args.print_every == 0:
                self.ckp.write_log('[{}/{}]\t{}\t{:.1f}+{:.1f}s'.format(
                    (batch + 1) * self.args.batch_size,
                    len(train_loader.dataset),
                    self.loss.display_loss(batch),
                    timer_model.release(),
                    timer_data.release()))

            timer_data.tic()

        self.loss.end_log(len(train_loader))
        self.error_last = self.loss.log[-1, -1]
        self.optimizer.schedule()


    def _calc_fisher_information(self, lr, hr):
        self.optimizer.zero_grad()

        sr = self.model(lr, 0)
        mle = self.loss(sr, hr)
        mle.backward()

        params = []
        for param in self.model.parameters():
            params.append(self.reg_lambda * param.grad ** 2)
        return params

    def _si_more_opt(self, train_loader):
        # 2.Backup the weight of current task
        task_param = {}
        for n, p in self.params.items():
            task_param[n] = p.clone().detach()

        # 3.Calculate the importance of weights for current task
        importance = self._calc_importance(train_loader)
        # Save the weight and importance of weights of current task
        self.task_count += 1
        if len(self.regularization_terms)>0:
            # Always use only one slot in self.regularization_terms
            self.regularization_terms[1] = {'importance':importance, 'task_param':task_param}
        else:
            # Use a new slot to store the task-specific information
            self.regularization_terms[self.task_count] = {'importance':importance, 'task_param':task_param}

    def _calc_importance(self, dataloader):
        # Initialize the importance matrix
        if len(self.regularization_terms)>0: # The case of after the first task
            importance = self.regularization_terms[1]['importance']
            prev_params = self.regularization_terms[1]['task_param']
        else:  # It is in the first task
            importance = {}
            for n, p in self.params.items():
                importance[n] = p.clone().detach().fill_(0)  # zero initialized
            prev_params = self.initial_params

        # Calculate or accumulate the Omega (the importance matrix)
        for n, p in importance.items():
            delta_theta = self.params[n].detach() - prev_params[n]
            p += self.w[n]/(delta_theta**2 + self.damping_factor)
            self.w[n].zero_()

        return importance



    def multitest(self):
        num_of_tasks = len(self.loader_train)

        for task in range(1, num_of_tasks + 1):
            self.ckp.write_log(
                'testing the model on task {}\t'.format(task)
            )

            tname = self.loader_train[task-1].dataset.image_type
            if task == 1:
                self.args.save = os.path.join(self.args.save, 'task_{}_{}'.format(task, tname))
            else:
                self.args.save = self.args.save.replace(os.path.basename(self.args.save), 'task_{}_{}'.format(task, tname))
            self.ckp = utility.checkpoint(self.args)

            # 把每一个任务对应的数据集取出来
            test_loader = self.loader_test[task - 1]

            self.test(test_loader, task, tname)


    def multitrain(self):
        '''多任务训练'''
        num_of_tasks = len(self.loader_train)

        # 训练
        for task in range(1, num_of_tasks + 1):
            self.ckp.write_log(
                'Training the model on task {}\t'.format(task)
            )

            tname = self.loader_train[task-1].dataset.image_type
            if task == 1:
                self.args.save = os.path.join(self.args.save, 'task_{}_{}'.format(task, tname))
            else:
                self.args.save = self.args.save.replace(os.path.basename(self.args.save), 'task_{}_{}'.format(task, tname))
            print(self.args.save)
            self.ckp = utility.checkpoint(self.args)

            # 把每一个任务对应的数据集取出来
            train_loader = self.loader_train[task - 1]
            test_loader = self.loader_test[task - 1]

            # 初始化正则参数reg_params
            if self.args.lml == "our" and task == 1:
                self.model.model.init_reg_params()
            elif self.args.lml == "our" and task>1:
                self.model.model.init_reg_params_across_tasks()
                print("-------------------- init across task ------------------------")

            self.test(test_loader, task, tname)

            # 实验2：增益训练，每个task都有一个baseline值和(nEpochs-1)个psnr值，除第一个任务外
            for epoch in range(1, self.args.epochs+1):

                if(self.args.lml == "our"):
                    self.lwu_train(train_loader)
                elif(self.args.lml == "si"):
                    self.si_train(task, train_loader)
                elif(self.args.lml == "icarl"):
                    self.icarl_train(train_loader)
                elif(self.args.lml == "ewc"):
                    self.ewc_train(epoch, train_loader)
                else:
                    raise AssertionError(
                            "No matched lifelong machine learning method. It supports that [lwu, ewc, si, icarl]")
                
                if epoch % 10 == 0 or epoch == self.args.epochs:
                    self.test(test_loader, task, tname)

                if self.args.lml == "our" and epoch == self.args.epochs:
                    # 创建一个Ω更新的优化器类，传入到模型当中更新参数的Ω值
                    optimizer_ft_model = omega_update(self.model.model.reg_params)
                    # 更新Ω
                    self.model.model = compute_omega_grads_norm(self.model.model, train_loader,
                                                        optimizer_ft_model, self.device)
                    self.test(test_loader, task, tname)

            if self.args.lml == "our" and task > 1:
                self.model.model = consolidate_reg_params(self.model.model)
                
            if self.args.lml == "si":
                # SI 流程
                self._si_more_opt(train_loader)

        self.ckp.write_log(
            'Computing the forgetting of the model on all task.\t'
        )