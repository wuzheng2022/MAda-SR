from __future__ import print_function
import math

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

import copy
import os
import shutil


class local_sgd(optim.SGD):
	def __init__(self, params, reg_lambda, lr = 0.001, momentum = 0, dampening = 0, weight_decay = 0, nesterov = False):
		super(local_sgd, self).__init__(params, lr, momentum, dampening, weight_decay, nesterov)
		self.reg_lambda = reg_lambda

	def __setstate__(self, state):
		super(local_sgd, self).__setstate__(state)
		#state是一个有默认值的字典, 默认值类型为字典;保存optimizer的当前状态

	def step(self, reg_params, device, closure=None):

		loss = None

		if closure is not None:
			loss = closure()

		# self.param_groups保存所有待优化的参数;其中的每一项都是一个字典,对应一组待优化参数及相关的参数
		for group in self.param_groups:
			weight_decay = group['weight_decay']
			momentum = group['momentum']
			dampening = group['dampening']
			nesterov = group['nesterov']

			for p in group['params']:
				
				if p.grad is None:
					continue
				# 直接能拿到参数的梯度
				d_p = p.grad.data
				# 对于freeze_layers层里面的参数，更新方式与Ω和λ有关
				if p in reg_params:

					param_dict = reg_params[p]
					# 获取该参数的重要值Ω
					omega = param_dict['omega']
					# 前一个任务训练完成的参数值
					init_val = param_dict['init_val']
					# 当前任务中该参数的值
					curr_param_value = p.data
					curr_param_value = curr_param_value.to(device)
					init_val = init_val.to(device)
					omega = omega.to(device)

					#get the difference
					# 计算训练后，参数被改变的值
					param_diff = curr_param_value - init_val

					#get the gradient for the penalty term for change in the weights of the parameters
					#获取参数权重变化的惩罚项的梯度,gij=(θij-θinit）*λ*Ωij*2
					local_grad = torch.mul(param_diff, 2*self.reg_lambda*omega)
					
					del param_diff
					del omega
					del init_val
					del curr_param_value
					# 累加，有点像损失函数中的第二项的导数
					d_p = d_p + local_grad
					
					del local_grad
					
				# 常规的参数更新方式
				if (weight_decay != 0):
					d_p.add_(weight_decay, p.data)

				if (momentum != 0):
					param_state = self.state[p]
					if 'momentum_buffer' not in param_state:
						buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
					else:
						buf = param_state['momentum_buffer']
						buf.mul_(momentum).add_(1 - dampening, d_p)
					if nesterov:
						d_p = d_p.add(momentum, buf)
					else:
						d_p = buf

				p.data.add_(-group['lr'], d_p)

		return loss

class local_adam(optim.Adam):
    """Implements Adam algorithm.
    """

    def __init__(self, params, reg_lambda, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False):
        super(local_adam, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)
        self.reg_lambda = reg_lambda

    def __setstate__(self, state):
        super(local_sgd, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, reg_params, device, closure=None):
        """Performs a single optimization step.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data  # gt
                
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                # -------------------------------------------------------------------------
                # 对于freeze_layers层里面的参数，更新方式与Ω和λ有关
                if p in reg_params:

                    param_dict = reg_params[p]
                    # 获取该参数的重要值Ω
                    omega = param_dict['omega']
                    # 前一个任务训练完成的参数值
                    init_val = param_dict['init_val']
                    # 当前任务中该参数的值
                    curr_param_value = p.data
                    curr_param_value = curr_param_value.to(device)
                    init_val = init_val.to(device)
                    omega = omega.to(device)

                    #get the difference
                    # 计算训练后，参数被改变的值
                    param_diff = curr_param_value - init_val

                    #get the gradient for the penalty term for change in the weights of the parameters
                    #获取参数权重变化的惩罚项的梯度,gij=(θij-θinit）*λ*Ωij*2
                    local_grad = torch.mul(param_diff, 2*self.reg_lambda*omega)
                    
                    del param_diff
                    del omega
                    del init_val
                    del curr_param_value
                    # 累加，损失函数中的第二项的导数
                    grad = grad + local_grad
                    
                    del local_grad

                # 之前的step累计数据
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values：梯度值的指数移动平均值 mt
                    state['exp_avg'] = torch.zeros_like(p.data)
                    # Exponential moving average of squared gradient values：平方梯度值的指数移动平均值 vt
                    state['exp_avg_sq'] = torch.zeros_like(p.data)
                    if amsgrad: #是否使用此的AMSGrad变体
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p.data)

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                # 偏置校正
                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad.add_(group['weight_decay'], p.data)

                # Decay the first and second moment running average coefficient
                # mt = mt-1*β1+(1-β1)*gt
                # exp_avg.mul_(beta1).add_(1 - beta1, grad)
                # # vt = vt-1*β2+(1-β2)*gt*gt
                # exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    # √vt/(1-β2)^t+eps
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                # lr/(1-β1)^t
                step_size = group['lr'] / bias_correction1
                # 更新参数：θt+1 = θt - mt/（√vt/(1-β2)^t+eps）
                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


class omega_update(optim.Adam):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0, amsgrad=False):
        super(omega_update, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def __setstate__(self, state):
        super(omega_update, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, reg_params, batch_index, batch_size, device, closure = None):
        loss = None

        if closure is not None:
            loss = closure()
        #
        for group in self.param_groups:
            for p in group['params']:
                
                if p.grad is None:
                    continue

                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                if p in reg_params:

                    grad_data = p.grad.data

                    #The absolute value of the grad_data that is to be added to omega 
                    # 获取该参数的梯度
                    grad_data_copy = p.grad.data.clone()
                    # 取梯度的绝对值
                    grad_data_copy = grad_data_copy.abs()
                    
                    param_dict = reg_params[p]

                    omega = param_dict['omega']
                    omega = omega.to(device)

                    current_size = (batch_index+1)*batch_size
                    step_size = 1/float(current_size)

                    #Incremental update for the omega - Ω的增量更新
                    # Ωij = Ωij+(1/(N+1)N)*(gij-N*Ωij)
                    omega = omega + step_size*(grad_data_copy - batch_size*(omega))

                    param_dict['omega'] = omega

                    reg_params[p] = param_dict

        return loss


class omega_vector_update(optim.Adam):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,weight_decay=0, amsgrad=False):
        super(omega_vector_update, self).__init__(params, lr, betas, eps, weight_decay, amsgrad)

    def __setstate__(self, state):
        super(omega_vector_update, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    def step(self, reg_params, finality, batch_index, batch_size, use_gpu, closure = None):
        loss = None

        device = torch.device("cuda:0" if use_gpu else "cpu")

        if closure is not None:
            loss = closure()

        for group in self.param_groups:

            for p in group['params']:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']
                
                if p in reg_params:

                    grad_data = p.grad.data

                    #The absolute value of the grad_data that is to be added to omega 
                    grad_data_copy = p.grad.data.clone()
                    grad_data_copy = grad_data_copy.abs()
                    
                    param_dict = reg_params[p]

                    if not finality:
                        
                        if 'temp_grad' in reg_params.keys():
                            temp_grad = param_dict['temp_grad']
                        
                        else:
                            temp_grad = torch.FloatTensor(p.data.size()).zero_()
                            temp_grad = temp_grad.to(device)
                        
                        temp_grad = temp_grad + grad_data_copy
                        param_dict['temp_grad'] = temp_grad
                        
                        del temp_data


                    else:
                        
                        #temp_grad variable
                        temp_grad = param_dict['temp_grad']
                        temp_grad = temp_grad + grad_data_copy

                        #omega variable
                        omega = param_dict['omega']
                        omega.to(device)

                        current_size = (batch_index+1)*batch_size
                        step_size = 1/float(current_size)

                        #Incremental update for the omega  
                        omega = omega + step_size*(temp_grad - batch_size*(omega))

                        param_dict['omega'] = omega
                        
                        reg_params[p] = param_dict

                        del omega
                        del param_dict
                    
                    del grad_data
                    del grad_data_copy

        return loss

