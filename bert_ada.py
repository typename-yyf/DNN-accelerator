import torch

def get_available_cuda_device() -> int:
    max_devs = torch.cuda.device_count()
    for i in range(max_devs):
        try:
            mem = torch.cuda.mem_get_info(i)
        except:
            continue
        if mem[0] / mem[1] > 0.85:
            return i
    return -1

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

availabe_device = get_available_cuda_device()
if availabe_device < 0:
    raise Exception("no available devices")
torch.cuda.set_device(availabe_device)

import torch.nn as nn
import exp_models
import base_models
from transformers import BertConfig
from Dataset import Wikitext
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim
from base_loggers import l_train_loss, l_test_loss, l_ntk, l_grad, l_dis_wi_w0, \
    l_learning_rate, l_grad_all, l_hessian


import numpy as np
import random

from file_writer import file_writer

class bert_test(exp_models.exp_models):
    _accelerator: Accelerator
    
    _l_train_loss: l_train_loss
    _l_test_loss: l_test_loss
    _l_ntk: l_ntk
    _l_grad: l_grad
    _l_lr: l_learning_rate
    _l_dis_wi_w0: l_dis_wi_w0
    
    _writer: SummaryWriter
    _file_writer: file_writer
    _num_epochs: int
    _start_epoch: int
    
    _mask: dict
    
    def _freeze(self):
        i = 0
        p = self._base_model.named_parameters()
        for name, para in p:
            if "heads" in name:
                with torch.no_grad():
                    para.grad = para.grad * self._mask[i].to("cuda")
                    i += 1
    
    def _random_freeze(self):
        i = 0
        p = self._base_model.named_parameters()
        for name, para in p:
            if "heads" in name:
                with torch.no_grad():
                    t_mask = torch.ones_like(para.grad)
                    
                    para.grad = para.grad * t_mask
                    i += 1
    

    def _save(self, file_path: str, epoch: int):
        self._accelerator.save_state(file_path)
        
    def _load(self, file_path: str):
        torch.cuda.empty_cache()
        
        self._accelerator.load_state(file_path)

    def __init__(self, model_name: str, config_file: str):
        config = BertConfig.from_json_file(config_file)
        
        self._base_model   = base_models.BertForMLM(config=config)
        self._dataset      = Wikitext(config=config)
        self._writer       = SummaryWriter("log/" + model_name)
        self._file_writer  = file_writer("log/" + model_name + "_t")
        
        self._train_loader = self._dataset.train_loader
        self._val_loader   = self._dataset.val_loader
        self._test_loader  = self._dataset.test_loader
        
        self._l_train_loss = l_train_loss(self._base_model, self._writer)
        self._l_test_loss  = l_test_loss(self._base_model, self._writer)
        self._l_ntk        = l_ntk(self._base_model, self._file_writer)
        self._l_grad       = l_grad(self._base_model, self._file_writer)
        self._l_lr         = l_learning_rate(self._base_model, self._writer)
        self._l_dis_wi_w0  = l_dis_wi_w0(self._base_model, self._writer)
        self._l_grad_all   = l_grad_all(self._base_model, self._file_writer)
        self._l_hessian    = l_hessian(self._base_model, self._file_writer)
        # self._l_hessian_lmax     = l_hessian_lmax(self._base_model, self._file_writer)
        # self._l_hessian_lmax_all = l_hessian_lmax_all(self._base_model, self._file_writer)
        
        self._accelerator  = Accelerator()
        
    
    def init_model(self, pth_path: str="", epoch: int=0) -> None:
        self._num_epochs  = 400
        self._start_epoch = epoch
        
        num_updates = self._num_epochs * len(self._train_loader)

        # bert_small
        # self._optimizer = optim.AdamW(self._base_model.parameters(), lr=1e-3, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)
        
        # bert_base
        self._optimizer = optim.AdamW(self._base_model.parameters(), lr=4e-4, weight_decay=0)

        self._lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self._optimizer,
            num_warmup_steps=num_updates * 0.05,
            num_training_steps=num_updates,
        )
            
        # self._mask = file_writer.read_file("log/overlapped_mask", "cuda")
        # self._mask = torch.ones((576, 768))
        
        self._base_model, self._optimizer, self._lr_scheduler, \
            self._train_loader, self._val_loader, self._test_loader = \
        self._accelerator.prepare(
            self._base_model, 
            self._optimizer, 
            self._lr_scheduler, 
            self._train_loader, 
            self._val_loader, 
            self._test_loader
        )
        if pth_path != "":
            self._load(pth_path)
        
        
        
        

    
    def train(self) -> None:
        
        g_norm = -1
        
        # 这两个参数分别指定了模型第一次发生爆炸时的batch和epoch，主要用于保证在爆炸之前，模型与先前的训练过程的一致性
        exp_batch = 208
        exp_epoch = 14
        
        # 相关超参
        exp_threshold = 3 # 爆炸幅度，当检测到| grad_{t+1} | > exp_threshold * | grad_t |认为是大幅度的梯度变化
        alpha1 = 0.2
        alpha2 = 0.8 # 正常grad的对数平均参数
        p_lr = 0.5 # 模型爆炸后lr衰减倍数，这个参数越小模型越稳定
        d_lr = 2e-5 # 模型检测到大梯度但是未发生爆炸时lr增加幅度，这个参数越小模型越稳定
        
        
        
        
        for epoch in range(self._start_epoch, self._num_epochs):
            
            self._base_model.train()
                
            self._l_ntk.compute(train_loader=self._train_loader, epoch=epoch)
            self._l_ntk.flush()
                
                
            self._l_dis_wi_w0.compute(parameter=self._base_model.named_parameters(), epoch=epoch)
            self._l_dis_wi_w0.flush()
                
            self._l_lr.compute(optimizer=self._optimizer, epoch=epoch)
            self._l_lr.flush()
            
            
            for i, batch in enumerate(self._train_loader):
                normal_grad = True
                
                loss, _ = self._base_model(**batch)
                self._optimizer.zero_grad()
                
                loss.backward()
                
                tg_norm = 0
                with torch.no_grad():
                    for name, p in self._base_model.named_parameters():
                        if "heads" in name:
                            tg_norm += p.grad.norm().item()
                            break
                
                if g_norm < 0:
                    g_norm = tg_norm
                else:
                    if tg_norm > g_norm * exp_threshold:
                        normal_grad = False
                        print("large grad detected")
                        self._save(".tmp_pth", 0)
                    else:
                        g_norm = tg_norm * alpha1 + g_norm * alpha2
                    
                
                self._optimizer.step()
                if i <= exp_batch and epoch == exp_epoch:
                    self._lr_scheduler.step()
                
                if normal_grad == False:
                    self._base_model.eval()
                    with torch.no_grad():
                        loss_after, _ = self._base_model(**batch)
                    self._base_model.train()
                    
                    if loss_after > loss:
                        print("explosion detected, at batch", i)
                        
                        self._load(".tmp_pth")
                        for param_group in self._optimizer.param_groups:
                            param_group['lr'] *= p_lr
                    elif i > exp_batch or epoch != exp_epoch: 
                        for param_group in self._optimizer.param_groups:
                            param_group['lr'] += d_lr
                        
                
                if i % 20 == 0:
                    print(loss.item())
                    
                
                self._l_train_loss.compute(loss=loss, epoch=epoch)
                
            
            self._l_test_loss.compute(test_loader=self._test_loader, epoch=epoch)
            
            self._l_test_loss.flush()
            self._l_train_loss.flush()

        

import sys
import argparse
import random


if __name__ == "__main__":
    
    # 10315 这个seed目前在17个epoch就可以复现不稳定性
    # 29779 27308 这个seed用于bert_small
    # seed = random.randint(0, 100000)
    seed = 82722
    set_seed(int(seed))
    
    
    
    # b = bert_test(model_name="debug" + "_" + str(seed), config_file="/home/yfyang/t/config/bert.json")
    b = bert_test(model_name=sys.argv[1] + "_" + str(seed), config_file="config/bert.json")
    
    # b.init_model()
    b.init_model("/home/yfyang/t/brk_pt14", 14)
    b.train()
