import torch.nn as nn
import exp_models
import base_models
from transformers import BertConfig
from Dataset import Wikitext
from accelerate import Accelerator
from torch.utils.tensorboard import SummaryWriter
from transformers import BertConfig, get_cosine_schedule_with_warmup
import torch.optim as optim
from base_loggers import l_train_loss, l_test_loss, l_ntk, l_grad, l_dis_wi_w0, l_learning_rate, l_grad_all

import torch
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
        
        self._accelerator  = Accelerator()
        
    
    def init_model(self) -> None:
        
        self._num_epochs = 400
        num_updates = self._num_epochs * len(self._train_loader)

        self._optimizer = optim.AdamW(self._base_model.parameters(), lr=1e-3, weight_decay=0.01, betas=[0.9, 0.999], eps=1e-6)

        self._lr_scheduler = get_cosine_schedule_with_warmup(
            optimizer=self._optimizer,
            num_warmup_steps=num_updates * 0.05,
            num_training_steps=num_updates,
        )
        
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

    
    def train(self) -> None:
        self._l_dis_wi_w0.init_w0(self._base_model.named_parameters())
        
        for epoch in range(self._num_epochs):
            self._base_model.train()
            
            self._l_ntk.compute(train_loader=self._train_loader, epoch=epoch)
            self._l_ntk.flush()
            
            self._l_dis_wi_w0.compute(parameter=self._base_model.named_parameters(), epoch=epoch)
            self._l_dis_wi_w0.flush()
            
            self._l_lr.compute(optimizer=self._optimizer, epoch=epoch)
            self._l_lr.flush()
            
            for i, batch in enumerate(self._train_loader):
                
                loss, _ = self._base_model(**batch)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                self._lr_scheduler.step()
                
                self._l_train_loss.compute(loss=loss, epoch=epoch)
                # self._l_grad.compute(parameter=self._base_model.named_parameters())
            
            if epoch % 10 == 0 or (epoch >= 90 and epoch < 100):
                self._l_grad_all.compute(parameter=self._base_model.named_parameters(), epoch=epoch)
            
            self._l_test_loss.compute(test_loader=self._test_loader, epoch=epoch)
            
            self._l_test_loss.flush()
            self._l_train_loss.flush()
            # self._l_grad.flush()
            
            
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

import sys

if __name__ == "__main__":
    # 10315 这个seed目前在17个epoch就可以复现不稳定性
    # 29779 27308 这个seed用于bert_small
    seed = 29779
    set_seed(int(seed))
    
    availabe_device = get_available_cuda_device()
    if availabe_device < 0:
        raise Exception("no available devices")
    torch.cuda.set_device(availabe_device)
    
    b = bert_test(model_name="bert_" + str(seed), config_file="config/bert_small.json")
    
    b.init_model()
    b.train()
