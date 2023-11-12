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
torch.cuda.set_device(0)

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

    def _save(self, file_path: str, epoch: int):
        self._accelerator.save_state(file_path)
        
    def _load(self, file_path: str):
        torch.cuda.empty_cache()
        
        self._accelerator.load_state(file_path)

    def __init__(self, model_name: str, config_file: str):
        config = BertConfig.from_json_file(config_file)
        
        self._base_model   = base_models.BertForMLM(config=config)
        self._dataset      = Wikitext(config=config)
        self._writer       = SummaryWriter("/home/yfyang/t/log/" + model_name)
        self._file_writer  = file_writer("/home/yfyang/t/log/" + model_name + "_t")
        
        self._train_loader = self._dataset.train_loader
        self._val_loader   = self._dataset.val_loader
        self._test_loader  = self._dataset.test_loader
        
        self._l_train_loss = l_train_loss(self._base_model, self._writer)
        self._l_test_loss  = l_test_loss(self._base_model, self._writer)
        self._l_ntk        = l_ntk(self._base_model, self._file_writer)
        self._l_grad       = l_grad(self._base_model, self._file_writer)
        self._l_lr         = l_learning_rate(self._base_model, self._writer)
        
        self._accelerator  = Accelerator()
        
    
    def init_model(self, pth_path: str="", from_pretrained="", epoch: int=0) -> None:
        self._num_epochs  = 400
        self._start_epoch = epoch
        
        num_updates = self._num_epochs * len(self._train_loader)

        if from_pretrained != "":
            pretrained_dict = torch.load(from_pretrained)
            model_dict      = self._base_model.state_dict()
            
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if "embeddings" in k}
            model_dict.update(pretrained_dict)
            self._base_model.load_state_dict(model_dict)

        self._optimizer = optim.AdamW(self._base_model.parameters(), lr=4e-4, weight_decay=0)

        # 这里最好改成constant
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
        if pth_path != "":
            self._load(pth_path)
    
    def _freeze(self, stage):
        p = self._base_model.named_parameters()
        for name, para in p:
            if (("bert.encoder.layers." + str(stage)) in name) or ("predictions" in name):
                with torch.no_grad():
                    para.grad *= 0
    
    def train(self) -> None:
        for epoch in range(self._start_epoch, self._num_epochs):
            
            self._base_model.train()
                
            self._l_ntk.compute(train_loader=self._train_loader, epoch=epoch)
            self._l_ntk.flush()
                
            self._l_lr.compute(optimizer=self._optimizer, epoch=epoch)
            self._l_lr.flush()
            
            
            for i, batch in enumerate(self._train_loader):
                
                loss, _ = self._base_model(**batch)
                self._optimizer.zero_grad()
                loss.backward()
                
                self._freeze(epoch // 100)
                
                self._optimizer.step()
                self._lr_scheduler.step()
                
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
    seed = 41719
    set_seed(int(seed))
    
    
    
    b = bert_test(model_name="debug" + "_" + str(seed), config_file="/home/yfyang/t/config/bert_small.json")
    # b = bert_test(model_name=sys.argv[1] + "_" + str(seed), config_file="config/bert_small.json")
    
    # b.init_model()
    b.init_model("", "/home/yfyang/t/log/brkpt_62/pytorch_model.bin", 0)
    b.train()
