from logger import *
from file_writer import file_writer
import torch
import util


r'''
这个文件定义了一些基础的logger，用于分析模型训练过程中产生的一些参数。
如果需要对模型训练过程中产生的其他参数进行分析，请在此定义。

这一部分我还会继续完善，如果有需要可以直接跑旧代码
'''

class l_train_loss(logger):
    
    _epoch: int
    _t_loss: float
    _t_batches: int
    
    def __init__(self, model: Module, writer: SummaryWriter, accelerator: Accelerator=None):
        super(l_train_loss, self).__init__(model, writer, accelerator)
        
        self._t_loss = 0
        self._t_batches = 0
        
    def compute(self, loss: torch.tensor, epoch: int):
        # 在每个batch计算完后更新一次
        self._epoch = epoch
        self._t_loss += loss.item()
        self._t_batches += 1
    
    def flush(self):
        
        train_loss = self._t_loss / self._t_batches
        self._writer.add_scalar("train_loss", train_loss, self._epoch)
        
        print("train_loss:", train_loss)
        
        self._t_loss = 0
        self._t_batches = 0
        
class l_test_loss(logger):
    
    _loss: float
    _epoch: int
    
    def __init__(self, model: Module, writer: SummaryWriter, accelerator: Accelerator=None):
        super(l_test_loss, self).__init__(model, writer, accelerator)
        
        self._loss = 0
        
    def compute(self, test_loader, epoch: int):
        for i, batch in enumerate(test_loader):
            with torch.no_grad():
                loss, _ = self._model(**batch)
                
            self._loss += loss.item()
        
        self._loss /= test_loader.dataset.__len__()
        self._epoch = epoch
            
    def flush(self):
        self._writer.add_scalar("test_loss", self._loss, self._epoch)
        
        print("test_loss:", self._loss)
        
        self._loss = 0

class l_ntk_single(logger):
    _head_lmax: torch.Tensor
    _epoch: int
    
    def __init__(self, model: Module, writer: file_writer, accelerator: Accelerator=None):
        super(l_ntk_single, self).__init__(model, writer, accelerator)
        self._epoch = 0
        
    def compute(self, train_loader, epoch: int, ker_size: int=4):
        config = self._model.config
        self._epoch = epoch
        
        head_NTKs_norm = [
            [0 for _ in range(config.num_attention_heads)] 
            for _ in range(config.num_hidden_layers)
        ]
        head_grads = [
            [[] for _ in range(config.num_attention_heads)] 
            for _ in range(config.num_hidden_layers)
        ]
    
        for i, batch in enumerate(train_loader):
            # 这里只计算一部分的kernel，太大复杂度太高无法计算
            if i >= ker_size: break
            
            batch = {k: v[:1, :] for k, v in batch.items()}
            _, logits = self._model(**batch)

            logits[0: 1].backward(torch.ones_like(logits[0: 1]))
            for name, param in self._model.named_parameters():
                # 这里只关心每个head的ntk，如果需要计算全连接层的则计算带有layer的参数
                if 'heads' in name:
                    grad = param.grad.detach().flatten()
                    splited = name.split('.')
                    l = int(splited[splited.index('layers') + 1])
                    h = int(splited[splited.index('heads') + 1])
                    head_grads[l][h].append(grad.unsqueeze(0).clone())
                
            self._model.zero_grad()
            torch.cuda.empty_cache()
        
        for l, layer in enumerate(head_grads):
            for h, head in enumerate(layer):
                J_head = torch.concat(head)
                
                # 这里可以优化一下速度
                J_head_norm = J_head.T / torch.norm(J_head.T, dim=0)
                head_NTK_norm = J_head_norm.T @ J_head_norm
                head_NTKs_norm[l][h] = head_NTK_norm.T
            
        del head_grads

        self._head_lmax = torch.stack([
            torch.stack([
                util.lmax(head_NTKs_norm[l][h]) 
                    for h in range(config.num_attention_heads)
            ])
            for l in range(config.num_hidden_layers)
        ])
        
    def flush(self):
        self._writer.add_tensor("lmax_single", self._head_lmax, self._epoch)
        self._head_lmax = None


class l_ntk(logger):
    _head_lmax: torch.Tensor
    _epoch: int
    
    def __init__(self, model: Module, writer: file_writer, accelerator: Accelerator=None):
        super(l_ntk, self).__init__(model, writer, accelerator)
        self._epoch = 0
        
    def compute(self, train_loader, epoch: int, ker_size: int=4):
        config = self._model.config
        self._epoch = epoch
        
        head_NTKs_norm = [
            [0 for _ in range(config.num_attention_heads)] 
            for _ in range(config.num_hidden_layers)
        ]
        head_grads = [
            [[] for _ in range(config.num_attention_heads)] 
            for _ in range(config.num_hidden_layers)
        ]
    
        for i, batch in enumerate(train_loader):
            torch.cuda.empty_cache()
            # 这里只计算一部分的kernel，太大复杂度太高无法计算
            if i >= ker_size: break
            
            _, logits = self._model(**batch)

            logits.backward(torch.ones_like(logits))
            for name, param in self._model.named_parameters():
                # 这里只关心每个head的ntk，如果需要计算全连接层的则计算带有layer的参数
                if 'heads' in name:
                    grad = param.grad.detach().flatten()
                    splited = name.split('.')
                    l = int(splited[splited.index('layers') + 1])
                    h = int(splited[splited.index('heads') + 1])
                    head_grads[l][h].append(grad.unsqueeze(0).clone())
                
            self._model.zero_grad()
            torch.cuda.empty_cache()
        
        for l, layer in enumerate(head_grads):
            for h, head in enumerate(layer):
                J_head = torch.concat(head)
                
                # 这里可以优化一下速度
                J_head_norm = J_head.T / torch.norm(J_head.T, dim=0)
                head_NTK_norm = J_head_norm.T @ J_head_norm
                head_NTKs_norm[l][h] = head_NTK_norm.T
            
        del head_grads

        self._head_lmax = torch.stack([
            torch.stack([
                util.lmax(head_NTKs_norm[l][h]) 
                    for h in range(config.num_attention_heads)
            ])
            for l in range(config.num_hidden_layers)
        ])
        
    def flush(self):
        self._writer.add_tensor("lmax", self._head_lmax, self._epoch)
        self._head_lmax = None
    
class l_ntk_single_head(logger):

    def __init__(
        self, 
        model: Module, writer: FileIO, accelerator: Accelerator=None,
        head: int=0, layer: int=0
    ):
        super(l_ntk_single_head, self).__init__(model, writer, accelerator)
        pass

class l_grad(logger):
    _head_max_grad: torch.Tensor
    _head_sum_grad: torch.Tensor
    _epoch: int

    def __init__(
        self, 
        model: Module, writer: file_writer, accelerator: Accelerator=None
    ):
        super(l_grad, self).__init__(model, writer, accelerator)
        
        layers = model.config.num_hidden_layers
        heads  = model.config.num_attention_heads
        
        self._head_max_grad = torch.zeros((layers, heads), dtype=torch.float32)
        self._head_sum_grad = torch.zeros((layers, heads), dtype=torch.float32)
        
        self._epoch = 0
        
    def compute(self, parameter, epoch: int):
        self._epoch = 0
        
        for name, para in parameter:
            if "heads" in name:
                grad = para.grad.detach()
                grad = torch.abs(grad)
                grad_sum = torch.sum(grad).item()
                grad_max = torch.max(grad).item()
                
                splited = name.split('.')
                l = int(splited[splited.index('layers') + 1])
                h = int(splited[splited.index('heads') + 1])
                
                self._head_sum_grad[l][h] += grad_sum
                self._head_max_grad[l][h] = max(grad_max, self._head_max_grad[l][h])
    
    def flush(self):
        self._writer.add_tensor("grad_max", self._head_max_grad, self._epoch)
        self._writer.add_tensor("grad_sum", self._head_sum_grad, self._epoch)
        
        self._head_max_grad = torch.zeros_like(self._head_max_grad)
        self._head_sum_grad = torch.zeros_like(self._head_sum_grad)

class l_hessian(logger):
    _max_eigs: torch.Tensor
    
    def __init__(self, model: Module, writer: file_writer, accelerator: Accelerator=None):
        super(l_hessian, self).__init__(model, writer, accelerator)         
        
    def compute(self, parameter, loss, n: int=100):
        params = []
        for name, p in parameter:
            if "heads" in name:
                params.append(p)
        
        V = [torch.rand_like(p, device=p.device) for p in params]
        V = [v / torch.norm(v) for v in V]
        for _ in range(n):
            self._model.zero_grad()
            grad = torch.autograd.grad(loss, params, create_graph=True)
            grad = [g / torch.norm(g) for g in grad]
            Hv = torch.autograd.grad(grad, params, V, only_inputs=True, retain_graph=True)
            self._max_eigs = torch.tensor([torch.sum(h * v) for h, v in zip(Hv, V)], device=params[0].device)
            V = [v / torch.norm(v) for v in Hv]
    def flush(self, epoch: int):
        self._writer.add_tensor("hessian_lmax", self._max_eigs, epoch)

class l_learning_rate(logger):
    
    _epoch: int
    _lr: float
    
    def __init__(self, model: Module, writer: SummaryWriter, accelerator: Accelerator=None):
        super(l_learning_rate, self).__init__(model, writer, accelerator)
        
    def compute(self, optimizer, epoch: int):
        self._lr = optimizer.param_groups[0]["lr"]
        self._epoch = epoch
        
    def flush(self):
        self._writer.add_scalar("learning_rate", self._lr, self._epoch)

r'''
    计算：|| wi - w0 ||2 / || wi ||2
'''    
class l_dis_wi_w0(logger):

    _w0: dict
    _w0_norm: float
    _ans: float
    _epoch: int
    _only_head: bool
    
    def __init__(
        self, model: Module, writer: SummaryWriter,
        accelerator: Accelerator=None, only_head: bool=True
    ):
        super(l_dis_wi_w0, self).__init__(model, writer, accelerator)
        
        self._w0_norm   = 0.0
        self._ans       = 0.0
        self._w0        = {}
        self._only_head = only_head
        
    
    def init_w0(self, parameter, max_paras: int=1):
        for name, para in parameter:
            if (not self._only_head) or ("head" in name):
                self._w0[name] = para.detach().clone()
                self._w0_norm += torch.norm(para).item()
                
                max_paras -= 1
                if max_paras <= 0:
                    break
                
    
    def compute(self, parameter, epoch: int):
        self._epoch = epoch
        
        for name, para in parameter:
            p = self._w0.get(name)
            if not p is None:
                self._ans += torch.norm((p - para.detach())).item()
                
    def flush(self):
        self._writer.add_scalar("dis_wi_w0", self._ans, self._epoch)
        self._epoch = 0
    
class l_grad_all(logger):
    _epoch: int

    def __init__(self, model: Module, writer: SummaryWriter, accelerator: Accelerator=None):
        super(l_grad_all, self).__init__(model, writer, accelerator)
        
        self._epoch = 0
        
    
    def compute(self, parameter, epoch: int):
        self._epoch = epoch
        
        i = 0
        for name, para in parameter:
            if "heads" in name:
                self._writer.add_tensor("grad_all_epoch_" + str(self._epoch), para.grad.detach(), i)
                i += 1
            
    def flush(self):
        pass
