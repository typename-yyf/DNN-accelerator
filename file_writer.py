import torch
import numpy as np
import os

r'''
这个文件的作用类似于tensorboard中的writer，但是其写入的是tensor，利用add_tensor往指定文件中写入即可。
具体实例可以查看base_logger.py中l_ntk的相关代码
'''

class file_writer():
    _datas: dict
    _directory_path: str
    
    
    def __init__(self, directory_path: str):
        self._directory_path = directory_path
        self._datas = {}
        
        if not os.path.exists(self._directory_path):
            os.mkdir(self._directory_path)
        
    def add_tensor(self, name: str, t: torch.Tensor):
        fd = self._datas.get(name)
        if fd is None:
            fd = open(self._directory_path + "/" + name, "wb")
            self._datas[name] = fd
            
        tensor_to_write = t.cpu().numpy().tobytes()
        fd.write(tensor_to_write)
            
    def __del__(self):
        for key, value in self._datas.items():
            value.close()
