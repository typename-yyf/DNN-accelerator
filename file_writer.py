import torch
import numpy as np
import os


r'''
这个文件的作用类似于tensorboard中的writer，但是其写入的是tensor，利用add_tensor往指定文件中写入即可。
具体实例可以查看base_logger.py中l_ntk的相关代码

利用file_writer.read_file可以读取文件，具体实例可以查看最下方示例代码
'''


class file_writer():
    _datas: dict
    _directory_path: str

    def __init__(self, directory_path: str):
        self._directory_path = directory_path
        self._datas = {}

        if not os.path.exists(self._directory_path):
            os.mkdir(self._directory_path)

    @staticmethod
    def _write_tensor(t: torch.Tensor, fd):
        tensor_to_write = t.cpu().numpy()
        shape = tensor_to_write.shape
        tensor_to_write = tensor_to_write.tobytes()
        
        fd.write(len(shape).to_bytes(length=4, byteorder="little"))
        for i in shape:
            fd.write(i.to_bytes(length=4, byteorder="little"))
        fd.write(tensor_to_write)

    @staticmethod
    def _read_tensor(fd, dev: str = "cuda") -> (torch.Tensor, None):
        size_of_int = 4
        size_of_element = 4  # 默认都是float32

        shape = []
        shape_size = int.from_bytes(fd.read(size_of_int), byteorder="little")
        size = 1

        for i in range(shape_size):
            length = int.from_bytes(fd.read(size_of_int), byteorder="little")
            size *= length
            shape.append(length)

        buf = fd.read(size * size_of_element)
        if len(buf) < size * size_of_element:
            return None

        rv = np.frombuffer(buf, dtype=np.float32).reshape(shape)
        rv = torch.Tensor(rv)
        if dev == "cuda":
            rv = rv.to("cuda")
        return rv

    def add_tensor(self, name: str, t: torch.Tensor, label: int):
        fd = self._datas.get(name)
        if fd is None:
            fd = open(self._directory_path + "/" + name, "wb")
            self._datas[name] = fd

        file_writer._write_tensor(t, fd)
        fd.write(label.to_bytes(length=4, byteorder="little"))

    @staticmethod
    def read_file(file: str) -> dict:
        size_of_int: int = 4

        data: dict = {}
        fd = open(file, "rb")

        while True:
            t = file_writer._read_tensor(fd)
            if t is None:
                break

            label = int.from_bytes(fd.read(size_of_int), byteorder="little")
            data[label] = t

        return data

    def __del__(self):
        for key, value in self._datas.items():
            value.close()

if __name__ == "__main__":
    fw = file_writer("fwtest")

    x = torch.tensor([1, 2, 3, 4], dtype=torch.float32)
    y = torch.tensor([[5, 6, 7]], dtype=torch.float32)

    fw.add_tensor("test", x, 1)
    fw.add_tensor("test", y, 2)

    del fw

    d = file_writer.read_file("fwtest/test")
