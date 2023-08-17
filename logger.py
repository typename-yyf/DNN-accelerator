from torch.nn import Module
from torch.utils.tensorboard import SummaryWriter
from accelerate import Accelerator
from io import FileIO

r'''
这个文件定义了logger，用于计算并储存或输出模型训练过程中产生的一些参数，如loss，λmax等。
如果要计算某一参数如loss，在base_loggers.py中定义一个类，继承于logger。
其初始化参数包括：
    model：该logger用于哪个模型
    writer：该logger用于输出的通道，一般为tensorboard的writer或文件输出，如果想定义其他输出方式也可
    accelerator：加速器，如果不需要多卡训练不用管
每个子类必须实现compute与flush两个方法，分别用于计算参数，以及将参数输出至writer中
具体实例可以查看base_loggers.py中的文件

flush与compute分离的原因是因为一些参数可能需要多次更新，这两者的参数根据需要自行指定
'''

class logger:
    _model: Module
    _writer: FileIO|SummaryWriter
    _accelerator: Accelerator
    
    def __init__(self, model: Module, writer: FileIO|SummaryWriter, accelerator: Accelerator=None):
        self._model = model
        self._writer = writer
        self._accelerator = accelerator
    
    def compute(self, *args, **largs):
        raise Exception("logger: Unfinished method 'compute' in", self.__str__())
    
    def flush(self, *args, **largs):
        raise Exception("logger: Unfinished method 'flush' in", self.__str__())