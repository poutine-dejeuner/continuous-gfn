import torch

def get_mem():
    f, t = torch.cuda.mem_get_info()
    a = torch.cuda.memory_allocated()
    ma = torch.cuda.max_memory_allocated()
    print("free", f/1e9)
    print("total", t/1e9)
    print("alloc", a/1e9)
    print("max alloc", ma/1e9)
    print('\n')

get_mem()
x = torch.rand((8,)*9, device = torch.device('cuda'))

get_mem()
del x
torch.cuda.empty_cache()

get_mem()
