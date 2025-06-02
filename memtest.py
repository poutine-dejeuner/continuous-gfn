import torch
from icecream import ic

from nanophoto.get_trained_models import (get_cpx_fields_unet_cnn_fompred,
                                          index_double_and_frame, channel_add
                                          )
from nanophoto.models import UNet, convnet
from torch.cuda import memory_allocated as memalloc

print('test du model sequentiel unet-cnn')
m0 = memalloc()
model = get_cpx_fields_unet_cnn_fompred()
m1 = memalloc()
z = torch.rand(16, 101, 91, device=torch.device('cuda'))
y =  model(z)
m2 = memalloc()

ic(m0/1024**2, m1/1024**2, m2/1024**2)
ic((m1-m0)/1024**2)
ic((m2-m1)/1024**2)

# print('test du modele sequentiel unet-cnn avec torch.no_grad')
# m0 = memalloc()
# model = get_cpx_fields_unet_cnn_fompred()
# m1 = memalloc()
# with torch.no_grad():
#     z = torch.rand(16, 101, 91, device=torch.device('cuda'))
#     y =  model(z)
# m2 = memalloc()
# ic((m1-m0)/1024**2)
# ic((m2-m1)/1024**2)

# del model, z
# torch.cuda.empty_cache()
# ic(memalloc()/1024**2)

print('test model sequentiel en mode eval')
m0 = memalloc()
model = get_cpx_fields_unet_cnn_fompred()
model.eval()
m1 = memalloc()
z = torch.rand(16, 101, 91, device=torch.device('cuda'))
y =  model(z)
m2 = memalloc()
ic((m1-m0)/1024**2)
ic((m2-m1)/1024**2)

print('test initialisation unet et cnn separement')
m0 = memalloc()
unet = UNet(in_channels=1, out_channels=4)
m1 = memalloc()
cnn = convnet((4,))
m2 = memalloc()
ic((m1-m0)/1024**2)
ic((m2-m1)/1024**2)

print('test forward de unet et cnn separement')
m0 = memalloc()
y1 = unet(torch.rand(16, 1, 64,64))
m1 = memalloc()
y2 = cnn(torch.rand((16, 4, 190, 205)))
m2 = memalloc()
ic((m1-m0)/1024**2)
ic((m2-m1)/1024**2)

print('test forward du modele sequentiel complet mais assemblé ici')
chadd = channel_add()
indfr = index_double_and_frame()
m0 = memalloc()
model = torch.nn.Sequential(indfr, chadd, unet, cnn).eval().cuda()
m1 = memalloc()
y = model(torch.rand(4,101,91).cuda())
m2 = memalloc()
ic((m1-m0)/1024**2)
ic((m2-m1)/1024**2)

ic(torch.cuda.max_memory_allocated()/1024**2)
