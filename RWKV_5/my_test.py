from my_rwkv_init import rwkv_init
import torch

model = rwkv_init()
print(model.device)
in_put = torch.ones(16,12*32,768).to("cuda")
#ones_f16 = in_put.to(torch.bfloat16)

print(in_put.shape)
out_put = model(in_put)
print(out_put.shape)