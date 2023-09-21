import torch
import torch.nn as nn

def gaussian_kernel(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    # print(n_samples)
    total = torch.cat([source, target], dim=0)
    
    total = total.view(n_samples, -1)  # Reshape tensor to [batch_size, -1]
    
    total0 = total.unsqueeze(0).expand(n_samples, n_samples, -1)
    total1 = total.unsqueeze(1).expand(n_samples, n_samples, -1)
    L2_distance = ((total0 - total1) ** 2).sum(2)
    
    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul ** i) for i in range(kernel_num)]
    
    kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
    return sum(kernel_val)

def mmd_rbf(source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    source = source.squeeze(0).squeeze(0).permute(0, 1, 2)
    target = target.squeeze(0).squeeze(0).permute(0, 1, 2)
    loss = 0
    for i in range(source.shape[0]):
        src = source[i, :, :]
        trg = target[i, :, :]
        batch_size = int(src.size()[0])
        kernels = gaussian_kernel(src, trg, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma)
        XX = kernels[:batch_size, :batch_size]
        YY = kernels[batch_size:, batch_size:]
        XY = kernels[:batch_size, batch_size:]
        YX = kernels[batch_size:, :batch_size]
        loss += torch.mean(XX + YY - XY - YX)
    return loss

# 示例
source = torch.randn(2, 256, 256)
target = torch.randn(2, 256, 256)
loss = mmd_rbf(source, target)
print(loss)



