import torch, math

def GaussianFilter(depth_map, sigma=5):
    H, W = depth_map.shape
    mask = torch.zeros_like(depth_map)
    x = torch.arange(0, W, step=1)
    gau_fun = lambda x : H * math.exp(-(x - W//2)**2 / (2*sigma**2))
    for i in range(W):
        mask[H - int(round(gau_fun(i))):, i] = 1
    depth_map *= mask
    return depth_map


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    depth_map = torch.ones(size=(30, 15))
    mask = GaussianFilter(depth_map)
    # gau_fun = gau_fun.cpu().numpy()
    plt.imshow(mask)
    plt.axis('off')
    plt.show()


