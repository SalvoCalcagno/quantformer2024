import torch

def compute_ssim_4d(x, y, value_range=1.0):

    bs, _, nvars, _ = x.shape
    
    x = x.permute(0, 2, 1, 3)
    y = y.permute(0, 2, 1, 3)
    
    x = x.reshape(bs, nvars, -1)
    y = y.reshape(bs, nvars, -1)
    
    return compute_ssim_3d(x, y, value_range=value_range)


def compute_ssim_3d(x, y, value_range=1.0):
    
    bs, nvars, _ = x.shape
    
    # mean
    mu_x = x.mean(dim=-1)
    mu_y = y.mean(dim=-1)
    
    # variance
    sigma_x = x.var(dim=-1)
    sigma_y = y.var(dim=-1)

    # covariance
    sigma_xy = ((x - mu_x.unsqueeze(-1)) * (y - mu_y.unsqueeze(-1))).mean(dim=-1)

    # constants
    c1 = (0.01 * value_range) ** 2
    c2 = (0.03 * value_range) ** 2
    
    return (((2*mu_x*mu_y + c1) * (2*sigma_xy + c2))/((mu_x**2 + mu_y**2 + c1)*(sigma_x + sigma_y + c2))).mean()

def compute_ssim_2d(x, y, value_range=1.0):
    
    nvars, _ = x.shape
    
    # mean
    mu_x = x.mean(dim=-1)
    mu_y = y.mean(dim=-1)
    
    # variance
    sigma_x = x.var(dim=-1)
    sigma_y = y.var(dim=-1)

    # covariance
    sigma_xy = ((x - mu_x.unsqueeze(-1)) * (y - mu_y.unsqueeze(-1))).mean(dim=-1)

    # constants
    c1 = (0.01 * value_range) ** 2
    c2 = (0.03 * value_range) ** 2
    
    return (((2*mu_x*mu_y + c1) * (2*sigma_xy + c2))/((mu_x**2 + mu_y**2 + c1)*(sigma_x + sigma_y + c2))).mean()