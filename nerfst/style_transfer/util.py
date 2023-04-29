import torch
import torch.nn as nn
import numpy as np

def calc_histogram_loss(A, B, histogram_block):
    input_hist = histogram_block(A)
    target_hist = histogram_block(B)
    histogram_loss = (1/np.sqrt(2.0) * (torch.sqrt(torch.sum(
        torch.pow(torch.sqrt(target_hist) - torch.sqrt(input_hist), 2)))) / 
        input_hist.shape[0])

    return histogram_loss
    
    
# B, C, H, W; mean var on HW
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def cosine_dismat(A, B):
    A = A.view(A.shape[0], A.shape[1], -1)
    B = B.view(B.shape[0], B.shape[1], -1)

    A_norm = torch.sqrt((A**2).sum(1))
    B_norm = torch.sqrt((B**2).sum(1))

    A = (A/A_norm.unsqueeze(dim=1).expand(A.shape)).permute(0,2,1)
    B = (B/B_norm.unsqueeze(dim=1).expand(B.shape))
    dismat = 1.-torch.bmm(A, B) 

    return dismat


def calc_remd_loss(A, B):
    C = cosine_dismat(A, B)
    m1, _ = C.min(1)
    m2, _ = C.min(2)
    
    remd = torch.max(m1.mean(), m2.mean())

    return remd

def calc_ss_loss(A, B):
    MA = cosine_dismat(A, A)
    MB = cosine_dismat(B, B)
    Lself_similarity = torch.abs(MA-MB).mean() 

    return Lself_similarity


def calc_moment_loss(A, B):
    A = A.view(A.shape[0], A.shape[1], -1)
    B = B.view(B.shape[0], B.shape[1], -1)

    mu_a = torch.mean(A, 1, keepdim=True)
    mu_b = torch.mean(B, 1, keepdim=True)
    mu_d = torch.abs(mu_a - mu_b).mean()

    A_c = A - mu_a
    B_c = B - mu_b
    cov_a = torch.bmm(A_c, A_c.permute(0,2,1)) / (A.shape[2]-1)
    cov_b = torch.bmm(B_c, B_c.permute(0,2,1)) / (B.shape[2]-1)
    cov_d = torch.abs(cov_a - cov_b).mean()
    loss = mu_d + cov_d
    return loss


def calc_mse_loss(A, B):
    return nn.MSELoss(A, B)