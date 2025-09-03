import torch

from functools import partial
from torch.distributions.gamma import Gamma
from einops import rearrange


def anneal_dsm_score_estimation(scorenet, x, labels=None, loss_type='a', hook=None, cond=None, cond_mask=None, gamma=False, L1=False, all_frames=False):

    net = scorenet.module if hasattr(scorenet, 'module') else scorenet
    version = getattr(net, 'version', 'SMLD').upper()
    net_type = getattr(net, 'type') if isinstance(getattr(net, 'type'), str) else 'v1'

    if all_frames:
        x = torch.cat([x, cond], dim=1)
        cond = None

    # z, perturbed_x
    if version == "SMLD":
        sigmas = net.sigmas
        if labels is None:
            labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)
        used_sigmas = sigmas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        z = torch.randn_like(x)
        perturbed_x = x + used_sigmas * z
    elif version == "DDPM" or version == "DDIM" or version == "FPNDM":
        alphas = net.alphas
        if labels is None:
            labels = torch.randint(0, len(alphas), (x.shape[0],), device=x.device)
        used_alphas = alphas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        if gamma:
            used_k = net.k_cum[labels].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
            used_theta = net.theta_t[labels].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
            z = Gamma(used_k, 1 / used_theta).sample()
            z = (z - used_k*used_theta) / (1 - used_alphas).sqrt()
        else:
            z = torch.randn_like(x)
        perturbed_x = used_alphas.sqrt() * x + (1 - used_alphas).sqrt() * z
    scorenet = partial(scorenet, cond=cond)

    # Loss
    if L1:
        def pow_(x):
            return x.abs()
    else:
        def pow_(x):
            return 1 / 2. * x.square()
    loss = pow_((z - scorenet(perturbed_x, labels, cond_mask=cond_mask)).reshape(len(x), -1)).sum(dim=-1)

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)


def anneal_dsm_score_estimation_dwp(scorenet, dnet, x, labels=None, loss_type='a', hook=None, cond=None, cond_mask=None, gamma=False, L1=False, all_frames=False, c_per_frame=8):

    pnet = scorenet.module if hasattr(scorenet, 'module') else scorenet
    version = getattr(pnet, 'version', 'SMLD').upper()
    net_type = getattr(pnet, 'type') if isinstance(getattr(pnet, 'type'), str) else 'v1'

    d_in = rearrange(cond, 'b (t c) h w -> b t c h w', c=c_per_frame)
    d_out, _ = dnet(d_in)
    d_out = rearrange(d_out, 'b t c h w -> b (t c) h w')

    # debug wheather dnet is correct
    # from matplotlib import pyplot as plt
    # temp = d_out[0,0,:,:].cpu()
    # plt.imshow(temp)
    # plt.savefig('experiments/debug/d_out.png')
    # plt.close()
    # temp = x[0,0,:,:].cpu()
    # plt.imshow(temp)
    # plt.savefig('experiments/debug/x.png')
    # plt.close()
    # temp = cond[0,0,:,:].cpu()
    # plt.imshow(temp)
    # plt.savefig('experiments/debug/cond.png')
    # plt.close()
    
    # 原本的x与d_out的残差r作为新的target
    x = x - d_out

    # temp = x[0,0,:,:].cpu()
    # plt.imshow(temp)
    # plt.savefig('experiments/debug/x_residual.png')
    # plt.close()

    if all_frames:
        x = torch.cat([x, cond], dim=1)
        cond = None
    
    # z, perturbed_x
    if version == "SMLD":
        sigmas = pnet.sigmas
        if labels is None:
            labels = torch.randint(0, len(sigmas), (x.shape[0],), device=x.device)
        used_sigmas = sigmas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        z = torch.randn_like(x)
        perturbed_x = x + used_sigmas * z
    elif version == "DDPM" or version == "DDIM" or version == "FPNDM":
        alphas = pnet.alphas
        if labels is None:
            labels = torch.randint(0, len(alphas), (x.shape[0],), device=x.device)
        used_alphas = alphas[labels].reshape(x.shape[0], *([1] * len(x.shape[1:])))
        if gamma:
            used_k = pnet.k_cum[labels].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
            used_theta = pnet.theta_t[labels].reshape(x.shape[0], *([1] * len(x.shape[1:]))).repeat(1, *x.shape[1:])
            z = Gamma(used_k, 1 / used_theta).sample()
            z = (z - used_k*used_theta) / (1 - used_alphas).sqrt()
        else:
            z = torch.randn_like(x)
        perturbed_x = used_alphas.sqrt() * x + (1 - used_alphas).sqrt() * z
    scorenet = partial(scorenet, cond=cond)

    # Loss
    if L1:
        def pow_(x):
            return x.abs()
    else:
        def pow_(x):
            return 1 / 2. * x.square()
    loss = pow_((z - scorenet(perturbed_x, labels, cond_mask=cond_mask)).reshape(len(x), -1)).sum(dim=-1)

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)
