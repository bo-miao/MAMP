import torch
import torch.nn.functional as F
import numpy as np
from scipy import interpolate
from PIL import Image
import torch.nn as nn
from torch.autograd import Variable

class InputPadder:
    """ Pads images such that dimensions are divisible by 8 """
    def __init__(self, dims, r=8, mode='sintel'):
        self.ht, self.wd = dims[-2:]
        pad_ht = (((self.ht // r) + 1) * r - self.ht) % r
        pad_wd = (((self.wd // r) + 1) * r - self.wd) % r
        if mode == 'sintel':
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, pad_ht//2, pad_ht - pad_ht//2]
        else:
            self._pad = [pad_wd//2, pad_wd - pad_wd//2, 0, pad_ht]

    def pad(self, *inputs):
        return [F.pad(x, self._pad, mode='replicate') for x in inputs]

    def unpad(self,x):
        ht, wd = x.shape[-2:]
        c = [self._pad[2], ht-self._pad[3], self._pad[0], wd-self._pad[1]]
        return x[..., c[0]:c[1], c[2]:c[3]]

def forward_interpolate(flow):
    flow = flow.detach().cpu().numpy()
    dx, dy = flow[0], flow[1]

    ht, wd = dx.shape
    x0, y0 = np.meshgrid(np.arange(wd), np.arange(ht))

    x1 = x0 + dx
    y1 = y0 + dy
    
    x1 = x1.reshape(-1)
    y1 = y1.reshape(-1)
    dx = dx.reshape(-1)
    dy = dy.reshape(-1)

    valid = (x1 > 0) & (x1 < wd) & (y1 > 0) & (y1 < ht)
    x1 = x1[valid]
    y1 = y1[valid]
    dx = dx[valid]
    dy = dy[valid]

    flow_x = interpolate.griddata(
        (x1, y1), dx, (x0, y0), method='nearest', fill_value=0)

    flow_y = interpolate.griddata(
        (x1, y1), dy, (x0, y0), method='nearest', fill_value=0)

    flow = np.stack([flow_x, flow_y], axis=0)
    return torch.from_numpy(flow).float()


def bilinear_sampler(img, coords, mode='bilinear', mask=False):
    """ Wrapper for grid_sample, uses pixel coordinates """
    # (BHW, C, H, W)
    H, W = img.shape[-2:]
    # (BHW, 2r+1, 2r+1, 1), Normalize to -1~1
    xgrid, ygrid = coords.split([1,1], dim=-1)
    xgrid = 2*xgrid/(W-1) - 1
    ygrid = 2*ygrid/(H-1) - 1
    # (BHW, 2r+1, 2r+1, 2)
    grid = torch.cat([xgrid, ygrid], dim=-1)
    img = F.grid_sample(img, grid, align_corners=True)

    if mask:
        mask = (xgrid > -1) & (ygrid > -1) & (xgrid < 1) & (ygrid < 1)
        return img, mask.float()

    return img


def coords_grid(batch, ht, wd):
    coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    coords = torch.stack(coords[::-1], dim=0).float()
    return coords[None].repeat(batch, 1, 1, 1)


def upflow8(flow, mode='bilinear'):
    new_size = (8 * flow.shape[2], 8 * flow.shape[3])
    return 8 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def upflow2(flow, mode='bilinear'):
    new_size = (2 * flow.shape[2], 2 * flow.shape[3])
    return 2 * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def upflow(flow, r=8, mode='bilinear'):
    if r == 1:
        return flow

    new_size = (r * flow.shape[2], r * flow.shape[3])
    return r * F.interpolate(flow, size=new_size, mode=mode, align_corners=True)


def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None]  # N C H W


def resize_optical_flow(flow, h, w, mode='bilinear'):
    flow = flow.clone()
    _, _, flow_h, flow_w = flow.shape
    scale_h, scale_w = h/flow_h, w/flow_w
    flow = F.interpolate(flow, size=(h, w), mode=mode, align_corners=True)
    flow[:, 0] = flow[:, 0] * scale_w
    flow[:, 1] = flow[:, 1] * scale_h
    flow = torch.round(flow)
    return flow

def clamp_optical_flow(flow):
    _, _, h, w = flow.shape
    res = torch.zeros_like(flow).to(flow.device)

    res[:, 0] = torch.clamp(flow[:, 0], -1*w, w)
    res[:, 1] = torch.clamp(flow[:, 1], -1*h, h)
    res = torch.round(res)
    return res



