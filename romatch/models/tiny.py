import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import torch
from pathlib import Path
import math
import numpy as np
from torch import nn
from PIL import Image
from torchvision.transforms import ToTensor
from romatch.utils.kde import kde
import time

import pos_embed_fast_cuda

class BasicLayer(nn.Module):
    """
        Basic Convolutional Layer: Conv2d -> BatchNorm -> ReLU
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bias=False, relu = True):
        super().__init__()
        self.layer = nn.Sequential(
                                        nn.Conv2d( in_channels, out_channels, kernel_size, padding = padding, stride=stride, dilation=dilation, bias = bias),
                                        nn.BatchNorm2d(out_channels, affine=False),
                                        nn.ReLU(inplace = True) if relu else nn.Identity()
                                    )

    def forward(self, x):
        return self.layer(x)

class TinyRoMa(nn.Module):
    """
        Implementation of architecture described in 
        "XFeat: Accelerated Features for Lightweight Image Matching, CVPR 2024."
    """

    def __init__(self, xfeat = None, 
                 freeze_xfeat = True, 
                 sample_mode = "threshold_balanced", 
                 symmetric = False, 
                 exact_softmax = False):
        super().__init__()
        del xfeat.heatmap_head, xfeat.keypoint_head, xfeat.fine_matcher
        if freeze_xfeat:
            xfeat.train(False)
            self.xfeat = [xfeat]# hide params from ddp
        else:
            self.xfeat = nn.ModuleList([xfeat])
        self.freeze_xfeat = freeze_xfeat
        match_dim = 256
        self.coarse_matcher = nn.Sequential(
            BasicLayer(64+64+2, match_dim,),
            BasicLayer(match_dim, match_dim,), 
            BasicLayer(match_dim, match_dim,), 
            BasicLayer(match_dim, match_dim,), 
            nn.Conv2d(match_dim, 3, kernel_size=1, bias=True, padding=0))
        fine_match_dim = 64
        self.fine_matcher = nn.Sequential(
            BasicLayer(24+24+2, fine_match_dim,),
            BasicLayer(fine_match_dim, fine_match_dim,), 
            BasicLayer(fine_match_dim, fine_match_dim,), 
            BasicLayer(fine_match_dim, fine_match_dim,), 
            nn.Conv2d(fine_match_dim, 3, kernel_size=1, bias=True, padding=0),)
        self.sample_mode = sample_mode
        self.sample_thresh = 0.05
        self.symmetric = symmetric
        self.exact_softmax = exact_softmax
        self._grid_cache = {}
    
    @property
    def device(self):
        return self.fine_matcher[-1].weight.device
    
    def get_grid(self, B, H, W, device, dtype):
        key = (H, W, device, dtype)

        if key not in self._grid_cache:
            xs = torch.linspace(
                -1 + 1 / W,
                1 - 1 / W,
                W,
                device=device,
                dtype=dtype,
            )
            ys = torch.linspace(
                -1 + 1 / H,
                1 - 1 / H,
                H,
                device=device,
                dtype=dtype,
            )

            grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
            grid = torch.stack((grid_x, grid_y), dim=-1)  # [H,W,2]
            self._grid_cache[key] = grid[None]            # [1,H,W,2]

        return self._grid_cache[key].expand(B, H, W, 2)

    def preprocess_tensor(self, x):
        """ Guarantee that image is divisible by 32 to avoid aliasing artifacts. """
        H, W = x.shape[-2:]
        _H, _W = (H//32) * 32, (W//32) * 32
        rh, rw = H/_H, W/_W

        x = F.interpolate(x, (_H, _W), mode='bilinear', align_corners=False)
        return x, rh, rw        
    
    def forward_single(self, x):
        with torch.inference_mode(self.freeze_xfeat or not self.training):
            xfeat = self.xfeat[0]
            with torch.no_grad():
                x = x.mean(dim=1, keepdim = True)
                x = xfeat.norm(x)

            #main backbone
            x1 = xfeat.block1(x)
            x2 = xfeat.block2(x1 + xfeat.skip1(x))
            x3 = xfeat.block3(x2)
            x4 = xfeat.block4(x3)
            x5 = xfeat.block5(x4)
            x4 = F.interpolate(x4, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
            x5 = F.interpolate(x5, (x3.shape[-2], x3.shape[-1]), mode='bilinear')
            feats = xfeat.block_fusion( x3 + x4 + x5 )
        if self.freeze_xfeat:
            return x2.clone(), feats.clone()
        return x2, feats

    def to_pixel_coordinates(self, coords, H_A, W_A, H_B = None, W_B = None):
        if coords.shape[-1] == 2:
            return self._to_pixel_coordinates(coords, H_A, W_A) 
        
        if isinstance(coords, (list, tuple)):
            kpts_A, kpts_B = coords[0], coords[1]
        else:
            kpts_A, kpts_B = coords[...,:2], coords[...,2:]
        return self._to_pixel_coordinates(kpts_A, H_A, W_A), self._to_pixel_coordinates(kpts_B, H_B, W_B)

    def _to_pixel_coordinates(self, coords, H, W):
        kpts = torch.stack((W/2 * (coords[...,0]+1), H/2 * (coords[...,1]+1)),axis=-1)
        return kpts
    
    def pos_embed(self, corr_volume, fast_grid_cache):
        B, H1, W1, H0, W0 = corr_volume.shape 
        if fast_grid_cache:
            grid = self.get_grid(
                B=B,
                H=H1,
                W=W1,
                device=corr_volume.device,
                dtype=corr_volume.dtype,
            ).reshape(H1*W1, 2)
        else:
            grid = torch.stack(
                    torch.meshgrid(
                        torch.linspace(-1+1/W1,1-1/W1, W1), 
                        torch.linspace(-1+1/H1,1-1/H1, H1), 
                        indexing = "xy"), 
                    dim = -1).float().to(corr_volume).reshape(H1*W1, 2)
        down = 4
        if not self.training and not self.exact_softmax:
            if fast_grid_cache:
                grid_lr = self.get_grid(
                    B=B,
                    H=H1//down,
                    W=W1//down,
                    device=corr_volume.device,
                    dtype=corr_volume.dtype,
                ).reshape(H1*W1 //down**2, 2)
            else:
                grid_lr = torch.stack(
                    torch.meshgrid(
                        torch.linspace(-1+down/W1,1-down/W1, W1//down), 
                        torch.linspace(-1+down/H1,1-down/H1, H1//down), 
                        indexing = "xy"), 
                    dim = -1).float().to(corr_volume).reshape(H1*W1 //down**2, 2)
            if fast_grid_cache:
                cv_4d = corr_volume.reshape(B, H1 * W1, H0, W0)
                best_match = cv_4d.argmax(dim=1)  # [B, H0, W0]
                cv_low = corr_volume[:, ::down, ::down].reshape(
                    B,
                    H1 * W1 // down**2,
                    H0,
                    W0,
                )
                best_score = cv_4d.gather(
                    dim=1,
                    index=best_match[:, None],
                )
                P_lowres = torch.cat((cv_low, best_score), dim=1).softmax(dim=1)
                P = P_lowres[:, :-1]
                P_best = P_lowres[:, -1:]
                P_flat = P.flatten(2).transpose(1, 2).contiguous()
                pos = torch.matmul(P_flat, grid_lr)
                pos_embeddings = pos.transpose(1, 2).reshape(B, 2, H0, W0)
                grid_flat = grid.reshape(-1, 2)   # 关键修复
                best_grid = grid_flat[best_match].permute(0, 3, 1, 2)
                pos_embeddings.add_(P_best * best_grid)
            else:
                cv = corr_volume
                best_match = cv.reshape(B,H1*W1,H0,W0).argmax(dim=1) # B, HW, H, W
                P_lowres = torch.cat((cv[:,::down,::down].reshape(B,H1*W1 // down**2,H0,W0), best_match[:,None]),dim=1).softmax(dim=1)
                pos_embeddings = torch.einsum('bchw,cd->bdhw', P_lowres[:,:-1], grid_lr)
                pos_embeddings += P_lowres[:,-1] * grid[best_match].permute(0,3,1,2)
        else:
            P = corr_volume.reshape(B,H1*W1,H0,W0).softmax(dim=1) # B, HW, H, W
            pos_embeddings = torch.einsum('bchw,cd->bdhw', P, grid)
        return pos_embeddings
    
    def visualize_warp(self, warp, certainty, im_A = None, im_B = None, 
                       im_A_path = None, im_B_path = None, symmetric = True, save_path = None, unnormalize = False):
        device = warp.device
        H,W2,_ = warp.shape
        W = W2//2 if symmetric else W2
        if im_A is None:
            from PIL import Image
            im_A, im_B = Image.open(im_A_path).convert("RGB"), Image.open(im_B_path).convert("RGB")
        if not isinstance(im_A, torch.Tensor):
            im_A = im_A.resize((W,H))
            im_B = im_B.resize((W,H))    
            x_B = (torch.tensor(np.array(im_B)) / 255).to(device).permute(2, 0, 1)
            if symmetric:
                x_A = (torch.tensor(np.array(im_A)) / 255).to(device).permute(2, 0, 1)
        else:
            if symmetric:
                x_A = im_A
            x_B = im_B
        im_A_transfer_rgb = F.grid_sample(
        x_B[None], warp[:,:W, 2:][None], mode="bilinear", align_corners=False
        )[0]
        if symmetric:
            im_B_transfer_rgb = F.grid_sample(
            x_A[None], warp[:, W:, :2][None], mode="bilinear", align_corners=False
            )[0]
            warp_im = torch.cat((im_A_transfer_rgb,im_B_transfer_rgb),dim=2)
            white_im = torch.ones((H,2*W),device=device)
        else:
            warp_im = im_A_transfer_rgb
            white_im = torch.ones((H, W), device = device)
        vis_im = certainty * warp_im + (1 - certainty) * white_im
        if save_path is not None:
            from romatch.utils import tensor_to_pil
            tensor_to_pil(vis_im, unnormalize=unnormalize).save(save_path)
        return vis_im
    
    def corr_volume(self, feat0, feat1, fast_grid_cache=False):
        """
            input:
                feat0 -> torch.Tensor(B, C, H, W)
                feat1 -> torch.Tensor(B, C, H, W)
            return:
                corr_volume -> torch.Tensor(B, H, W, H, W)
        """
        B, C, H0, W0 = feat0.shape
        B, C, H1, W1 = feat1.shape
        if fast_grid_cache:
            feat0 = feat0.flatten(2).contiguous()
            feat1 = feat1.flatten(2).contiguous()
            with torch.cuda.amp.autocast(enabled=feat0.is_cuda, dtype=torch.float16):
                corr = torch.bmm(feat1.transpose(1, 2), feat0)
                corr = corr / math.sqrt(C)
                corr_volume = corr.view(B, H1, W1, H0, W0).float()
        else:
            feat0 = feat0.view(B, C, H0*W0)
            feat1 = feat1.view(B, C, H1*W1)
            corr_volume = torch.einsum('bci,bcj->bji', feat0, feat1).reshape(B, H1, W1, H0 , W0)/math.sqrt(C) #16*16*16
        return corr_volume
    
    # @torch.inference_mode()
    # def match_from_path(self, im0_path, im1_path):
    #     device = self.device
    #     im0 = ToTensor()(Image.open(im0_path))[None].to(device)
    #     im1 = ToTensor()(Image.open(im1_path))[None].to(device)
    #     return self.match(im0, im1, batched = False)

    @torch.inference_mode()
    def match_from_path(self, im0_path, im1_path, return_timing, fast_grid_cache):
        device = self.device

        target_size = (1008, 752)  # PIL 是 (W, H)

        im0 = Image.open(im0_path).convert("RGB").resize(target_size)
        im1 = Image.open(im1_path).convert("RGB").resize(target_size)

        im0 = ToTensor()(im0)[None].to(device)
        im1 = ToTensor()(im1)[None].to(device)

        return self.match(im0, im1, batched=False, return_timing=return_timing, fast_grid_cache=fast_grid_cache)
        
    @torch.inference_mode()
    def match(self, im0, im1, *args, batched=True, return_timing=False, fast_grid_cache=False):
        if isinstance(im0, (str, Path)):
            return self.match_from_path(im0, im1, return_timing, fast_grid_cache=fast_grid_cache)
        elif isinstance(im0, Image.Image):
            batched = False
            device = self.device
            im0 = ToTensor()(im0)[None].to(device)
            im1 = ToTensor()(im1)[None].to(device)

        B, C, H0, W0 = im0.shape
        B, C, H1, W1 = im1.shape

        self.train(False)

        if im0.is_cuda:
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        corresps = self.forward(
            {"im_A": im0, "im_B": im1},
            return_timing=return_timing,
            fast_grid_cache=fast_grid_cache
        )

        if return_timing:
            corresps, timing = corresps
        else:
            timing = None

        if im0.is_cuda:
            torch.cuda.synchronize()

        t1 = time.perf_counter()

        flow = F.interpolate(
            corresps[4]["flow"],
            size=(H0, W0),
            mode="bilinear",
            align_corners=False
        ).permute(0, 2, 3, 1).reshape(B, H0, W0, 2)

        if fast_grid_cache:
            grid = self.get_grid(
                B=B,
                H=H0,
                W=W0,
                device=flow.device,
                dtype=flow.dtype,
            )
        else:
            grid = torch.stack(
                torch.meshgrid(
                    torch.linspace(-1 + 1 / W0, 1 - 1 / W0, W0),
                    torch.linspace(-1 + 1 / H0, 1 - 1 / H0, H0),
                    indexing="xy"
                ),
                dim=-1
            ).float().to(flow.device).expand(B, H0, W0, 2)

        certainty = F.interpolate(
            corresps[4]["certainty"],
            size=(H0, W0),
            mode="bilinear",
            align_corners=False
        )
        warp, cert = torch.cat((grid, flow), dim=-1), certainty[:, 0].sigmoid()

        if im0.is_cuda:
            torch.cuda.synchronize()

        t2 = time.perf_counter()

        if return_timing:
            timing["forward_total_ms"] = (t1 - t0) * 1000
            timing["match_post_ms"] = (t2 - t1) * 1000
            timing["match_total_ms"] = (t2 - t0) * 1000

        if batched:
            if return_timing:
                return warp, cert, timing
            return warp, cert
        else:
            if return_timing:
                return warp[0], cert[0], timing
            return warp[0], cert[0]

    def sample(
        self,
        matches,
        certainty,
        num=5_000,
    ):
        H,W,_ = matches.shape
        if "threshold" in self.sample_mode:
            upper_thresh = self.sample_thresh
            certainty = certainty.clone()
            certainty[certainty > upper_thresh] = 1
        matches, certainty = (
            matches.reshape(-1, 4),
            certainty.reshape(-1),
        )
        expansion_factor = 4 if "balanced" in self.sample_mode else 1
        good_samples = torch.multinomial(certainty, 
                        num_samples = min(expansion_factor*num, len(certainty)), 
                        replacement=False)
        good_matches, good_certainty = matches[good_samples], certainty[good_samples]
        if "balanced" not in self.sample_mode:
            return good_matches, good_certainty 
        use_half = True if matches.device.type == "cuda" else False
        down = 1 if matches.device.type == "cuda" else 8
        density = kde(good_matches, std=0.1, half = use_half, down = down)
        p = 1 / (density+1)
        p[density < 10] = 1e-7 # Basically should have at least 10 perfect neighbours, or around 100 ok ones
        balanced_samples = torch.multinomial(p, 
                        num_samples = min(num,len(good_certainty)), 
                        replacement=False)
        return good_matches[balanced_samples], good_certainty[balanced_samples]
        
            
    def forward(self, batch, return_timing=False, fast_grid_cache=False):
        """
            input:
                x -> torch.Tensor(B, C, H, W) grayscale or rgb images
            return:

        """
        timing = {}

        def sync():
            if im0.is_cuda:
                torch.cuda.synchronize()
        im0 = batch["im_A"]
        im1 = batch["im_B"]
        corresps = {}
        im0, rh0, rw0 = self.preprocess_tensor(im0)
        im1, rh1, rw1 = self.preprocess_tensor(im1)
        B, C, H0, W0 = im0.shape
        B, C, H1, W1 = im1.shape
        to_normalized = torch.tensor((2/W1, 2/H1, 1)).to(im0.device)[None,:,None,None]
 

        sync()
        t_xfeat0 = time.perf_counter()
        if im0.shape[-2:] == im1.shape[-2:]:
            x = torch.cat([im0, im1], dim=0)
            x = self.forward_single(x)
            feats_x0_c, feats_x1_c = x[1].chunk(2)
            feats_x0_f, feats_x1_f = x[0].chunk(2)
        else:
            feats_x0_f, feats_x0_c = self.forward_single(im0)
            feats_x1_f, feats_x1_c = self.forward_single(im1)
        sync()
        t_xfeat1 = time.perf_counter()
        if return_timing:
            timing["xfeat_ms"] = (t_xfeat1 - t_xfeat0) * 1000


        sync()
        t_corr_volume0 = time.perf_counter()
        corr_volume = self.corr_volume(feats_x0_c, feats_x1_c)
        sync()
        t_corr_volume1 = time.perf_counter()
        if return_timing:
            timing["corr_volume_ms"] = (t_corr_volume1 - t_corr_volume0) * 1000

        sync()
        t_pos0 = time.perf_counter()
        coarse_warp = self.pos_embed(corr_volume, fast_grid_cache)
        sync()
        t_pos1 = time.perf_counter()
        if return_timing:
            timing["pos_embed_ms"] = (t_pos1 - t_pos0) * 1000


        coarse_matches = torch.cat((coarse_warp, torch.zeros_like(coarse_warp[:,-1:])), dim=1)
        feats_x1_c_warped = F.grid_sample(feats_x1_c, coarse_matches.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)
        coarse_input = torch.cat((feats_x0_c, feats_x1_c_warped, coarse_warp), dim=1)

        sync()
        t_coarse0 = time.perf_counter()
        coarse_matches_delta = self.coarse_matcher(coarse_input)
        sync()
        t_coarse1 = time.perf_counter()
        if return_timing:
            timing["coarse_matcher_ms"] = (t_coarse1 - t_coarse0) * 1000

        coarse_matches = coarse_matches + coarse_matches_delta * to_normalized
        corresps[8] = {"flow": coarse_matches[:,:2], "certainty": coarse_matches[:,2:]}
        coarse_matches_up = F.interpolate(coarse_matches, size = feats_x0_f.shape[-2:], mode = "bilinear", align_corners = False)        
        coarse_matches_up_detach = coarse_matches_up.detach()#note the detach
        feats_x1_f_warped = F.grid_sample(feats_x1_f, coarse_matches_up_detach.permute(0, 2, 3, 1)[...,:2], mode = 'bilinear', align_corners = False)

        fine_input = torch.cat(
            (feats_x0_f, feats_x1_f_warped, coarse_matches_up_detach[:, :2]),
            dim=1
        )
        sync()
        t_fine0 = time.perf_counter()
        fine_matches_delta = self.fine_matcher(fine_input)
        sync()
        t_fine1 = time.perf_counter()
        if return_timing:
            timing["fine_matcher_ms"] = (t_fine1 - t_fine0) * 1000

        fine_matches = coarse_matches_up_detach+fine_matches_delta * to_normalized
        corresps[4] = {"flow": fine_matches[:,:2], "certainty": fine_matches[:,2:]}
        if return_timing:
            return corresps, timing
        return corresps