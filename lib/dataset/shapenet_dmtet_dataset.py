import os
import sys
import torch
from torch.utils.data import Dataset
import json

import argparse

class ShapeNetDMTetDataset(Dataset):
    def __init__(self, root, grid_mask, deform_scale=1.0, aug=False, filter_meta_path=None, normalize_sdf=True):
        super().__init__()
        self.fpath_list = json.load(open(root, 'r'))
        self.deform_scale = deform_scale
        self.normalize_sdf = normalize_sdf
        print(f"dataset  with sdf normalized: {normalize_sdf}")
        self.coeff = torch.tensor([1.0, 1.0, self.deform_scale, self.deform_scale, self.deform_scale]).view(-1, 1, 1, 1)
        self.aug = aug
        self.grid_mask = grid_mask.cpu()
        self.resolution = self.grid_mask.size(-1)

        if filter_meta_path is not None:
            self.filter_ids = json.load(open(filter_meta_path, 'r'))
            full_id_list = [int(x.rstrip().split('_')[-1][:-3]) for i, x in enumerate(self.fpath_list)]
            fpath_idx_list = [i for i, x in enumerate(full_id_list) if x in self.filter_ids]
            self.fpath_list = [self.fpath_list[i] for i in fpath_idx_list]
    
    def __len__(self):
        return len(self.fpath_list)

    def __getitem__(self, idx):
        with torch.no_grad():
            datum = torch.load(self.fpath_list[idx], map_location='cpu')
            if self.normalize_sdf:
                sdf_sign = torch.sign(datum[:, :1])
                sdf_sign[sdf_sign == 0] = 1.0
                datum[:, :1] = sdf_sign
            if self.aug:
                nonempty_mask = (datum[1:].abs().sum(dim=0, keepdim=True) != 0)
                datum[1:] = datum[1:] + (torch.rand(3)[:, None, None, None] - 0.5) * 0.01 * nonempty_mask / (datum.size(-1) / self.resolution)

                if datum.size(-1) < self.resolution:
                    datum = datum * self.grid_mask[0, :, :datum.size(-1), :datum.size(-1), :datum.size(-1)]
                else:
                    datum = datum * self.grid_mask[0]

        if datum.size(-1) < self.resolution: 
            diff = self.resolution - datum.size(-1)
            datum = torch.nn.functional.pad(datum, (0, diff, 0, diff, 0, diff, 0, 0))
        return datum
