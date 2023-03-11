# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch

class Dataset(torch.utils.data.Dataset):
    """Basic dataset interface"""
    def __init__(self): 
        super().__init__()

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self):
        raise NotImplementedError

    def collate(self, batch):
        iter_res, iter_spp = batch[0]['resolution'], batch[0]['spp']
        res_dict = {
                'mv' : torch.cat(list([item['mv'] for item in batch]), dim=0),
                'mvp' : torch.cat(list([item['mvp'] for item in batch]), dim=0),
                'campos' : torch.cat(list([item['campos'] for item in batch]), dim=0),
                'resolution' : iter_res,
                'spp' : iter_spp,
                'img' : torch.cat(list([item['img'] for item in batch]), dim=0)
            }

        if 'spts' in batch[0]:
            res_dict['spts'] = batch[0]['spts']
        if 'vpts' in batch[0]:
            res_dict['vpts'] = batch[0]['vpts']
        if 'faces' in batch[0]:
            res_dict['faces'] = batch[0]['faces']
        if 'rast_triangle_id' in batch[0]:
            res_dict['rast_triangle_id'] = batch[0]['rast_triangle_id']
        
        if 'depth' in batch[0]:
            res_dict['depth'] = torch.cat(list([item['depth'] for item in batch]), dim=0)
        if 'normal' in batch[0]:
            res_dict['normal'] = torch.cat(list([item['normal'] for item in batch]), dim=0)
        if 'geo_normal' in batch[0]:
            res_dict['geo_normal'] = torch.cat(list([item['geo_normal'] for item in batch]), dim=0)
        if 'geo_viewdir' in batch[0]:
            res_dict['geo_viewdir'] = torch.cat(list([item['geo_viewdir'] for item in batch]), dim=0)
        if 'pos' in batch[0]:
            res_dict['pos'] = torch.cat(list([item['pos'] for item in batch]), dim=0)
        if 'mask' in batch[0]:
            res_dict['mask'] = torch.cat(list([item['mask'] for item in batch]), dim=0)
        if 'mask_cont' in batch[0]:
            res_dict['mask_cont'] = torch.cat(list([item['mask_cont'] for item in batch]), dim=0)
        if 'envlight_transform' in batch[0]:
            if batch[0]['envlight_transform'] is not None:
                res_dict['envlight_transform'] = torch.cat(list([item['envlight_transform'] for item in batch]), dim=0)
            else:
                res_dict['envlight_transform'] = None

        try:
            res_dict['depth_second'] = torch.cat(list([item['depth_second'] for item in batch]), dim=0)
        except:
            pass
        try:
            res_dict['normal_second'] = torch.cat(list([item['normal_second'] for item in batch]), dim=0)
        except:
            pass
        try:
            res_dict['img_second'] = torch.cat(list([item['img_second'] for item in batch]), dim=0)
        except:
            pass


        return res_dict