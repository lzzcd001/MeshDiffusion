# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import numpy as np
import torch

from ..render import mesh
from ..render import render
from ..render import regularizer


import kaolin
import pytorch3d.ops
from ..render import util as render_utils

import torch.nn.functional as F

from ..render import renderutils as ru

###############################################################################
# Marching tetrahedrons implementation (differentiable), adapted from
# https://github.com/NVIDIAGameWorks/kaolin/blob/master/kaolin/ops/conversions/tetmesh.py
###############################################################################

class DMTet:
    def __init__(self):
        self.triangle_table = torch.tensor([
                [-1, -1, -1, -1, -1, -1],
                [ 1,  0,  2, -1, -1, -1],
                [ 4,  0,  3, -1, -1, -1],
                [ 1,  4,  2,  1,  3,  4],
                [ 3,  1,  5, -1, -1, -1],
                [ 2,  3,  0,  2,  5,  3],
                [ 1,  4,  0,  1,  5,  4],
                [ 4,  2,  5, -1, -1, -1],
                [ 4,  5,  2, -1, -1, -1],
                [ 4,  1,  0,  4,  5,  1],
                [ 3,  2,  0,  3,  5,  2],
                [ 1,  3,  5, -1, -1, -1],
                [ 4,  1,  2,  4,  3,  1],
                [ 3,  0,  4, -1, -1, -1],
                [ 2,  0,  1, -1, -1, -1],
                [-1, -1, -1, -1, -1, -1]
                ], dtype=torch.long, device='cuda')

        self.num_triangles_table = torch.tensor([0,1,1,2,1,2,2,1,1,2,2,1,2,1,1,0], dtype=torch.long, device='cuda')
        self.base_tet_edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype=torch.long, device='cuda')

    ###############################################################################
    # Utility functions
    ###############################################################################

    def sort_edges(self, edges_ex2):
        with torch.no_grad():
            order = (edges_ex2[:,0] > edges_ex2[:,1]).long()
            order = order.unsqueeze(dim=1)

            a = torch.gather(input=edges_ex2, index=order, dim=1)      
            b = torch.gather(input=edges_ex2, index=1-order, dim=1)  

        return torch.stack([a, b],-1)

    def map_uv(self, faces, face_gidx, max_idx):
        N = int(np.ceil(np.sqrt((max_idx+1)//2)))
        tex_y, tex_x = torch.meshgrid(
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            torch.linspace(0, 1 - (1 / N), N, dtype=torch.float32, device="cuda"),
            indexing='ij'
        )

        pad = 0.9 / N

        uvs = torch.stack([
            tex_x      , tex_y,
            tex_x + pad, tex_y,
            tex_x + pad, tex_y + pad,
            tex_x      , tex_y + pad
        ], dim=-1).view(-1, 2)

        def _idx(tet_idx, N):
            x = tet_idx % N
            y = torch.div(tet_idx, N, rounding_mode='trunc')
            return y * N + x

        tet_idx = _idx(torch.div(face_gidx, 2, rounding_mode='trunc'), N)
        tri_idx = face_gidx % 2

        uv_idx = torch.stack((
            tet_idx * 4, tet_idx * 4 + tri_idx + 1, tet_idx * 4 + tri_idx + 2
        ), dim = -1). view(-1, 3)

        return uvs, uv_idx

    ###############################################################################
    # Marching tets implementation
    ###############################################################################

    def __call__(self, pos_nx3, sdf_n, tet_fx4):
        with torch.no_grad():
            occ_n = sdf_n > 0
            occ_fx4 = occ_n[tet_fx4.reshape(-1)].reshape(-1,4)
            occ_sum = torch.sum(occ_fx4, -1)
            valid_tets = (occ_sum>0) & (occ_sum<4)
            occ_sum = occ_sum[valid_tets]

            # find all vertices
            all_edges = tet_fx4[valid_tets][:,self.base_tet_edges].reshape(-1,2)
            all_edges = self.sort_edges(all_edges)
            unique_edges, idx_map = torch.unique(all_edges,dim=0, return_inverse=True)  
            
            unique_edges = unique_edges.long()
            mask_edges = occ_n[unique_edges.reshape(-1)].reshape(-1,2).sum(-1) == 1
            mapping = torch.ones((unique_edges.shape[0]), dtype=torch.long, device="cuda") * -1
            mapping[mask_edges] = torch.arange(mask_edges.sum(), dtype=torch.long,device="cuda")
            idx_map = mapping[idx_map] # map edges to verts

            interp_v = unique_edges[mask_edges]
        edges_to_interp = pos_nx3[interp_v.reshape(-1)].reshape(-1,2,3)
        edges_to_interp_sdf = sdf_n[interp_v.reshape(-1)].reshape(-1,2,1)
        edges_to_interp_sdf[:,-1] *= -1

        denominator = edges_to_interp_sdf.sum(1,keepdim = True)

        edges_to_interp_sdf = torch.flip(edges_to_interp_sdf, [1])/denominator
        verts = (edges_to_interp * edges_to_interp_sdf).sum(1)
        idx_map = idx_map.reshape(-1,6)

        v_id = torch.pow(2, torch.arange(4, dtype=torch.long, device="cuda"))
        tetindex = (occ_fx4[valid_tets] * v_id.unsqueeze(0)).sum(-1)
        num_triangles = self.num_triangles_table[tetindex]

        # Generate triangle indices
        faces = torch.cat((
            torch.gather(
                input=idx_map[num_triangles == 1], 
                dim=1, 
                index=self.triangle_table[tetindex[num_triangles == 1]][:, :3]
            ).reshape(-1,3),
            torch.gather(
                input=idx_map[num_triangles == 2], 
                dim=1, 
                index=self.triangle_table[tetindex[num_triangles == 2]][:, :6]
            ).reshape(-1,3),
        ), dim=0)

        # Get global face index (static, does not depend on topology)
        num_tets = tet_fx4.shape[0]
        tet_gidx = torch.arange(num_tets, dtype=torch.long, device="cuda")[valid_tets]
        face_gidx = torch.cat((
            tet_gidx[num_triangles == 1]*2,
            torch.stack((tet_gidx[num_triangles == 2]*2, tet_gidx[num_triangles == 2]*2 + 1), dim=-1).view(-1)
        ), dim=0)

        uvs, uv_idx = self.map_uv(faces, face_gidx, num_tets*2)


        face_to_valid_tet = torch.cat((
            tet_gidx[num_triangles == 1],
            torch.stack((tet_gidx[num_triangles == 2], tet_gidx[num_triangles == 2]), dim=-1).view(-1)
        ), dim=0)

        valid_vert_idx = tet_fx4[tet_gidx[num_triangles > 0]].long().unique()

        return verts, faces, uvs, uv_idx, face_to_valid_tet.long(), valid_vert_idx

###############################################################################
# Regularizer
###############################################################################

def sdf_reg_loss(sdf, all_edges):
    sdf_f1x6x2 = sdf[all_edges.reshape(-1)].reshape(-1,2)
    mask = torch.sign(sdf_f1x6x2[...,0]) != torch.sign(sdf_f1x6x2[...,1])
    sdf_f1x6x2 = sdf_f1x6x2[mask]
    sdf_diff = torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,0], (sdf_f1x6x2[...,1] > 0).float()) + \
            torch.nn.functional.binary_cross_entropy_with_logits(sdf_f1x6x2[...,1], (sdf_f1x6x2[...,0] > 0).float())
    return sdf_diff



class Buffer(object):
    def __init__(self, shape, capacity, device) -> None:
        self.len_curr = 0
        self.pointer = 0
        self.capacity = capacity
        self.buffer = torch.zeros((capacity, ) + shape, device=device)
    
    def push(self, x):
        '''
            Push one single data point into the buffer
        '''
        self.buffer[self.pointer] = x
        self.pointer = (self.pointer + 1) % self.capacity
        if self.len_curr < self.capacity:
            self.len_curr += 1
    
    def avg(self):
        return torch.sign(torch.sign(self.buffer[:self.len_curr]).float().mean(dim=0)).float()
        # return self.buffer[:self.len_curr].mean(dim=0)
        # return self.buffer[:self.len_curr][-1]

###############################################################################
#  Geometry interface
###############################################################################

class DMTetGeometry(torch.nn.Module):
    def __init__(self, grid_res, scale, FLAGS, root='./', grid_to_tet=None, deform_scale=2.0, **kwargs):
        super(DMTetGeometry, self).__init__()

        self.FLAGS         = FLAGS
        self.grid_res      = grid_res
        self.marching_tets = DMTet()
        self.cropped       = True
        self.tanh          = False
        self.deform_scale  = deform_scale

        self.grid_to_tet = grid_to_tet

        if self.cropped:
            print("use cropped tets")
            tets = np.load(os.path.join(root, 'data/tets/{}_tets_cropped.npz'.format(self.grid_res)))
        else:
            tets = np.load(os.path.join(root, 'data/tets/{}_tets.npz'.format(self.grid_res)))
        print('tet min and max', tets['vertices'].min() * scale, tets['vertices'].max() * scale)
        self.verts    = torch.tensor(tets['vertices'], dtype=torch.float32, device='cuda') * scale
        self.indices  = torch.tensor(tets['indices'], dtype=torch.long, device='cuda')
        self.generate_edges()

        # Random init
        sdf = torch.rand_like(self.verts[:,0]).clamp(-1.0, 1.0) - 0.1
        # sdf = torch.sign(sdf) * 0.1
        # sdf = self.verts.pow(2).sum(dim=-1).sqrt() - 0.5

        self.sdf    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=True)
        self.register_parameter('sdf', self.sdf)

        self.deform = torch.nn.Parameter(torch.zeros_like(self.verts), requires_grad=True)
        self.register_parameter('deform', self.deform)

        self.alpha = None

        self.sdf_ema    = torch.nn.Parameter(sdf.clone().detach(), requires_grad=False)
        self.deform_ema = torch.nn.Parameter(self.deform.clone().detach(), requires_grad=False)

        # self.ema_coeff = 0.7
        self.ema_coeff = 0.9

        self.sdf_buffer = Buffer(sdf.size(), capacity=200, device='cuda')

    def generate_edges(self):
        with torch.no_grad():
            edges = torch.tensor([0,1,0,2,0,3,1,2,1,3,2,3], dtype = torch.long, device = "cuda")
            all_edges = self.indices[:,edges].reshape(-1,2)
            all_edges_sorted = torch.sort(all_edges, dim=1)[0]
            self.all_edges = torch.unique(all_edges_sorted, dim=0)

    def getAABB(self):
        return torch.min(self.verts, dim=0).values, torch.max(self.verts, dim=0).values
    
    def getVertNNDist(self):
        v_deformed = (self.verts + 2 / (self.grid_res * 2) * torch.tanh(self.deform)).unsqueeze(0)
        return pytorch3d.ops.knn.knn_points(
                v_deformed, v_deformed, K=2
            ).dists[0, :, -1].detach() ## K=2 because dist(self, self)=0

    def getTetCenters(self):
        v_deformed = self.get_deformed() # size: N x 3
        face_verts = v_deformed[self.indices] # size: M x 4 x 3
        face_centers = face_verts.mean(dim=1) # size: M x 3

        return face_centers

    def getValidTetIdx(self):
        # Run DM tet to get a base mesh
        v_deformed = self.get_deformed()
        verts, faces, uvs, uv_idx, tet_gidx, valid_vert_idx = self.marching_tets(v_deformed, self.sdf, self.indices)
        return tet_gidx.long()

    def getValidVertsIdx(self):
        # Run DM tet to get a base mesh
        v_deformed = self.get_deformed()
        verts, faces, uvs, uv_idx, tet_gidx, valid_vert_idx = self.marching_tets(v_deformed, self.sdf, self.indices)
        return self.indices[tet_gidx.long()].unique()

    def getMesh(self, material, noise=0.0, ema=False):
        # Run DM tet to get a base mesh
        v_deformed = self.get_deformed(ema=ema)

        if ema:
            sdf = self.sdf_ema
        else:
            sdf = self.sdf

        verts, faces, uvs, uv_idx, tet_gidx, valid_vert_idx = self.marching_tets(v_deformed, sdf, self.indices)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        if material is not None:
            # Run mesh operations to generate tangent space
            imesh = mesh.auto_normals(imesh)
            imesh = mesh.compute_tangents(imesh)
        imesh.valid_vert_idx = valid_vert_idx

        return imesh

    def getMesh_no_deform(self, material, noise=0.0, ema=False):
        # Run DM tet to get a base mesh
        if ema:
            # sdf = self.sdf * (1 - self.ema_coeff) + self.sdf_ema.detach() * self.ema_coeff
            sdf = self.sdf_ema
        else:
            sdf = self.sdf

        verts, faces, uvs, uv_idx, tet_gidx, valid_vert_idx = self.marching_tets(self.verts, torch.sign(sdf), self.indices)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh

    def getMesh_no_deform_gd(self, material, noise=0.0, ema=False):
        # Run DM tet to get a base mesh
        v_deformed = self.get_deformed(no_grad=True)


        if ema:
            # sdf = self.sdf * (1 - self.ema_coeff) + self.sdf_ema.detach() * self.ema_coeff
            sdf = self.sdf_ema
        else:
            sdf = self.sdf

        verts, faces, uvs, uv_idx, tet_gidx, valid_vert_idx = self.marching_tets(v_deformed, sdf, self.indices)
        imesh = mesh.Mesh(verts, faces, v_tex=uvs, t_tex_idx=uv_idx, material=material)

        # Run mesh operations to generate tangent space
        imesh = mesh.auto_normals(imesh)
        imesh = mesh.compute_tangents(imesh)

        return imesh

    def get_deformed(self, no_grad=False, ema=False):
        if no_grad:
            deform = self.deform.detach()
        else:
            deform = self.deform

        if self.tanh:
            v_deformed = self.verts + 2 / (self.grid_res * 2) * torch.tanh(deform) * self.deform_scale
        else:
            v_deformed = self.verts + 2 / (self.grid_res * 2) * deform * self.deform_scale
        return v_deformed
    
    def get_angle(self):
        with torch.no_grad():
            comb_list = [
                (0, 1, 2, 3),
                (0, 1, 3, 2),
                (0, 2, 3, 1),
                (1, 2, 3, 0)
            ]

            directions = torch.zeros(self.indices.size(0), 4).cuda()
            dir_vec = torch.zeros(self.indices.size(0), 4, 3).cuda()
            vert_inds = torch.zeros(self.indices.size(0), 4).cuda().long()
            count = 0
            vpos_list = self.get_deformed()
            for comb in comb_list:
                face = self.indices[:, comb[:3]]
                face_pos = vpos_list[face, :]
                face_center = face_pos.mean(1, keepdim=False)
                v = self.indices[:, comb[3]]
                test_vec = vpos_list[v]
                ref_vec = render_utils.safe_normalize(vpos_list[face[:, 0]] - face_center)
                distance_vec = test_vec - render_utils.dot(test_vec, ref_vec) * ref_vec
                directions[:, count] = torch.sign(render_utils.dot(test_vec, distance_vec)[:, 0])
                dir_vec[:, count, :] = distance_vec
                vert_inds[:, count] = v
                count += 1
            return directions, dir_vec, vert_inds


    def clamp_deform(self):
        if not self.tanh:
            self.deform.data[:] = self.deform.data.clamp(-0.99, 0.99)
            self.sdf.data[:] = self.sdf.data.clamp(-1.0, 1.0)
 
    def render(self, glctx, target, lgt, opt_material, bsdf=None, ema=False, xfm_lgt=None, get_visible_tets=False):
        opt_mesh = self.getMesh(opt_material, ema=ema)
        tet_centers = self.getTetCenters() if get_visible_tets else None
        return render.render_mesh(
            glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
            msaa=True, background=target['background'], bsdf=bsdf, xfm_lgt=xfm_lgt, tet_centers=tet_centers)

    def render_with_mesh(self, glctx, target, lgt, opt_material, bsdf=None, noise=0.0, ema=False, xfm_lgt=None):
        opt_mesh = self.getMesh(opt_material, noise=noise, ema=ema)
        return opt_mesh, render.render_mesh(
            glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
            msaa=True, background=target['background'], bsdf=bsdf, xfm_lgt=xfm_lgt)
    
    def update_ema(self, ema_coeff=0.9):
        self.sdf_buffer.push(self.sdf)
        self.sdf_ema.data[:] = self.sdf_buffer.avg()
        # self.sdf_ema.data[:] = self.sdf.data[:] * (1 - ema_coeff) + self.sdf_ema.data[:] * ema_coeff
        # self.deform_ema.data[:] = self.deform.data[:] * (1 - ema_coeff) + self.deform_ema.data[:] * ema_coeff
        self.deform_ema.data[:] = self.deform.data[:]


    def render_ema(self, glctx, target, lgt, opt_material, bsdf=None, xfm_lgt=None):
        opt_mesh = self.getMesh(opt_material, ema=True)
        return render.render_mesh(
            glctx, opt_mesh, target['mvp'], target['campos'], lgt, target['resolution'], spp=target['spp'], 
            msaa=True, background=target['background'], bsdf=bsdf, xfm_lgt=xfm_lgt)

    def init_with_gt_surface(self, gt_verts, surface_faces, campos):
        with torch.no_grad():
            surface_face_verts = gt_verts[surface_faces]
            surface_centers = surface_face_verts.mean(dim=1)
            v_pos = self.get_deformed()
            results = pytorch3d.ops.knn_points(v_pos[None, ...], surface_centers[None, ...])
            dists, nn_idx = results.dists, results.idx
            displacement = (v_pos - surface_centers[nn_idx[0, :, 0]])
            view_dirs = campos - surface_centers
            normals = torch.cross(
                surface_face_verts[:, 0] - surface_face_verts[:, 1], surface_face_verts[:, 0] - surface_face_verts[:, 2])
            mask = ((normals * view_dirs).sum(dim=-1, keepdim=True) >= 0).float()
            normals = normals * mask - normals * (1 - mask)
            outside_verts_idx = ((displacement * normals[nn_idx[0, :, 0]]).sum(dim=-1) > 0)
            self.sdf.data[outside_verts_idx] = 1.0
            

    def tick(self, glctx, target, lgt, opt_material, loss_fn, iteration, with_reg=True, xfm_lgt=None, no_depth_thin=True):

        if iteration < 100:
            self.deform.requires_grad = False
            self.deform_scale = 2.0
        else:
            self.deform.requires_grad = True
            self.deform_scale = 2.0
        
        if iteration > 200 and iteration < 2000 and iteration % 20 == 0:
            with torch.no_grad():
                v_pos = self.get_deformed()
                v_pos_camera_homo = ru.xfm_points(v_pos[None, ...], target['mvp'])
                v_pos_camera = v_pos_camera_homo[:, :, :2] / v_pos_camera_homo[:, :, -1:]
                v_pos_camera_discrete = ((v_pos_camera * 0.5 + 0.5).clip(0, 1) * (target['resolution'][0] - 1)).long()
                target_mask = target['mask_cont'][:, :, :, 0] == 0
                for k in range(target_mask.size(0)):
                    assert v_pos_camera_discrete[k].min() >= 0 and v_pos_camera_discrete[k].max() < target['resolution'][0]
                    v_mask = target_mask[k, v_pos_camera_discrete[k, :, 1], v_pos_camera_discrete[k, :, 0]].view(v_pos.size(0))
                    # print(v_mask.sum())
                    self.sdf.data[v_mask] = self.sdf.data[v_mask].abs().clamp(0.0, 1.0)

        # ==============================================================================================
        #  Render optimizable object with identical conditions
        # ==============================================================================================
        imesh, buffers = self.render_with_mesh(glctx, target, lgt, opt_material, noise=0.0, xfm_lgt=xfm_lgt)

        # ==============================================================================================
        #  Compute loss
        # ==============================================================================================

        # Image-space loss, split into a coverage component and a color component
        color_ref = target['img']
        img_loss = torch.tensor(0.0).cuda()
        img_loss = torch.nn.functional.mse_loss(buffers['shaded'][..., 3:], color_ref[..., 3:])
        img_loss = img_loss + loss_fn(buffers['shaded'][..., 0:3] * color_ref[..., 3:], color_ref[..., 0:3] * color_ref[..., 3:])
        mask = (target['mask_cont'][:, :, :, 0] == 1.0).float()
        mask_curr = (buffers['mask_cont'][:, :, :, 0] == 1.0).float()

        if iteration % 300 == 0 and iteration < 1790:
            self.deform.data[:] *= 0.4

        if no_depth_thin:
            valid_depth_mask = (
                (target['depth_second'] >= 0).float() * ((target['depth_second'] - target['depth']).abs() >= 5e-3).float()
            ).detach()
        else:
            valid_depth_mask = 1.0
        

        depth_diff = (buffers['depth'][:, :, :, :1] - target['depth'][:, :, :, :1]).abs() * mask.unsqueeze(-1) * valid_depth_mask
        l1_loss_mask = (depth_diff < 1.0).float()
        img_loss = img_loss + (l1_loss_mask * depth_diff + (1 - l1_loss_mask) * depth_diff.pow(2)).mean() * 100.0

        reg_loss = torch.tensor(0.0).cuda()

        # SDF regularizer
        iter_thres = 0
        sdf_weight = self.FLAGS.sdf_regularizer - (self.FLAGS.sdf_regularizer - 0.01) * min(1.0, 4.0 * ((iteration - iter_thres) / (self.FLAGS.iter - iter_thres)))

        sdf_mask = torch.zeros_like(self.sdf, device=self.sdf.device)
        sdf_mask[imesh.valid_vert_idx] = 1.0
        sdf_masked = self.sdf.detach() * sdf_mask + self.sdf * (1 - sdf_mask)
        reg_loss = sdf_reg_loss(sdf_masked, self.all_edges).mean() * sdf_weight * 2.5 # Dropoff to 0.01


        # Albedo (k_d) smoothnesss regularizer
        reg_loss += torch.mean(buffers['kd_grad'][..., :-1] * buffers['kd_grad'][..., -1:]) * 0.03 * min(1.0, iteration / 500)

        # Visibility regularizer
        reg_loss += torch.mean(buffers['occlusion'][..., :-1] * buffers['occlusion'][..., -1:]) * 1e0 * min(1.0, iteration / 500)

        pred_points = kaolin.ops.mesh.sample_points(imesh.v_pos.unsqueeze(0), imesh.t_pos_idx, 50000)[0][0]
        target_pts = target['spts']
        chamfer = kaolin.metrics.pointcloud.chamfer_distance(pred_points.unsqueeze(0), target_pts.unsqueeze(0)).mean()
        reg_loss += chamfer


        return img_loss, reg_loss
