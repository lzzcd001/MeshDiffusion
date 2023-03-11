# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import os
import time
import argparse
import json
import sys

import glob
import tqdm

import numpy as np
import torch
import nvdiffrast.torch as dr
import xatlas

# Import topology / geometry trainers
from lib.geometry.dmtet import DMTetGeometry

import lib.render.renderutils as ru
from lib.render import material
from lib.render import util
from lib.render import mesh
from lib.render import texture
from lib.render import mlptexture
from lib.render import light
from lib.render import render

from pytorch3d.io import save_obj

import pymeshlab

RADIUS = 3.0

# Enable to debug back-prop anomalies
# torch.autograd.set_detect_anomaly(True)

###############################################################################
# Loss setup
###############################################################################

@torch.no_grad()
def createLoss(FLAGS):
    if FLAGS.loss == "smape":
        return lambda img, ref: ru.image_loss(img, ref, loss='smape', tonemapper='none')
    elif FLAGS.loss == "mse":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='none')
    elif FLAGS.loss == "logl1":
        return lambda img, ref: ru.image_loss(img, ref, loss='l1', tonemapper='log_srgb')
    elif FLAGS.loss == "logl2":
        return lambda img, ref: ru.image_loss(img, ref, loss='mse', tonemapper='log_srgb')
    elif FLAGS.loss == "relmse":
        return lambda img, ref: ru.image_loss(img, ref, loss='relmse', tonemapper='none')
    else:
        assert False

###############################################################################
# Mix background into a dataset image
###############################################################################

@torch.no_grad()
def prepare_batch(target, bg_type='black'):
    # assert len(target['img'].shape) == 4, "Image shape should be [n, h, w, c]"
    if bg_type == 'checker':
        background = torch.tensor(util.checkerboard(target['img'].shape[1:3], 8), dtype=torch.float32, device='cuda')[None, ...]
    elif bg_type == 'black':
        background = torch.zeros((1, target['resolution'][0], target['resolution'][1]) + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'white':
        background = torch.ones((1, target['resolution'][0], target['resolution'][1]) + (3,), dtype=torch.float32, device='cuda')
    elif bg_type == 'reference':
        background = target['img'][..., 0:3]
    elif bg_type == 'random':
        background = torch.rand((1, target['resolution'][0], target['resolution'][1]) + (3,), dtype=torch.float32, device='cuda')
    else:
        assert False, "Unknown background type %s" % bg_type

    target['mv'] = target['mv'].cuda()
    target['mvp'] = target['mvp'].cuda()
    target['campos'] = target['campos'].cuda()
    target['background'] = background

    return target

###############################################################################
# UV - map geometry & convert to a mesh
###############################################################################

@torch.no_grad()
def xatlas_uvmap(glctx, geometry, mat, FLAGS):
    eval_mesh = geometry.getMesh(mat)
    
    # Create uvs with xatlas
    v_pos = eval_mesh.v_pos.detach().cpu().numpy()
    t_pos_idx = eval_mesh.t_pos_idx.detach().cpu().numpy()
    vmapping, indices, uvs = xatlas.parametrize(v_pos, t_pos_idx)

    # Convert to tensors
    indices_int64 = indices.astype(np.uint64, casting='same_kind').view(np.int64)
    
    uvs = torch.tensor(uvs, dtype=torch.float32, device='cuda')
    faces = torch.tensor(indices_int64, dtype=torch.int64, device='cuda')

    new_mesh = mesh.Mesh(v_tex=uvs, t_tex_idx=faces, base=eval_mesh)

    mask, kd, ks, normal = render.render_uv(glctx, new_mesh, FLAGS.texture_res, eval_mesh.material['kd_ks_normal'])
    
    if FLAGS.layers > 1:
        kd = torch.cat((kd, torch.rand_like(kd[...,0:1])), dim=-1)

    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')

    new_mesh.material = material.Material({
        'bsdf'   : mat['bsdf'],
        'kd'     : texture.Texture2D(kd, min_max=[kd_min, kd_max]),
        'ks'     : texture.Texture2D(ks, min_max=[ks_min, ks_max]),
        'normal' : texture.Texture2D(normal, min_max=[nrm_min, nrm_max])
    })

    return new_mesh

###############################################################################
# Utility functions for material
###############################################################################

def initial_guess_material(geometry, mlp, FLAGS, init_mat=None):
    kd_min, kd_max = torch.tensor(FLAGS.kd_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.kd_max, dtype=torch.float32, device='cuda')
    ks_min, ks_max = torch.tensor(FLAGS.ks_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.ks_max, dtype=torch.float32, device='cuda')
    nrm_min, nrm_max = torch.tensor(FLAGS.nrm_min, dtype=torch.float32, device='cuda'), torch.tensor(FLAGS.nrm_max, dtype=torch.float32, device='cuda')
    if mlp:
        mlp_min = torch.cat((kd_min[0:3], ks_min, nrm_min), dim=0)
        mlp_max = torch.cat((kd_max[0:3], ks_max, nrm_max), dim=0)
        mlp_map_opt = mlptexture.MLPTexture3D(geometry.getAABB(), channels=9, min_max=[mlp_min, mlp_max])
        mat =  material.Material({'kd_ks_normal' : mlp_map_opt})
    else:
        # Setup Kd (albedo) and Ks (x, roughness, metalness) textures
        if FLAGS.random_textures or init_mat is None:
            num_channels = 4 if FLAGS.layers > 1 else 3
            kd_init = torch.rand(size=FLAGS.texture_res + [num_channels], device='cuda') * (kd_max - kd_min)[None, None, 0:num_channels] + kd_min[None, None, 0:num_channels]
            kd_map_opt = texture.create_trainable(kd_init , FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])

            ksR = np.random.uniform(size=FLAGS.texture_res + [1], low=0.0, high=0.01)
            ksG = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[1].cpu(), high=ks_max[1].cpu())
            ksB = np.random.uniform(size=FLAGS.texture_res + [1], low=ks_min[2].cpu(), high=ks_max[2].cpu())

            ks_map_opt = texture.create_trainable(np.concatenate((ksR, ksG, ksB), axis=2), FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])
        else:
            kd_map_opt = texture.create_trainable(init_mat['kd'], FLAGS.texture_res, not FLAGS.custom_mip, [kd_min, kd_max])
            ks_map_opt = texture.create_trainable(init_mat['ks'], FLAGS.texture_res, not FLAGS.custom_mip, [ks_min, ks_max])

        # Setup normal map
        if FLAGS.random_textures or init_mat is None or 'normal' not in init_mat:
            normal_map_opt = texture.create_trainable(np.array([0, 0, 1]), FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])
        else:
            normal_map_opt = texture.create_trainable(init_mat['normal'], FLAGS.texture_res, not FLAGS.custom_mip, [nrm_min, nrm_max])

        mat = material.Material({
            'kd'     : kd_map_opt,
            'ks'     : ks_map_opt,
            'normal' : normal_map_opt
        })

    if init_mat is not None:
        mat['bsdf'] = init_mat['bsdf']
    else:
        mat['bsdf'] = 'pbr'

    return mat

###############################################################################
# Validation & testing
###############################################################################

def rotate_scene(FLAGS, itr):
    fovy = np.deg2rad(45)
    cam_radius = RADIUS
    proj_mtx = util.perspective(fovy, FLAGS.display_res[1] / FLAGS.display_res[0], FLAGS.cam_near_far[0], FLAGS.cam_near_far[1])

    # Smooth rotation for display.
    ang    = (itr / 50) * np.pi * 2
    mv     = util.translate(0, 0, -cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
    mvp    = proj_mtx @ mv
    campos = torch.linalg.inv(mv)[:3, 3]

    res_dict = {
        'mv': mv[None, ...].cuda(), 
        'mvp': mvp[None, ...].cuda(), 
        'campos': campos[None, ...].cuda(),
        'spp': 1,
        'resolution': FLAGS.display_res
    }

    return res_dict

def validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS):
    result_dict = {}
    with torch.no_grad():
        lgt.build_mips()
        if FLAGS.camera_space_light:
            lgt.xfm(target['mv'])

        buffers = geometry.render(glctx, target, lgt, opt_material)

        result_dict['opt'] = util.rgb_to_srgb(buffers['shaded'][...,0:3])[0]
        result_image = result_dict['opt']
   
        return result_image, result_dict

def validate(glctx, geometry, opt_material, lgt, dataset_validate, out_dir, FLAGS):

    # ==============================================================================================
    #  Validation loop
    # ==============================================================================================
    mse_values = []
    psnr_values = []

    dataloader_validate = torch.utils.data.DataLoader(dataset_validate, batch_size=1, collate_fn=dataset_validate.collate)

    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'metrics.txt'), 'w') as fout:
        fout.write('ID, MSE, PSNR\n')

        print("Running validation")
        for it, target in enumerate(dataloader_validate):

            # Mix validation background
            target = prepare_batch(target, FLAGS.background)

            result_image, result_dict = validate_itr(glctx, target, geometry, opt_material, lgt, FLAGS)
           
            # Compute metrics
            opt = torch.clamp(result_dict['opt'], 0.0, 1.0) 
            ref = torch.clamp(result_dict['ref'], 0.0, 1.0)

            mse = torch.nn.functional.mse_loss(opt, ref, size_average=None, reduce=None, reduction='mean').item()
            mse_values.append(float(mse))
            psnr = util.mse_to_psnr(mse)
            psnr_values.append(float(psnr))

            line = "%d, %1.8f, %1.8f\n" % (it, mse, psnr)
            fout.write(str(line))

            for k in result_dict.keys():
                np_img = result_dict[k].detach().cpu().numpy()
                util.save_image(out_dir + '/' + ('val_%06d_%s.png' % (it, k)), np_img)

        avg_mse = np.mean(np.array(mse_values))
        avg_psnr = np.mean(np.array(psnr_values))
        line = "AVERAGES: %1.4f, %2.3f\n" % (avg_mse, avg_psnr)
        fout.write(str(line))
        print("MSE,      PSNR")
        print("%1.8f, %2.3f" % (avg_mse, avg_psnr))
    return avg_psnr

###############################################################################
# Main shape fitter function / optimization loop
###############################################################################

class Trainer(torch.nn.Module):
    def __init__(self, glctx, geometry, lgt, mat, optimize_geometry, optimize_light, image_loss_fn, FLAGS):
        super(Trainer, self).__init__()

        self.glctx = glctx
        self.geometry = geometry
        self.light = lgt
        self.material = mat
        self.optimize_geometry = optimize_geometry
        self.optimize_light = optimize_light
        self.image_loss_fn = image_loss_fn
        self.FLAGS = FLAGS

        if not self.optimize_light:
            with torch.no_grad():
                self.light.build_mips()

        self.params = list(self.material.parameters())
        self.params += list(self.light.parameters()) if optimize_light else []
        self.geo_params = list(self.geometry.parameters()) if optimize_geometry else []

    def forward(self, target, it):
        if self.optimize_light:
            self.light.build_mips()
            if self.FLAGS.camera_space_light:
                self.light.xfm(target['mv'])

        return self.geometry.tick(glctx, target, self.light, self.material, self.image_loss_fn, it)

#----------------------------------------------------------------------------
# Main function.
#----------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('--config', type=str, default=None, help='Config file')
    parser.add_argument('-i', '--iter', type=int, default=5000)
    parser.add_argument('-b', '--batch', type=int, default=1)
    parser.add_argument('-s', '--spp', type=int, default=1)
    parser.add_argument('-l', '--layers', type=int, default=1)
    parser.add_argument('-r', '--train-res', nargs=2, type=int, default=[512, 512])
    parser.add_argument('-dr', '--display-res', type=int, default=None)
    parser.add_argument('-tr', '--texture-res', nargs=2, type=int, default=[1024, 1024])
    parser.add_argument('-di', '--display-interval', type=int, default=0)
    parser.add_argument('-si', '--save-interval', type=int, default=1000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=0.01)
    parser.add_argument('-mr', '--min-roughness', type=float, default=0.08)
    parser.add_argument('-mip', '--custom-mip', action='store_true', default=False)
    parser.add_argument('-rt', '--random-textures', action='store_true', default=False)
    parser.add_argument('-bg', '--background', default='checker', choices=['black', 'white', 'checker', 'reference'])
    parser.add_argument('-o', '--out-dir', type=str, default='./viz_tet')
    parser.add_argument('-sp', '--sample-path', type=str, default=None)
    parser.add_argument('-bm', '--base-mesh', type=str, default=None)
    parser.add_argument('-ds', '--deform-scale', type=float, default=2.0)
    parser.add_argument('-vn', '--viz-name', type=str, default='viz')
    parser.add_argument('--unnormalized_sdf', action="store_true")
    parser.add_argument('--validate', type=bool, default=True)
    parser.add_argument('--angle-ind', type=int, default=25, help='z-axis rotation of the object, from 0 to 50')
    
    FLAGS = parser.parse_args()

    FLAGS.mtl_override        = None                     # Override material of model
    FLAGS.dmtet_grid          = 64                       # Resolution of initial tet grid. We provide 64 and 128 resolution grids. Other resolutions can be generated with https://github.com/crawforddoran/quartet
    FLAGS.mesh_scale          = 2.1                      # Scale of tet grid box. Adjust to cover the model
    FLAGS.env_scale           = 1.0                      # Env map intensity multiplier
    FLAGS.envmap              = None                     # HDR environment probe
    FLAGS.display             = None                     # Conf validation window/display. E.g. [{"relight" : <path to envlight>}]
    FLAGS.camera_space_light  = False                    # Fixed light in camera space. This is needed for setups like ethiopian head where the scanned object rotates on a stand.
    FLAGS.lock_light          = False                    # Disable light optimization in the second pass
    FLAGS.lock_pos            = False                    # Disable vertex position optimization in the second pass
    FLAGS.sdf_regularizer     = 0.2                      # Weight for sdf regularizer (see paper for details)
    FLAGS.laplace             = "relative"               # Mesh Laplacian ["absolute", "relative"]
    FLAGS.laplace_scale       = 10000.0                  # Weight for sdf regularizer. Default is relative with large weight
    FLAGS.pre_load            = True                     # Pre-load entire dataset into memory for faster training
    FLAGS.kd_min              = [ 0.0,  0.0,  0.0,  0.0] # Limits for kd
    FLAGS.kd_max              = [ 1.0,  1.0,  1.0,  1.0]
    FLAGS.ks_min              = [ 0.0, 0.08,  0.0]       # Limits for ks
    FLAGS.ks_max              = [ 1.0,  1.0,  1.0]
    FLAGS.nrm_min             = [-1.0, -1.0,  0.0]       # Limits for normal map
    FLAGS.nrm_max             = [ 1.0,  1.0,  1.0]
    FLAGS.cam_near_far        = [0.1, 1000.0]
    FLAGS.learn_light         = False
    FLAGS.cropped             = False
    FLAGS.random_lgt          = False

    if FLAGS.config is not None:
        data = json.load(open(FLAGS.config, 'r'))
        for key in data:
            FLAGS.__dict__[key] = data[key]

    if FLAGS.display_res is None:
        FLAGS.display_res = FLAGS.train_res

    os.makedirs(FLAGS.out_dir, exist_ok=True)
    viz_path = os.path.join(FLAGS.out_dir, 'viz')
    mesh_path = os.path.join(FLAGS.out_dir, 'mesh')
    os.makedirs(viz_path, exist_ok=True)
    os.makedirs(mesh_path, exist_ok=True)

    glctx = dr.RasterizeGLContext()

    # ==============================================================================================
    #  Create env light with trainable parameters
    # ==============================================================================================

    lgt = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)

    # ==============================================================================================
    #  If no initial guess, use DMtets to create geometry
    # ==============================================================================================

    # Setup geometry for optimization
    resolution = FLAGS.dmtet_grid
    geometry = DMTetGeometry(
        resolution, FLAGS.mesh_scale, FLAGS, 
        deform_scale=FLAGS.deform_scale
    )

    mask = torch.load(f'../data/grid_mask_{resolution}.pt').view(1, resolution, resolution, resolution).to("cuda")


    ### compute the mapping from tet indices to 3D cubic grid vertex indices
    tet_path = FLAGS.tet_path
    tet = np.load(tet_path)
    vertices = torch.tensor(tet['vertices'])
    vertices_unique = vertices[:].unique()
    dx = vertices_unique[1] - vertices_unique[0]

    vertices_discretized = (torch.round(
        (vertices - vertices.min()) / dx)
    ).long()
            
    data_all = np.load(FLAGS.sample_path)
    print('shape of generated data', data_all.shape)

    for no_data in tqdm.trange(data_all.shape[0]):

        grid = torch.tensor(data_all[no_data])
        if FLAGS.unnormalized_sdf:
            raise NotImplementedError
            geometry.sdf.data[:] = (
                    grid[0, vertices_discretized[:, 0], vertices_discretized[:, 1], vertices_discretized[:, 2]]
                ).cuda()
        else:
            geometry.sdf.data[:] = torch.sign(
                    grid[0, vertices_discretized[:, 0], vertices_discretized[:, 1], vertices_discretized[:, 2]]
                ).cuda()
        geometry.deform.data[:] = (
                grid[1:, vertices_discretized[:, 0], vertices_discretized[:, 1], vertices_discretized[:, 2]]
            ).cuda().transpose(0, 1)

        geometry.deform.data[:] = geometry.deform.data[:].clip(-1.0, 1.0)

        ### mtl for visualization
        opt_material = {
            'name' : '_default_mat',
            # 'bsdf' : 'pbr',
            'bsdf' : 'diffuse',
            'kd'   : texture.Texture2D(torch.tensor([0.75, 0.3, 0.6], dtype=torch.float32, device='cuda')),
            'ks'   : texture.Texture2D(torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device='cuda'))
        }

        ### create and optimize mesh
        base_mesh = geometry.getMesh(opt_material)
        
        
        ### save image (before post-processing)
        v_pose = rotate_scene(FLAGS, FLAGS.angle_index) ## pick a pose (pose # from 0 to 50)
        result_image, _ = validate_itr(glctx, prepare_batch(v_pose, FLAGS.background), geometry, opt_material, lgt, FLAGS)
        result_image = result_image.detach().cpu().numpy()
        util.save_image(os.path.join(viz_path, ('%s_%06d.png' % (FLAGS.viz_name, no_data))), result_image)


        ### save post-processed mesh
        mesh_savepath = os.path.join(mesh_path, '{:06d}.obj'.format(no_data))
        save_obj(
            verts=base_mesh.v_pos,
            faces=base_mesh.t_pos_idx,
            f=mesh_savepath
        )

        ms = pymeshlab.MeshSet()
        ms.load_new_mesh(mesh_savepath)
        ms.meshing_isotropic_explicit_remeshing()
        ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, cotangentweight=False)
        # ms.apply_coord_laplacian_smoothing(stepsmoothnum=3, cotangentweight=True) ## for smoother surface
        ms.meshing_isotropic_explicit_remeshing()
        ms.apply_filter_script()
        ms.save_current_mesh(mesh_savepath)