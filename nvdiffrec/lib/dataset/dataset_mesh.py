# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import numpy as np
import torch
import sys

from ..render import util
from ..render import mesh
from ..render import render
from ..render import light

from .dataset import Dataset

import kaolin

###############################################################################
# Reference dataset using mesh & rendering
###############################################################################

class DatasetMesh(Dataset):

    def __init__(self, ref_mesh, glctx, cam_radius, FLAGS, validate=False):
        # Init 
        self.glctx              = glctx
        self.cam_radius         = cam_radius
        self.FLAGS              = FLAGS
        self.validate           = validate
        self.fovy               = np.deg2rad(45)
        self.aspect             = FLAGS.train_res[1] / FLAGS.train_res[0]
        self.random_lgt         = FLAGS.random_lgt
        self.camera_lgt         = False
        self.flat_shading       = FLAGS.dataset_flat_shading
        

        if self.FLAGS.local_rank == 0:
            print(f"use flag shading {FLAGS.dataset_flat_shading}")
            print("DatasetMesh: ref mesh has %d triangles and %d vertices" % (ref_mesh.t_pos_idx.shape[0], ref_mesh.v_pos.shape[0]))

        # Sanity test training texture resolution
        ref_texture_res = np.maximum(ref_mesh.material['kd'].getRes(), ref_mesh.material['ks'].getRes())
        if 'normal' in ref_mesh.material:
            ref_texture_res = np.maximum(ref_texture_res, ref_mesh.material['normal'].getRes())
        if self.FLAGS.local_rank == 0 and FLAGS.texture_res[0] < ref_texture_res[0] or FLAGS.texture_res[1] < ref_texture_res[1]:
            print("---> WARNING: Picked a texture resolution lower than the reference mesh [%d, %d] < [%d, %d]" % (FLAGS.texture_res[0], FLAGS.texture_res[1], ref_texture_res[0], ref_texture_res[1]))

        print("Loading env map")
        sys.stdout.flush()
        # Load environment map texture
        self.envlight = light.load_env(FLAGS.envmap, scale=FLAGS.env_scale)
        
        print("Computing tangents")
        sys.stdout.flush()
        try:
            self.ref_mesh = mesh.compute_tangents(ref_mesh)
        except Exception as e:
            print(e)
            print("Continue without tangents...")
            self.ref_mesh = ref_mesh

    def _rotate_scene(self, itr):
        proj_mtx = util.perspective(self.fovy, self.FLAGS.display_res[1] / self.FLAGS.display_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # Smooth rotation for display.
        ang    = (itr / 50) * np.pi * 2
        mv     = util.translate(0, 0, -self.cam_radius) @ (util.rotate_x(-0.4) @ util.rotate_y(ang))
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), self.FLAGS.display_res, self.FLAGS.spp

    def _random_scene(self):
        # ==============================================================================================
        #  Setup projection matrix
        # ==============================================================================================
        iter_res = self.FLAGS.train_res
        proj_mtx = util.perspective(self.fovy, iter_res[1] / iter_res[0], self.FLAGS.cam_near_far[0], self.FLAGS.cam_near_far[1])

        # ==============================================================================================
        #  Random camera & light position
        # ==============================================================================================

        # Random rotation/translation matrix for optimization.
        mv     = util.translate(0, 0, -self.cam_radius) @ util.random_rotation_translation(0.2)
        mvp    = proj_mtx @ mv
        campos = torch.linalg.inv(mv)[:3, 3]

        return mv[None, ...].cuda(), mvp[None, ...].cuda(), campos[None, ...].cuda(), iter_res, self.FLAGS.spp # Add batch dimension

    def __len__(self):
        return 50 if self.validate else (self.FLAGS.iter + 1) * self.FLAGS.batch

    def __getitem__(self, itr):
        # ==============================================================================================
        #  Randomize scene parameters
        # ==============================================================================================

        if self.validate:
            mv, mvp, campos, iter_res, iter_spp = self._rotate_scene(itr)
            camera_mv = None
        else:
            mv, mvp, campos, iter_res, iter_spp = self._random_scene()
            if self.random_lgt:
                rnd_rot = util.random_rotation()
                camera_mv = rnd_rot.unsqueeze(0).clone()
            elif self.camera_lgt:
                camera_mv = mv.clone()
            else:
                camera_mv = None



        with torch.no_grad():
            render_out = render.render_mesh(self.glctx, self.ref_mesh, mvp, campos, self.envlight, iter_res, spp=iter_spp, 
                                num_layers=self.FLAGS.layers, msaa=True, background=None, xfm_lgt=camera_mv, flat_shading=self.flat_shading)
            img = render_out['shaded']
            img_second = render_out['shaded_second']
            normal = render_out['normal']
            depth = render_out['depth']
            geo_normal = render_out['geo_normal']
            pos = render_out['pos']

            sample_points = torch.tensor(kaolin.ops.mesh.sample_points(self.ref_mesh.v_pos.unsqueeze(0), self.ref_mesh.t_pos_idx, 50000)[0][0])
            vertex_points = self.ref_mesh.v_pos
        
        return_dict = {
            'mv' : mv,
            'mvp' : mvp,
            'campos' : campos,
            'resolution' : iter_res,
            'spp' : iter_spp,
            'img' : img,
            'img_second' : img_second,
            'spts': sample_points,
            'vpts': vertex_points,
            'faces': self.ref_mesh.t_pos_idx,
            'depth': depth,
            'normal': normal,
            'geo_normal': geo_normal,
            'geo_viewdir': render_out['geo_viewdir'],
            'pos': pos,
            'envlight_transform': camera_mv,
            'mask': render_out['mask'],
            'mask_cont': render_out['mask_cont'],
            'rast_triangle_id': render_out['rast_triangle_id']
        }

        try:
            return_dict['depth_second'] = render_out['depth_second']
        except:
            pass

        try:
            return_dict['normal_second'] = render_out['normal_second']
        except:
            pass
        return return_dict
