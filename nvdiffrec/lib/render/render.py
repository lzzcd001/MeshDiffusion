# Copyright (c) 2020-2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved. 
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction, 
# disclosure or distribution of this material and related documentation 
# without an express license agreement from NVIDIA CORPORATION or 
# its affiliates is strictly prohibited.

import torch
import nvdiffrast.torch as dr

from . import util
from . import renderutils as ru
from . import light

# ==============================================================================================
#  Helper functions
# ==============================================================================================
def interpolate(attr, rast, attr_idx, rast_db=None):
    return dr.interpolate(attr.contiguous(), rast, attr_idx, rast_db=rast_db, diff_attrs=None if rast_db is None else 'all')

# ==============================================================================================
#  pixel shader
# ==============================================================================================
def shade(
        gb_pos,
        gb_geometric_normal,
        gb_normal,
        gb_tangent,
        gb_texc,
        gb_texc_deriv,
        view_pos,
        lgt,
        material,
        bsdf,
        xfm_lgt=None
    ):

    ################################################################################
    # Texture lookups
    ################################################################################
    perturbed_nrm = None
    alpha_mtl = None
    if 'kd_ks_normal' in material:
        # Combined texture, used for MLPs because lookups are expensive
        all_tex_jitter = material['kd_ks_normal'].sample(gb_pos + torch.normal(mean=0, std=0.01, size=gb_pos.shape, device="cuda"))
        all_tex = material['kd_ks_normal'].sample(gb_pos)
        assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
        kd, ks, perturbed_nrm = all_tex[..., :-6], all_tex[..., -6:-3], all_tex[..., -3:]
        # Compute albedo (kd) gradient, used for material regularizer
        kd_grad    = torch.sum(torch.abs(all_tex_jitter[..., :-6] - all_tex[..., :-6]), dim=-1, keepdim=True) / 3
    else:
        try:
            kd_jitter  = material['kd'].sample(gb_texc + torch.normal(mean=0, std=0.005, size=gb_texc.shape, device="cuda"), gb_texc_deriv)
            if 'alpha' in material:
                raise NotImplementedError
                try:
                    alpha_mtl = material['alpha'].sample(gb_texc, gb_texc_deriv)
                except:
                    alpha_mtl = material['alpha'].sample(gb_pos + torch.normal(mean=0, std=0.01, size=gb_pos.shape, device="cuda"))
            kd = material['kd'].sample(gb_texc, gb_texc_deriv)
            ks = material['ks'].sample(gb_texc, gb_texc_deriv)[..., 0:3] # skip alpha
            kd_grad    = torch.sum(torch.abs(kd_jitter[..., 0:3] - kd[..., 0:3]), dim=-1, keepdim=True) / 3
        except:
            kd_jitter  = kd = material['kd'].data[0].expand(*gb_pos.size())
            ks = material['ks'].data[0].expand(*gb_pos.size())[..., 0:3] # skip alpha
            kd_grad    = torch.sum(torch.abs(kd_jitter[..., 0:3] - kd[..., 0:3]), dim=-1, keepdim=True) / 3

    # Separate kd into alpha and color, default alpha = 1
    alpha = kd[..., 3:4] if kd.shape[-1] == 4 else torch.ones_like(kd[..., 0:1])
    if alpha_mtl is not None:
        alpha = alpha_mtl
    kd = kd[..., 0:3]

    ################################################################################
    # Normal perturbation & normal bend
    ################################################################################
    if 'no_perturbed_nrm' in material and material['no_perturbed_nrm']:
        perturbed_nrm = None

    use_python = (gb_tangent is None)
    
    gb_normal = ru.prepare_shading_normal(gb_pos, view_pos, perturbed_nrm, gb_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True, use_python=use_python)
    gb_geo_normal_corrected = ru.prepare_shading_normal(gb_pos, view_pos, None, gb_geometric_normal, gb_tangent, gb_geometric_normal, two_sided_shading=True, opengl=True, use_python=use_python)

    ################################################################################
    # Evaluate BSDF
    ################################################################################

    assert 'bsdf' in material or bsdf is not None, "Material must specify a BSDF type"
    bsdf = material['bsdf'] if bsdf is None else bsdf
    if bsdf == 'pbr':
        # do not use pbr
        raise NotImplementedError
        if isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_normal, kd, ks, view_pos, specular=True)
        else:
            assert False, "Invalid light type"
    elif bsdf == 'diffuse':
        if isinstance(lgt, light.EnvironmentLight):
            shaded_col = lgt.shade(gb_pos, gb_geo_normal_corrected, kd, ks, view_pos, specular=False, xfm_lgt=xfm_lgt)
        else:
            assert False, "Invalid light type"
    elif bsdf == 'normal':
        shaded_col = (gb_normal + 1.0)*0.5
    elif bsdf == 'tangent':
        shaded_col = (gb_tangent + 1.0)*0.5
    elif bsdf == 'kd':
        shaded_col = kd
    elif bsdf == 'ks':
        shaded_col = ks
    else:
        assert False, "Invalid BSDF '%s'" % bsdf
    
    nan_mask = torch.isnan(shaded_col)
    if nan_mask.any():
        raise
    if alpha is not None:
        nan_mask = torch.isnan(alpha)
        if nan_mask.any():
            raise
    
    # Return multiple buffers
    buffers = {
        'shaded'    : torch.cat((shaded_col, alpha), dim=-1),
        'kd_grad'   : torch.cat((kd_grad, alpha), dim=-1),
        'occlusion' : torch.cat((ks[..., :1], alpha), dim=-1),
        'normal'    : torch.cat((gb_normal, alpha), dim=-1),
        'depth'     : torch.cat(((gb_pos - view_pos).pow(2).sum(dim=-1, keepdim=True).sqrt(), alpha), dim=-1),
        'pos'       : torch.cat((gb_pos, alpha), dim=-1),
        'geo_normal': torch.cat((gb_geo_normal_corrected, alpha), dim=-1),
        'geo_viewdir': torch.cat((view_pos - gb_pos, alpha), dim=-1),
        'alpha'     : alpha
    }


    return buffers

# ==============================================================================================
#  Render a depth slice of the mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_layer(
        rast,
        rast_deriv,
        mesh,
        view_pos,
        lgt,
        resolution,
        spp,
        msaa,
        bsdf,
        xfm_lgt = None,
        flat_shading = False
    ):

    full_res = [resolution[0]*spp, resolution[1]*spp]

    ################################################################################
    # Rasterize
    ################################################################################

    # Scale down to shading resolution when MSAA is enabled, otherwise shade at full resolution
    if spp > 1 and msaa:
        rast_out_s = util.scale_img_nhwc(rast, resolution, mag='nearest', min='nearest')
    else:
        rast_out_s = rast

    ################################################################################
    # Interpolate attributes
    ################################################################################

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast_out_s, mesh.t_pos_idx.int())

    # Compute geometric normals. We need those because of bent normals trick (for bump mapping)
    v0 = mesh.v_pos[mesh.t_pos_idx[:, 0], :]
    v1 = mesh.v_pos[mesh.t_pos_idx[:, 1], :]
    v2 = mesh.v_pos[mesh.t_pos_idx[:, 2], :]
    face_normals = util.safe_normalize(torch.cross(v1 - v0, v2 - v0))
    face_normal_indices = (torch.arange(0, face_normals.shape[0], dtype=torch.int64, device='cuda')[:, None]).repeat(1, 3)
    gb_geometric_normal, _ = interpolate(face_normals[None, ...], rast_out_s, face_normal_indices.int())
    
    if flat_shading:
        gb_normal = mesh.f_nrm[rast_out_s[:, :, :, -1].long() - 1] # empty triangle get id=0; the first idx starts from 1
        gb_normal[rast_out_s[:, :, :, -1].long() == 0] = 0
    else:
        assert mesh.v_nrm is not None
        gb_normal, _ = interpolate(mesh.v_nrm[None, ...], rast_out_s, mesh.t_nrm_idx.int())

    if mesh.v_tng is not None:
        gb_tangent, _ = interpolate(mesh.v_tng[None, ...], rast_out_s, mesh.t_tng_idx.int()) # Interpolate tangents
    else:
        gb_tangent = None

    # Do not use texture coordinate in our case
    gb_texc, gb_texc_deriv = None, None


    ################################################################################
    # Shade
    ################################################################################
    buffers = shade(gb_pos, gb_geometric_normal, gb_normal, gb_tangent, gb_texc, gb_texc_deriv, 
        view_pos, lgt, mesh.material, bsdf, xfm_lgt=xfm_lgt)

    #### get a mask on mesh (used to identify foreground)
    mask_cont, _ = interpolate(torch.ones_like(mesh.v_pos[None, :, :1], device=mesh.v_pos.device), rast_out_s, mesh.t_pos_idx.int())
    mask = (mask_cont > 0).float()
    buffers['mask'] = mask
    buffers['mask_cont'] = mask_cont

    ################################################################################
    # Prepare output
    ################################################################################

    # Scale back up to visibility resolution if using MSAA
    if spp > 1 and msaa:
        for key in buffers.keys():
            if buffers[key] is not None:
                buffers[key] = util.scale_img_nhwc(buffers[key], full_res, mag='nearest', min='nearest')


    # Return buffers
    return buffers

# ==============================================================================================
#  Render a depth peeled mesh (scene), some limitations:
#  - Single mesh
#  - Single light
#  - Single material
# ==============================================================================================
def render_mesh(
        ctx,
        mesh,
        mtx_in,
        view_pos,
        lgt,
        resolution,
        spp         = 1,
        num_layers  = 1,
        msaa        = False,
        background  = None, 
        bsdf        = None,
        xfm_lgt     = None,
        tet_centers = None,
        flat_shading = False
    ):

    def prepare_input_vector(x):
        x = torch.tensor(x, dtype=torch.float32, device='cuda') if not torch.is_tensor(x) else x
        return x[:, None, None, :] if len(x.shape) == 2 else x
    
    def composite_buffer(key, layers, background, antialias):
        accum = background
        for buffers, rast in layers:
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
            if antialias:
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int())
            break ## HACK: the first layer only
        return accum

    def separate_buffer(key, layers, background, antialias):
        accum_list = []
        for buffers, rast in layers:
            accum = background
            alpha = (rast[..., -1:] > 0).float() * buffers[key][..., -1:]
            accum = torch.lerp(accum, torch.cat((buffers[key][..., :-1], torch.ones_like(buffers[key][..., -1:])), dim=-1), alpha)
            if antialias:
                accum = dr.antialias(accum.contiguous(), rast, v_pos_clip, mesh.t_pos_idx.int())
            accum_list.append(accum)
        return accum_list

    assert mesh.t_pos_idx.shape[0] > 0, "Got empty training triangle mesh (unrecoverable discontinuity)"
    assert background is None or (background.shape[1] == resolution[0] and background.shape[2] == resolution[1])

    full_res = [resolution[0]*spp, resolution[1]*spp]

    # Convert numpy arrays to torch tensors
    mtx_in      = torch.tensor(mtx_in, dtype=torch.float32, device='cuda') if not torch.is_tensor(mtx_in) else mtx_in
    view_pos    = prepare_input_vector(view_pos)

    # clip space transform
    v_pos_clip = ru.xfm_points(mesh.v_pos[None, ...], mtx_in)

    # Render all layers front-to-back
    with dr.DepthPeeler(ctx, v_pos_clip, mesh.t_pos_idx.int(), full_res) as peeler:
        rast, db = peeler.rasterize_next_layer()
        layers = [(render_layer(rast, db, mesh, view_pos, lgt, resolution, spp, msaa, bsdf, xfm_lgt, flat_shading), rast)]
        rast_1st_layer = rast
        # with torch.no_grad():
        if True:
            rast, db = peeler.rasterize_next_layer()
            layers2 = [(render_layer(rast, db, mesh, view_pos, lgt, resolution, spp, msaa, bsdf, xfm_lgt, flat_shading), rast)]

    # Setup background
    if background is not None:
        if spp > 1:
            background = util.scale_img_nhwc(background, full_res, mag='nearest', min='nearest')
        background = torch.cat((background, torch.zeros_like(background[..., 0:1])), dim=-1)
    else:
        background = torch.zeros(1, full_res[0], full_res[1], 4, dtype=torch.float32, device='cuda')

    # Composite layers front-to-back
    out_buffers = {}
    for key in layers[0][0].keys():
        if key == 'shaded':
            accum = composite_buffer(key, layers, background, True)
        elif (key == 'depth' or key == 'pos') and layers[0][0][key] is not None:
            accum = separate_buffer(key, layers, torch.ones_like(layers[0][0][key]) * 20.0, False)
        elif ('normal' in key) and layers[0][0][key] is not None:
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), True)
        elif layers[0][0][key] is not None:
            accum = composite_buffer(key, layers, torch.zeros_like(layers[0][0][key]), False)

        if (key == 'depth' or key == 'pos') and layers[0][0][key] is not None:
            out_buffers[key] = util.avg_pool_nhwc(accum[0], spp) if spp > 1 else accum[0]
        else:
            # Downscale to framebuffer resolution. Use avg pooling 
            out_buffers[key] = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    accum = composite_buffer('shaded', layers, background, True)
    out_buffers['shaded_second'] = util.avg_pool_nhwc(accum, spp) if spp > 1 else accum

    accum = separate_buffer('depth', layers2, -1 * torch.ones_like(layers2[0][0]['depth']), False)
    out_buffers['depth_second'] = util.avg_pool_nhwc(accum[0], spp) if spp > 1 else accum[0]


    accum = separate_buffer('normal', layers2, torch.zeros_like(layers2[0][0]['normal']), False)
    out_buffers['normal_second'] = util.avg_pool_nhwc(accum[0], spp) if spp > 1 else accum[0]

    rast_triangle_id = rast_1st_layer[:, :, :, -1].unique()
    if rast_triangle_id[0] == 0:
        if rast_triangle_id.size(0) > 1:
            rast_triangle_id = rast_triangle_id[1:] - 1 ## since by the convention of the rasterizer, 0 = empty
        else:
            rast_triangle_id = None
    out_buffers['rast_triangle_id'] = rast_triangle_id
    out_buffers['rast_depth'] = rast_1st_layer[:, :, :, -2] # z-buffer



    if tet_centers is not None:
        with torch.no_grad():
            v_pos_clip = v_pos_clip[0]
            assert full_res[0] == full_res[1]
            homo_transformed_tet_centers = ru.xfm_points(tet_centers[None, ...], mtx_in)
            transformed_tet_centers = homo_transformed_tet_centers[0, :, :3] / homo_transformed_tet_centers[0, :, 3:4]

            int_transformed_tet_centers = torch.round((transformed_tet_centers / 2.0 + 0.5) * (full_res[0] - 1)).long() # from the clip space (i.e., [-1, 1]^3) to the nearest integer coordinates in the canvas

            ### transpose THE "image"
            tmp_int_transformed_tet_centers = int_transformed_tet_centers.clone()
            int_transformed_tet_centers[:, 0] = tmp_int_transformed_tet_centers[:, 1]
            int_transformed_tet_centers[:, 1] = tmp_int_transformed_tet_centers[:, 0]


            valid_tet_centers = ((torch.logical_and((int_transformed_tet_centers <= full_res[0] - 1), int_transformed_tet_centers >= 0).float()).prod(dim=-1) == 1) # those tet centers in/on the edge of the clip space
            valid_int_transformed_tet_centers = int_transformed_tet_centers[valid_tet_centers]

            tet_center_dirs = (tet_centers - view_pos.view(1, 3))
            tet_center_depths = tet_center_dirs.pow(2).sum(-1).sqrt()

            ### Finding occluded tetrahedra
            valid_transformed_tet_center_depths = transformed_tet_centers[valid_tet_centers][:, -1] # get the depth in the clip space
            valid_tet_ids = torch.arange(tet_centers.size(0)).to(valid_tet_centers.device)[valid_tet_centers]
            

            corrected_rast_depth = out_buffers['rast_depth'].clone().detach()


            corrected_rast_depth[rast_1st_layer[:, :, :, -1] == 0] = 100 # for all pixels without any rasterized mesh, just set the depth to a large enough value

            '''
                Hacky way of finding most of the non-occluded tetrahedra (except for already rasterized ones):

                    For each pixel, find the min depth in a small neighborhood. 
                    If the center of a tetrahedron (coinciding with this pixel when rasterized) is smaller than this min depth,
                    this tetrahedron is certainly non-occluded.

                Doing this because exact per-pixel comparison for triangular meshes can be costly,
                plus we do not need to perfectly finding all visible tetrahedra.
            '''
            depth_search_range = 7 ### change this value for different resolution in rasterization
            corrected_rast_depth = -torch.nn.functional.max_pool2d(
                -corrected_rast_depth, 
                kernel_size=2*depth_search_range+1, 
                stride=1,
                padding=depth_search_range)
            
            valid_reference_depth = corrected_rast_depth[0, valid_int_transformed_tet_centers[:, 0], valid_int_transformed_tet_centers[:, 1]]
            depth_filter = valid_reference_depth >= valid_transformed_tet_center_depths


            empty_2d_mask = (rast_1st_layer[:, :, :, -1] == 0)
            empty_2d_mask = (-torch.nn.functional.max_pool2d(
                -empty_2d_mask.float(), 
                kernel_size=2*depth_search_range+1, 
                stride=1,
                padding=depth_search_range)).bool() ### similar philosophy for using a neighborhood
            empty_filter = empty_2d_mask[0, valid_int_transformed_tet_centers[:, 0], valid_int_transformed_tet_centers[:, 1]]

            ## visible tets are either determined by depth test or emptyness test
            out_buffers['visible_tet_id'] = valid_tet_ids[torch.logical_or(empty_filter, depth_filter)]

    return out_buffers

# ==============================================================================================
#  Render UVs
# ==============================================================================================
def render_uv(ctx, mesh, resolution, mlp_texture):

    # clip space transform 
    uv_clip = mesh.v_tex[None, ...]*2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())

    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos)
    assert all_tex.shape[-1] == 9 or all_tex.shape[-1] == 10, "Combined kd_ks_normal must be 9 or 10 channels"
    perturbed_nrm = all_tex[..., -3:]
    return (rast[..., -1:] > 0).float(), all_tex[..., :-6], all_tex[..., -6:-3], util.safe_normalize(perturbed_nrm)

# ==============================================================================================
#  Render UVs
# ==============================================================================================
def render_uv_nrm(ctx, mesh, resolution, mlp_texture):

    # clip space transform 
    uv_clip = mesh.v_tex[None, ...]*2.0 - 1.0

    # pad to four component coordinate
    uv_clip4 = torch.cat((uv_clip, torch.zeros_like(uv_clip[...,0:1]), torch.ones_like(uv_clip[...,0:1])), dim = -1)

    # rasterize
    rast, _ = dr.rasterize(ctx, uv_clip4, mesh.t_tex_idx.int(), resolution)

    # Interpolate world space position
    gb_pos, _ = interpolate(mesh.v_pos[None, ...], rast, mesh.t_pos_idx.int())

    # Sample out textures from MLP
    all_tex = mlp_texture.sample(gb_pos)
    perturbed_nrm = all_tex[..., -3:]
    return (rast[..., -1:] > 0).float(), util.safe_normalize(perturbed_nrm)
