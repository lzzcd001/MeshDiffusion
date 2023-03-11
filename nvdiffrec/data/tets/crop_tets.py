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
import argparse
from collections import defaultdict


def crop_tets(vertices, indices):
    assert indices.shape[1] == 4
    vertices_cropped = np.array(vertices)
    mask = None
    for k in range(3):
        if mask is None:
            mask = (vertices[:, k] != np.min(vertices[:, k])) & (vertices[:, k] != np.max(vertices[:, k]))
        else:
            mask = (vertices[:, k] != np.min(vertices[:, k])) & (vertices[:, k] != np.max(vertices[:, k])) & mask
        print(f"remaining: {mask.sum()} out of {vertices.shape[0]}")
    
    vertices_cropped = vertices[mask]

    vert_inds = np.arange(vertices.shape[0])
    vert_inds_unused_mask = (1.0 - mask).astype(np.bool)
    verts_inds_unused = vert_inds[vert_inds_unused_mask]

    print(f"{verts_inds_unused.shape[0]} out of {vertices.shape[0]}")

    remapping = defaultdict(lambda : -1)
    count = 0
    for i in range(vertices.shape[0]):
        if mask[i]:
            remapping[i] = count
            count += 1

    indices_cropped = np.zeros_like(indices, dtype=np.int32)
    count = 0
    for i in range(indices.shape[0]):
        flag = True
        tmp = np.zeros((4,))
        for k in range(4):
            if remapping[indices[i, k]] == -1:
                flag = False
                break
            else:
                tmp[k] = remapping[indices[i, k]]

        if flag:
            indices_cropped[count, :] = tmp[:]
            count += 1
        
        if i % 1000 == 0:
            print(f"iter {i} / {indices.shape[0]}")

    print(vertices_cropped.shape[0], np.min(indices_cropped), np.max(indices_cropped))
    indices_cropped = indices_cropped[:count]
    return vertices_cropped, indices_cropped


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int)
    args = parser.parse_args()

    resolution = args.resolution
    npzfile = f'{resolution}_tets.npz'
    data = np.load(npzfile)
    new_verts, new_inds = crop_tets(data['vertices'], data['indices'])
    np.savez_compressed(f'{resolution}_tets_cropped.npz', vertices=new_verts, indices=new_inds)