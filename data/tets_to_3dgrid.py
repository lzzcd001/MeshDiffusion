import numpy as np
import torch
import os
import tqdm
import argparse

def tet_to_grids(vertices, values_list, grid_size):
    grid = torch.zeros(4, grid_size, grid_size, grid_size, device=vertices.device)
    with torch.no_grad():
        for k, values in enumerate(values_list):
            if k == 0:
                grid[k, vertices[:, 0], vertices[:, 1], vertices[:, 2]] = values.squeeze()
            else:
                grid[1:, vertices[:, 0], vertices[:, 1], vertices[:, 2]] = values.transpose(0, 1)
    return grid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='nvdiffrec')
    parser.add_argument('-res', '--resolution', type=int)
    parser.add_argument('-ss', '--split-size', type=int, default=int(1e8))
    parser.add_argument('-ind', '--index', type=int)
    parser.add_argument('-r', '--root', type=str)
    parser.add_argument('-s', '--source', type=str)
    parser.add_argument('-t', '--target', type=str)
    FLAGS = parser.parse_args()

    tet_path = f'../nvdiffrec/data/tets/{FLAGS.resolution}_tets_cropped.npz'
    tet = np.load(tet_path)
    vertices = torch.tensor(tet['vertices'])
    vertices_unique = vertices[:].unique()
    dx = vertices_unique[1] - vertices_unique[0]
    vertices_discretized = (torch.round(
        (vertices - vertices.min()) / dx)
    ).long()

    save_folder = FLAGS.root

    grid_folder = os.path.join(save_folder, FLAGS.target)
    os.makedirs(grid_folder, exist_ok=True)

    tets_folder = os.path.join(save_folder, FLAGS.source)

    for k in tqdm.trange(FLAGS.split_size):
        global_index = k + FLAGS.index * FLAGS.split_size
        tet_path = os.path.join(tets_folder, 'dmt_dict_{:05d}.pt'.format(global_index))
        if os.path.exists(tet_path):
            tet = torch.load(tet_path, map_location="cpu")
            grid = tet_to_grids(vertices_discretized, (tet['sdf'].unsqueeze(-1), tet['deform']), FLAGS.resolution)
            torch.save(grid, os.path.join(grid_folder, 'grid_{:05d}.pt'.format(global_index)))