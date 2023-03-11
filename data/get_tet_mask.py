import numpy as np
import torch
import os
import sys
import json
import tqdm
import argparse

def tet_to_grids(vertices, grid_size):
    
    grid = torch.zeros(grid_size, grid_size, grid_size, device=vertices.device)
    with torch.no_grad():
        for i in tqdm.tqdm(range(vertices.size(0))):
            grid[vertices[i, 0].item(), vertices[i, 1].item(), vertices[i, 2].item()] = 1.0
    return grid


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--resolution', type=int)
    parser.add_argument('--tet_folder', type=str, default='../nvdiffrec/data/tets/')
    args = parser.parse_args()

    tet_path = f'{args.tet_folder}/{args.resolution}_tets_cropped.npz'
    tet = np.load(tet_path)

    vertices = torch.tensor(tet['vertices'])
    vertices_unique = vertices[:].unique()
    dx = vertices_unique[1] - vertices_unique[0]
    
    vertices_discretized = (torch.round(
        (vertices - vertices.min()) / dx)
    ).long()

    grid = tet_to_grids(vertices_discretized, args.resolution)
    torch.save(grid, f'grid_mask_{args.resolution}.pt')
