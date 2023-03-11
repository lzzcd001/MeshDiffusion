# MeshDiffusion: Score-based Generative 3D Mesh Modeling (ICLR 2023 Spotlight)

![MeshDiffusion Teaser](/assets/mesh_teaser.jpg)

This is the official implementation of MeshDiffusion (https://openreview.net/forum?id=0cpM2ApF9p6).

MeshDiffusion is a diffusion model for generating 3D meshes with a direct parametrization of deep marching tetrahedra (DMTet). Please refer to https://meshdiffusion.github.io for more details.


![MeshDiffusion Pipeline](/assets/meshdiffusion_pipeline.jpg)

## Getting Started

### Requirements

- Python >= 3.8
- CUDA 11.6
- Pytorch >= 1.6
- Pytorch3D


Follow the instructions to install requirements for nvdiffrec: https://github.com/NVlabs/nvdiffrec

### Pretrained Models

Download the model checkpoints from https://drive.google.com/drive/folders/15IjbUM1tQf8gS0YsRqY5ZbMs-leJgoJ0?usp=sharing.

## Inference

### Unconditional Generation

Run the following

```
python main_diffusion.py --config=$DIFFUSION_CONFIG --mode=uncond_gen \
--config.eval.eval_dir=$OUTPUT_PATH \
--config.eval.ckpt_path=$CKPT_PATH
```

Later run

```
cd nvdiffrec
python eval.py --config $DMTET_CONFIG --out-dir $OUT_DIR --sample-path $SAMPLE_PATH --deform-scale $DEFORM_SCALE [--angle-ind $ANGLE_INDEX]
```

where `$SAMPLE_PATH` is the generated sample `.npy` file in `$OUTPUT_PATH`, and `$DEFORM_SCALE` is the scale of deformation of tet vertices set for the DMTet dataset (we use 3.0 for resolution 64 as default; change the value for your own datasets). Change `$ANGLE_INDEX` to some number from 0 to 50 if images rendered from different angles are desired.

A mesh file (`.obj`) will be saved to the folder, which can be viewed in tools such as MeshLab. The saved images are rendered from raw meshes without post-processing and thus are used for fast sanity check only.


### Single-view Conditional Generation

First fit a DMTet from a single view of a mesh

```
cd nvdiffrec
python fit_singleview.py --config $DMTET_CONFIG --mesh-path $MESH_PATH --angle-ind $ANGLE_IND --out-dir $OUT_DIR --validate $VALIDATE
```

where `$ANGLE_IND` is an integer (0 to 50) controlling the z-axis rotation of the object. Set `$VALIDATE` to 1 if visualization of fitted DMTets is needed.

Then use the trained diffusion model to complete the occluded regions

```
cd ..

python main_diffusion.py --mode=cond_gen --config=$DIFFUSION_CONFIG \
--config.eval.eval_dir=$EVAL_DIR \
--config.eval.ckpt_path=$CKPT_PATH \
--config.eval.partial_dmtet_path=$OUT_DIR/tets/dmtet.pt \
--config.eval.tet_path=$TET_PATH \
--config.eval.batch_size=$EVAL_BATCH_SIZE
```

, in which `$TET_PATH` is the uniform tetrahedral grid (of resolution 64 or 128) file in `nvdiffrec/data/tets`.

Now store the completed meshes as `.obj` files in `$SAMPLE_PATH`

```
cd nvdiffrec
python eval.py --config $DMTET_CONFIG --sample-path $SAMPLE_PATH  --deform-scale $DEFORM_SCALE
```

Caution: the deformation scale should be consistent for single view fitting and the diffusion model. Check before you run conditional generation.



## Training

For ShapeNet, first create a list of paths of all ground-truth meshes and store them as a json file under `./nvdiffrec/data/shapenet_json`.

Then run the following

```
cd nvdiffrec
python fit_dmtets.py --config $DMTET_CONFIG --out-dir $DMTET_DATA_PATH --index 0 --split-size 100000
```

where `split_size` is set to any large number greater than the dataset size. In case of batch fitting with multiple jobs, change `split_size` to a suitable number and assign a different `index` for different jobs.

Create a meta file of all dmtet grid file locations for diffusion model training:

```
cd ../metadata/
python save_meta.py --data_path $DMTET_DATA_PATH/tets --json_path $META_FILE
```

Train a diffusion model

```
cd ..

python main_diffusion.py --mode=train --config=$DIFFUSION_CONFIG \
--config.data.meta_path=$META_FILE \
--config.data.filter_meta_path=$TRAIN_SPLIT_FILE
```

where `$TRAIN_SPLIT_FILE` is a json list of indices to be included in the training set. Examples in `metadata/train_split/`.

## Texture Generation

Follow the instructions in https://github.com/TEXTurePaper/TEXTurePaper and create text-conditioned textures for the generated meshes.

## Others

If tetrahedral grids of higher resolutions are needed, first follow the README in `nvdiffrec/data/tets` and use quartet (https://github.com/crawforddoran/quartet) to generate a uniform tetrahedral grid. Then run `nvdiffrec/data/tets/crop_tets.py` to remove the boundary (so that translational symmetry holds in the resulted grid).

## Citation
If you find our work useful to your research, please consider citing:

```
@InProceedings{Liu2023MeshDiffusion,
    title={MeshDiffusion: Score-based Generative 3D Mesh Modeling},
    author={Zhen Liu and Yao Feng and Michael J. Black and Derek Nowrouzezahrai and Liam Paull and Weiyang Liu},
    booktitle={International Conference on Learning Representations},
    year={2023},
    url={https://openreview.net/forum?id=0cpM2ApF9p6}
}
```

## Acknowledgement

This repo is adapted from https://github.com/NVlabs/nvdiffrec and https://github.com/yang-song/score_sde_pytorch.
