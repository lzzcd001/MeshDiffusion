# MeshDiffusion: Score-based Generative 3D Mesh Modeling (ICLR 2023 Spotlight)

![MeshDiffusion Teaser](/assets/mesh_teaser.png)

This is the official implementation of MeshDiffusion.

MeshDiffusion is a diffusion model for generating 3D meshes with a direct parametrization of deep marching tetrahedra (DMTet). Please refer to https://meshdiffusion.github.io for more details.


![MeshDiffusion Pipeline](/assets/meshdiffusion_pipeline.png)

## Getting Started

### Requirements

- Python >= 3.8
- CUDA 11.6
- Pytorch >= 1.6
- Pytorch3D


Install https://github.com/NVlabs/nvdiffrec

### Pretrained Models

Download the files from 

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
python eval.py --config $DMTET_CONFIG --sample-path $SAMPLE_PATH
```

where `$SAMPLE_PATH` is the generated sample `.npy` file in `$OUTPUT_PATH`.


### Single-view Conditional Generation

First fit a DMTet from a single view of a mesh

```
cd nvdiffrec
python fit_singleview.py --mesh-path $MESH_PATH --angle-ind $ANGLE_IND --out-dir $OUT_DIR --validate $VALIDATE
```

Then use the trained diffusion model to complete the occluded regions

```
cd ..
python main_diffusion.py --mode=cond_gen --config=$DIFFUSION_CONFIG \
--config.eval.eval_dir=$EVAL_DIR \
--config.eval.ckpt_path=$CKPT_PATH \
--config.eval.partial_dmtet_path=$OUT_DIR/tets/dmtet.pt \
--config.eval.tet_path=$TET_PATH
--config.eval.batch_size=$EVAL_BATCH_SIZE
```

Now visualize the completed meshes

```
cd nvdiffrec
python eval.py --config $DMTET_CONFIG --sample-path $SAMPLE_PATH
```

## Training

For ShapeNet, first create a list of paths of all ground-truth meshes and store them as a json file under `./nvdiffrec/data/shapenet_json`.

Then run the following

```
cd nvdiffrec
python fit_dmtets.py --config $DMTET_CONFIG --out-dir $DMTET_DATA_PATH
```



Create a meta file for diffusion model training:

```
cd ../metadata/
python save_meta.py --data_path $DMTET_DATA_PATH/tets --json_path $META_FILE
```

Train a diffusion model

```
cd ..
python main_diffusion.py --mode=train --config=$DIFFUSION_CONFIG \
--config.data.meta_path=$META_FILE
--config.data.filter_meta_path=$TRAIN_SPLIT_FILE
```

## Texture Completion

Follow the instructions in https://github.com/TEXTurePaper/TEXTurePaper and create text-conditioned textures for the generated meshes.

## Acknowledgement

This repo is adapted from https://github.com/NVlabs/nvdiffrec and https://github.com/yang-song/score_sde_pytorch.