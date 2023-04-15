## Usage

1. Download [Blender](https://www.blender.org/download/) and unzip to `$PATH_TO_BLENDER`. We used Blender 3.3.0.
2. Clone `https://github.com/HTDerekLiu/BlenderToolbox` under `$BLENDER_PATH`
3. In `blender_script.py`, set `BLENDER_PATH` accordingly. Also set `mesh_folder_path` and `output_path` to the source mesh folder path and the desired output path.
4. Change the scale and orientation of the mesh to render in `blender_script.py` if necessary
5. Optionally, change the number of samples (`num_samples`) in `blender_viz` to balance between speed and quality
6. Run `$PATH_TO_BLENDER/blender --background --python ./blender_script.py`


## Acknowledgement

Blender scripts and settings adapted from https://github.com/HTDerekLiu/BlenderToolbox and https://www.silviasellan.com/blender_figure.html.
