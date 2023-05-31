# HeartDeformNets

This repository contains the source code of our paper:

Fanwei Kong, Shawn Shadden, Learning Whole Heart Mesh Generation From Patient Images For Computational Simulations (2022)

<img width="1961" alt="network2" src="https://user-images.githubusercontent.com/31931939/184846001-eb3b9442-ae46-4152-a3dc-791e1ccdf946.png">


## Installation
To download our source code with submodules, please run 
```
git clone --recurse-submodules https://github.com/fkong7/HeartDeformNets.git
```
The dependencies of our implementation can be installed by running the following command.
```
pip install -r requirements.txt
```
## Template Meshes

We provide the following template meshes in `templates/meshes`
- `mmwhs_template.vtp`: The whole heart mesh template used in Task1, with the blood pools of 4 chambers, aorta and pulmonary arteries.
- `wh_template.vtp`: The whole heart mesh template used in Task2, containing pulmonary veins and vena cava inlet geometries for CFD simulations. 
- `lh_template.vtp` and  `rh_template.vtp`: The left heart and the right heart mesh template constructed from `wh_template.vtp` for 4-chamber CFD simulations of cardiac flow.
- `lv_template.vtp`: The left ventricle template constructed from `mmwhs_template.vtp` for CFD simulation of left ventricle flow. 

Those whole heart templates were created from the ground truth segmentation of a training sample. We include an example segmentation we used to create `wh_template.vtp` here: `templates/segmentation/wh.nii.gz`. To construct the training template mesh as well as the associated biharmonic coordinates and mesh information required during training, you need the following steps

- Compile the C++ code for computing biharmonc coordinates in `templates/bc` by 
```
mkdir build && cd build && cmake .. && make
```
- Specify the `output_dir` and the path of the segmentation file `seg_fn` in `create_template.sh` and then 
```
source create_template.sh
```

## Evaluation

We provide the pretrained network in `pretrained`. 
- `pretrained/task1_mmwhs.hdf5` is the pretrained network used in Task1, whole heart segmentation of the MMWHS dataset. 
- `pretrained/task2_wh.hdf5` is the pretrained network used in Task2, whole heart mesh generation with inlet vessel geometries for CFD simulations. 

The config files for both tasks are stored in `config`. The first task uses a template mesh without pulmonary veins and vena cava geometries and the second task uses another template mesh with those structures so that the predictions can be used for CHD simulations. Please make sure to use the correct template mesh depending on the task. The template mesh can be generated from the previous steps using the corresponding segmentation files. After changing the pathnames in the config files, you can use `predict.py` with the following arguments to generate predictions. 
```
python predict.py --config config/task2_wh.yaml
```

Some notes about the config options:
- `--image`: the images should be stored under with in `<image>/ct<attr>`, thus for `--attr _test`, and `--modality ct` the image volumes should be in `image_dir_name/ct_test`. You can use `--modality ct mr' to predict both on CT and MR images where CT images are stored in `image_dir_name/ct_test` and MR images are stored in `image_dir_name/mr_test`.
- `--mesh_dat` is the `<date>_bbw.dat` file generated from running `create_template.sh` on the training template.
- `--swap_dat`is optional for providing the biharmonic coordinates corresponding to a modified template (e.g. the CFD-suitable template created from the training template). (TO-DO provide instructions on interpolating biharmonic coordinates to a modified template)
- `--seg_id` is the list of segmentation class IDs we expect the predictions to have
- The results contain deformed test template meshes from each deformation block. The final mesh from the last deformation block is `block_2_*.vtp`.

## Training

To train our network model, please run the following command.
```
python train.py --config config/task2_wh.yaml
 ```
