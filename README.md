# HeartDeformNets

This repository contains the source code of our paper:

Fanwei Kong, Shawn Shadden, Learning Whole Heart Mesh Generation From Patient Images For Computational Simulations (2022)

<img width="1961" alt="network2" src="https://user-images.githubusercontent.com/31931939/184846001-eb3b9442-ae46-4152-a3dc-791e1ccdf946.png">


## Installation
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

You can use `predict.py` with the following arguments to generate predictions. 
```
python predict.py \
    --image /path/to/image/dir \
    --mesh_dat /path/to/the/training/data/file  \
    --swap_dat /path/to/the/biharmonic/weights/for/current/template # This is an optional the `<date>_bbw.dat` file generated from running `create_template.sh` on the training template. \
    --attr _test  \
    --mesh_tmplt  /path/to/polygon/mesh/template \
    --output /path/to/output/dir \
    --model /path/to/the/pretrained/model \
    --modality ct \
    --seg_id 1 2 3 4 5 6 7 \
    --mode test \
    --num_mesh 7 \
    --num_seg 1 \
    --num_block 3
```

Note:
- `--image`: the images should be stored under with in `<image>/ct<attr>`, thus for `--attr _test`, and `--modality ct` the image volumes should be in `image_dir_name/ct_test`. You can use `--modality ct mr' to predict both on CT and MR images where CT images are stored in `image_dir_name/ct_test` and MR images are stored in `image_dir_name/mr_test`.
- `--mesh_dat` is the `<date>_bbw.dat` file generated from running `create_template.sh` on the training template.
- `--swap_dat`is optional for providing the biharmonic coordinates corresponding to a modified template (e.g. the CFD-suitable template created from the training template). (TO-DO provide instructions on interpolating biharmonic coordinates to a modified template)
- `--seg_id` is the list of segmentation class IDs we expect the predictions to have
- The results contain deformed test template meshes from each deformation block. The final mesh from the last deformation block is `block_2_*.vtp`.

## Training

To train our network model, please run the following command. More details will be added soon.
```
python train.py \
    --im_trains /path/to/training/images \
    --im_vals /path/to/validation/images \
    --mesh /filename/of/the/training/dat/file  \
    --output /path/to/output/dir \
    --modality ct mr \
    --num_epoch 300 \
    --lr 0.001 \
    --size 128 128 128 \
    --mesh_ids 0 1 2 3 4 5 6 \
    --num_seg 1 \
    --num_block $num_block\
    --geom_weights 0.3 0.46 25 \
    --if_mask 
 ```
