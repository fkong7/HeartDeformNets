# HeartDeformNets

This repository contains the source code of our paper:

Fanwei Kong, Shawn Shadden, Learning Whole Heart Mesh Generation From Patient Images For Computational Simulations (2022)

<img width="1961" alt="network2" src="https://user-images.githubusercontent.com/31931939/184846001-eb3b9442-ae46-4152-a3dc-791e1ccdf946.png">


## Installation
The dependencies of our implementation can be installed by running the following command.
```
pip install -r requirements.txt
```

## Evaluation
* Download [`examples`](https://drive.google.com/drive/folders/1Gk019kV9tU6LJbq3vtuHnljew-SZarjv?usp=sharing) from Drive, and place it under `HeartDeformNets`The `examples` folder contains the following files
  - `templates`: files associated with training and testing templates
    - `75_75_600_bbw.dat`: mesh data and biharmonic coordinates for a whole heart template used during training.
    - `weights/weights_gcn.hdf5`: the pre-trained weights to deform the whole heart. 
    - `four_chamber_cfd/template_600pts.csv`: the biharmonic coordinates for the test template (i.e. a simulation-ready templates of the 4 heart cambers).
    - `four_chamber_cfd/cfd_mesh.vtp`: the polygon mesh of the test template.
  - `predict_local.sh`: example bash script to deform a test mesh template to match with the example image data using the pretrained weights.
* Run the following command to generate predictions
```
python predict.py \
    --image /path/to/image/dir \
    --mesh_dat /path/to/the/training/data/file \
    --swap_dat /path/to/the/biharmonic/weights/for/current/template \
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
* The results contain deformed test template meshes from each deformation block. The final mesh from the last deformation block is `block_2_*.vtp`.

## Training

To train our network model, please run the following command. More details will be added soon.
```
python train.py \
    --im_trains /path/to/training/images \
    --im_vals /path/to/validation images \
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
