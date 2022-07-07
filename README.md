# HeartDeformNets
## Denpendencies
python 3.7

pip install -r requirements.txt

## Prediction
* Download [`examples`](https://drive.google.com/drive/folders/1Gk019kV9tU6LJbq3vtuHnljew-SZarjv?usp=sharing) from Drive, and place it under `HeartDeformNets`The `examples` folder contains the following files
  - `images`: example CT image data
  - `templates`: files associated with training and testing templates
    - `06_27_2022_00_34_02_bbw.dat`: mesh data and biharmonic coordinates for a whole heart template used during training.
    - `weights/weights_gcn.hdf5`: the pre-trained weights to deform the whole heart. 
    - `biventricle_struct/template_600pts.csv`: the biharmonic coordinates for the test template (i.e. a biventricle mesh in this example).
    - `biventricle_struct/thickness_mesh.vtp`: the polygon mesh of the test template.
  - `predict_local.sh` example bash script to deform a test mesh template to match with the example image data using the pretrained weights
* The results contains deformed test template meshes from each deformation block. The final mesh from the last deformation block is `block_2_*.vtp`
