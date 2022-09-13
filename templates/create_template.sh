output_dir=train_dat/wh_noerode
seg_fn=segmentation/wh.nii.gz
target_node_num=3260
num_handles_arr=(75 600)
num_mesh=7

shape_deform_dir=bc/build

mkdir -p $output_dir
# Step 1: create template from segmentation
python create_template.py \
    --seg_fn $seg_fn \
    --target_node_num $target_node_num \
    --output $output_dir/template.obj \
    --if_turn_off_erode \
    --binary

##### Step 2: pre-process template for biharmonic coordinate calculation
python clean_multi_component_to_single.py --fn $output_dir/template.obj --output $output_dir/template_merged.obj --decimate_rate 0.

ctrl_pt_arr=()
weight_arr=()
COUNTER=0
for num_handles in "${num_handles_arr[@]}"
do
    ##### Step 3: sample control handles using FSP
    $shape_deform_dir/sample_ctrl_pts $output_dir/template_merged.obj $output_dir/template_merged_${num_handles}pts.obj ${num_handles}
    #####Step 4: compute biharmonic coordiantes
    $shape_deform_dir/bc \
        $output_dir/template_manifold.obj \
        q0.5a5e-7YV \
        $output_dir/template_merged_${num_handles}pts.obj  \
        $output_dir/template_manifold_${num_handles}pts.csv \
        $output_dir/template_merged_${num_handles}pts_adjusted.obj \
        $output_dir/template_manifold_tetra_node_${num_handles}pts.csv \
        $output_dir/template_manifold_tetra_elem_${num_handles}pts.csv

    $shape_deform_dir/bc_interp \
        $output_dir/template_manifold_${num_handles}pts.csv \
        $output_dir/template_manifold_tetra_node_${num_handles}pts.csv \
        $output_dir/template_manifold_tetra_elem_${num_handles}pts.csv \
        $output_dir/template.obj \
        $output_dir/template_${num_handles}pts.csv
    ctrl_pt_arr+=($output_dir/template_merged_${num_handles}pts_adjusted.obj)
    weight_arr+=($output_dir/template_${num_handles}pts.csv)
    COUNTER=$((COUNTER + 1))
done

# Step 5: create dat file for training
echo $ctrl_pt_arr
echo $weight_arr
python make_mesh_info_dat.py --tmplt_fn $output_dir/template.vtp --sample_fn $output_dir/template.vtp --weight_fns $weight_arr --ctrl_fns $ctrl_pt_arr --out_dir $output_dir --num_mesh $num_mesh --center_coords_fn meshes/vessel_centers.yaml
