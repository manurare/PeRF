export TIMM_FUSED_ATTN=0
export TORCH_HOME=/mnt/faster2/mra59/.cache/
export TORCH_EXTENSIONS_DIR=/mnt/faster2/mra59/.cache/torch_extensions

for scene in apartment_0 apartment_1 apartment_2 frl_apartment_0 frl_apartment_1 frl_apartment_2 frl_apartment_3 frl_apartment_4 frl_apartment_5 hotel_0 office_0 office_1 office_2 office_3 office_4 room_0 room_1 room_2 ; do
    input_image_path=example_data/replica/${scene}_lemniscate_1k_0/renders/input_rgb.jpg
    input_depth_path=example_data/replica/${scene}_lemniscate_1k_0/renders/input_depth.dpt
    expname=output/$scene
    
    python core_exp_runner.py --config-name nerf_lemniscate dataset.image_path=$input_image_path dataset.depth_path=$input_depth_path device.base_exp_dir=$expname
    python core_exp_runner.py --config-name nerf_lemniscate dataset.image_path=$input_image_path device.base_exp_dir=$expname mode=render_dense is_continue=true

    python metrics.py --exp_renders_dir ${expname}/WildDataset_renders/nerf_experiment/dense_images_new_persp/ --gt_renders_dir example_data/replica/${scene}_lemniscate_1k_0/renders
done