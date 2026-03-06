echo "Running evaluation for DFoT on SpatialVideo dataset"
output_dir="test_results/spatialvideo"
eval_result_dir="$output_dir/dfot_video_pose"
checkpoint_path="checkpoints/DFoT_16f_state_dict.ckpt"

algorithm="dfot_video_pose"

echo "Result directory: $eval_result_dir"
echo "Checkpoint path: $checkpoint_path" 

python -m main +name=single_image_to_long dataset=spatialvideo \
        algorithm=$algorithm experiment=video_generation \
        @diffusion/continuous \
        load=$checkpoint_path \
        'experiment.tasks=[validation]' experiment.validation.data.shuffle=False experiment.test.data.shuffle=False \
        dataset.context_length=1 dataset.frame_skip=1 dataset.n_frames=81 \
        algorithm.tasks.prediction.keyframe_density=0.197 \
        algorithm.tasks.interpolation.max_batch_size=4 experiment.validation.batch_size=1 \
        algorithm.tasks.prediction.history_guidance.name=stabilized_vanilla \
        +algorithm.tasks.prediction.history_guidance.guidance_scale=4.0 \
        +algorithm.tasks.prediction.history_guidance.stabilization_level=0.02  \
        algorithm.tasks.interpolation.history_guidance.name=vanilla \
        +algorithm.tasks.interpolation.history_guidance.guidance_scale=1.5 \
        'algorithm.logging.metrics=[fvd,fid,psnr,lpips,ssim]' \
        hydra.run.dir=$eval_result_dir 
