import os
import sys
import shutil

import pandas as pd
import numpy as np

spatialvideo_test_data_dir = "/data-nas/data/dataset/qunhe/SpatialVideo/test/processed_wan_vace_data/"
spatialvideo_test_split_file = "/data-nas/data/dataset/qunhe/SpatialVideo/test/processed_wan_vace_data/metadata_wan_funcontrol_captioned.csv"
dest_dir = "/data-nas/data/experiments/zhenqing/GeometryForcing/data/spatialvideo/"

os.makedirs(dest_dir, exist_ok=True)

spatialvideo_split = pd.read_csv(spatialvideo_test_split_file)


# resize video to 256x256 using imageio and save as mp4
def resize_video(input_path, output_path, resolution=(256, 256)):
    import imageio
    from PIL import Image

    reader = imageio.get_reader(input_path)
    fps = reader.get_meta_data()["fps"]
    writer = imageio.get_writer(output_path, fps=fps)

    for frame in reader:
        img = Image.fromarray(frame)
        img_resized = img.resize(resolution, Image.LANCZOS)
        writer.append_data(np.array(img_resized))

    writer.close()
    reader.close()
    
for idx in range(len(spatialvideo_split)):
    video_path = spatialvideo_split.loc[idx, "video"]
    scene_name = os.path.basename(os.path.dirname(video_path))
    
    camera_path = os.path.join(spatialvideo_test_data_dir, scene_name, "cameras.npz")
    
    dest_video_dir = os.path.join(dest_dir, "test_256")
    dest_camera_dir = os.path.join(dest_dir, "test_poses")
    os.makedirs(dest_video_dir, exist_ok=True)
    os.makedirs(dest_camera_dir, exist_ok=True)

    dest_path = os.path.join(dest_video_dir, f"{scene_name}.mp4")
    if os.path.exists(video_path):
        # shutil.copy(video_path, dest_path)
        resize_video(video_path, dest_path, resolution=(256, 256))
    else:
        print(f"Video file {video_path} does not exist.")
        
    dest_cam_path = os.path.join(dest_camera_dir, f"{scene_name}.npz")
    if os.path.exists(camera_path):
        shutil.copy(camera_path, dest_cam_path)
    else:
        print(f"Camera file {camera_path} does not exist.")
        
print(f"{len(spatialvideo_split)} videos prepared.")