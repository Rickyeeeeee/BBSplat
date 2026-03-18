#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
from argparse import ArgumentParser

CUDA_DEVICE_ID = 3

parser = ArgumentParser(description="Full evaluation script parameters")
parser.add_argument("--skip_training", action="store_true")
parser.add_argument("--skip_rendering", action="store_true")
parser.add_argument("--skip_metrics", action="store_true")
parser.add_argument("--skip_memory", action="store_true")
parser.add_argument("--output_path", default="eval/mipnerf360")
parser.add_argument("--mb", type=int, default=100)
args, _ = parser.parse_known_args()
mb = args.mb

points = int(float(mb) * 1000000. / 4. /1081.)
print(f"points: {points}")

mipnerf360_path = "/workspace/data/MipNerf360"
tnt_path = "/workspace/data/tandt_db/tandt"
db_path = "/workspace/data/tandt_db/db"

mipnerf360_output_path = f"/workspace/work/A2TG_Baselines/FixedMemory_{mb}MB/MipNerf360/bbsplat"
tnt_output_path = f"/workspace/work/A2TG_Baselines/FixedMemory_{mb}MB/tandt/bbsplat"
db_output_path = f"/workspace/work/A2TG_Baselines/FixedMemory_{mb}MB/db/bbsplat"

mipnerf360_outdoor_scenes = ["bicycle", "garden", "stump"]
mipnerf360_outdoor_points = [points for _ in mipnerf360_outdoor_scenes]
mipnerf360_indoor_scenes = ["room", "counter", "kitchen", "bonsai"]
mipnerf360_indoor_points = [points for _ in mipnerf360_indoor_scenes]
tanks_and_temples_scenes = ["truck", "train"]
tanks_and_temples_points = [points for _ in tanks_and_temples_scenes]
deep_blending_scenes = ["drjohnson", "playroom"] 
deep_blending_points = [points for _ in deep_blending_scenes] 

all_scenes = []
all_scenes.extend(mipnerf360_outdoor_scenes)
all_scenes.extend(mipnerf360_indoor_scenes)
all_scenes.extend(tanks_and_temples_scenes)
all_scenes.extend(deep_blending_scenes)

if not args.skip_training or not args.skip_rendering:
    args = parser.parse_args()

if not args.skip_training:
    common_args = " --quiet --eval --test_iterations -1"
    for scene, max_points in zip(mipnerf360_outdoor_scenes, mipnerf360_outdoor_points):
        source = mipnerf360_path + "/" + scene
        model = mipnerf360_output_path
        os.system(f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE_ID} python train.py -s " + source + " -i images_4 -m " + model + "/" + scene + f" --cap_max={max_points}" + f" --max_read_points={max_points-10_000}" + " --add_sky_box" + common_args)
    for scene, max_points in zip(mipnerf360_indoor_scenes, mipnerf360_indoor_points):
        source = mipnerf360_path + "/" + scene
        model = mipnerf360_output_path
        os.system(f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE_ID} python train.py -s " + source + " -i images_2 -m " + model + "/" + scene + f" --cap_max={max_points}" + f" --max_read_points={max_points-10_000}" + " --add_sky_box" + common_args)
    for scene, max_points in zip(tanks_and_temples_scenes, tanks_and_temples_points):
        source = tnt_path + "/" + scene
        model = tnt_output_path
        os.system(f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE_ID} python train.py -s " + source + " -m " + model + "/" + scene + f" --cap_max={max_points}" + f" --max_read_points={max_points-10_000}" + " --add_sky_box" + common_args)
    for scene, max_points in zip(deep_blending_scenes, deep_blending_points):
        source = db_path + "/" + scene
        model = db_output_path
        os.system(f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE_ID} python train.py -s " + source + " -m " + model + "/" + scene + f" --cap_max={max_points}" + f" --max_read_points={max_points-10_000}" + common_args)

all_output_path = []
for scene in mipnerf360_outdoor_scenes:
    all_output_path.append(mipnerf360_output_path)
for scene in mipnerf360_indoor_scenes:
    all_output_path.append(mipnerf360_output_path)
for scene in tanks_and_temples_scenes:
    all_output_path.append(tnt_output_path)
for scene in deep_blending_scenes:
    all_output_path.append(db_output_path)
    
if not args.skip_rendering:

    all_sources = []
    for scene in mipnerf360_outdoor_scenes:
        all_sources.append(mipnerf360_path + "/" + scene)
    for scene in mipnerf360_indoor_scenes:
        all_sources.append(mipnerf360_path + "/" + scene)
    for scene in tanks_and_temples_scenes:
        all_sources.append(tnt_path + "/" + scene)
    for scene in deep_blending_scenes:
        all_sources.append(db_path + "/" + scene)

    common_args = " --quiet --eval --skip_train --skip_mesh "
    for scene, source, output_path in zip(all_scenes, all_sources, all_output_path):
        os.system(f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE_ID} python render.py --iteration 30000 -s " + source + " -m " + output_path + "/" + scene + common_args)

if not args.skip_metrics:
    scenes_string = ""
    for scene, output_path in zip(all_scenes, all_output_path):
        scenes_string += "\"" + output_path + "/" + scene + "\" "
    
    os.system(f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE_ID} python metrics.py -m " + scenes_string)

if not args.skip_memory:
    for scene, output_path in zip(all_scenes, all_output_path):
        os.system(f"CUDA_VISIBLE_DEVICES={CUDA_DEVICE_ID} python mem.py -m " + output_path + "/" + scene)