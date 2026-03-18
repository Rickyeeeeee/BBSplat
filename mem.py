import torch
from scene import Scene
import os
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
import json

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Calculate memory usage")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Memory calculating " + args.model_path)

    dataset, iteration, pipe = model.extract(args), args.iteration, pipeline.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)
    
    point_count = gaussians.get_xyz.shape[0]
    mem_size_stats = {
        "point_count": point_count,
        "mem_bytes": point_count * 1081 * 4,
        "mem_mb": point_count * 1081 * 4 / (1000 * 1000),
        "mem_mib": point_count * 1081 * 4 / (1024 * 1024)
    }
    print(mem_size_stats)
    
    with open(args.model_path + "/mem.json", 'w') as fp:
        json.dump(mem_size_stats, fp, indent=True)