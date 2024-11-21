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

import torch
from scene import Scene
import os
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel


def final_prune(self, compress=False):
    prune_mask = (torch.sigmoid(self._mask) <= 0.01).squeeze()
    self.prune_points(prune_mask)
    if compress:
        self.sort_morton()

    for m in self.vq_scale.layers:
        m.training = False
    for m in self.vq_rot.layers:
        m.training = False

    self._xyz = self._xyz.clone().half().float()
    self._scaling, self.sca_idx, _ = self.vq_scale(self.get_scaling.unsqueeze(1))
    self._rotation, self.rot_idx, _ = self.vq_rot(self.get_rotation.unsqueeze(1))
    self._scaling = self._scaling.squeeze()
    self._rotation = self._rotation.squeeze()

    position_mb = self._xyz.shape[0] * 3 * 16 / 8 / 10 ** 6
    scale_mb = self._xyz.shape[
                   0] * self.rvq_bit * self.rvq_num / 8 / 10 ** 6 + 2 ** self.rvq_bit * self.rvq_num * 3 * 32 / 8 / 10 ** 6
    rotation_mb = self._xyz.shape[
                      0] * self.rvq_bit * self.rvq_num / 8 / 10 ** 6 + 2 ** self.rvq_bit * self.rvq_num * 4 * 32 / 8 / 10 ** 6
    opacity_mb = self._xyz.shape[0] * 16 / 8 / 10 ** 6
    hash_mb = self.recolor.params.shape[0] * 16 / 8 / 10 ** 6
    mlp_mb = self.mlp_head.params.shape[0] * 16 / 8 / 10 ** 6
    sum_mb = position_mb + scale_mb + rotation_mb + opacity_mb + hash_mb + mlp_mb

    mb_str = "Storage\nposition: " + str(position_mb) + "\nscale: " + str(scale_mb) + "\nrotation: " + str(
        rotation_mb) + "\nopacity: " + str(opacity_mb) + "\nhash: " + str(hash_mb) + "\nmlp: " + str(
        mlp_mb) + "\ntotal: " + str(sum_mb) + " MB"

    if compress:
        self._opacity, self.quant_opa, self.minmax_opa = self.post_quant(self.get_opacity)
        self.recolor.params, self.quant_hash, self.minmax_hash = self.post_quant(self.recolor.params, True)

        scale_mb, self.huf_sca, self.tab_sca = self.huffman_encode(self.sca_idx)
        scale_mb += 2 ** self.rvq_bit * self.rvq_num * 3 * 32 / 8 / 10 ** 6
        rotation_mb, self.huf_rot, self.tab_rot = self.huffman_encode(self.rot_idx)
        rotation_mb += 2 ** self.rvq_bit * self.rvq_num * 4 * 32 / 8 / 10 ** 6
        opacity_mb, self.huf_opa, self.tab_opa = self.huffman_encode(self.quant_opa)
        hash_mb, self.huf_hash, self.tab_hash = self.huffman_encode(self.quant_hash)
        mlp_mb = self.mlp_head.params.shape[0] * 16 / 8 / 10 ** 6
        sum_mb = position_mb + scale_mb + rotation_mb + opacity_mb + hash_mb + mlp_mb

        mb_str = mb_str + "\n\nAfter PP\nposition: " + str(position_mb) + "\nscale: " + str(
            scale_mb) + "\nrotation: " + str(rotation_mb) + "\nopacity: " + str(opacity_mb) + "\nhash: " + str(
            hash_mb) + "\nmlp: " + str(mlp_mb) + "\ntotal: " + str(sum_mb) + " MB"
    else:
        self._opacity = self.get_opacity.clone().half().float()
    torch.cuda.empty_cache()
    return mb_str

def get_size(dataset: ModelParams, iteration: int, pipeline: PipelineParams, skip_train: bool, skip_test: bool):
    with torch.no_grad():
        gaussians = GaussianModel(dataset)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        gaussians.precompute()

        storage = gaussians.final_prune_for_test(compress=True) # 默认采用后处理
        with open(os.path.join(args.model_path, "storage.txt"), 'w') as c:
            c.write(storage)




if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")

    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    get_size(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)