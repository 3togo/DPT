"""Compute depth maps for images in the input folder.
"""
import os
import glob
import torch
import cv2
import argparse

import util.io

from torchvision.transforms import Compose

from dpt.models import DPTDepthModel
from dpt.midas_net import MidasNet_large
from dpt.transforms import Resize, NormalizeImage, PrepareForNet

# from util.misc import visualize_attention


# def run(input_path, output_path, model_path, model_type="dpt_hybrid", optimize=True):
    # """Run MonoDepthNN to compute depth maps.

    # Args:
        # input_path (str): path to input folder
        # output_path (str): path to output folder
        # model_path (str): path to saved model
    # """
    # print("initialize")

    # # select device
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # print("device: %s" % device)
      # net_w = net_h = 384
        # model = DPTDepthModel(
            # path=model_path,
            # backbone="vitb_rn50_384",
            # non_negative=True,
            # enable_attention_hooks=False,
        # )
        # normalization = NormalizeImage(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # transform = Compose(
        # [
            # Resize(
                # net_w,
                # net_h,
                # resize_target=None,
                # keep_aspect_ratio=True,
                # ensure_multiple_of=32,
                # resize_method="minimal",
                # image_interpolation_method=cv2.INTER_CUBIC,
            # ),
            # normalization,
            # PrepareForNet(),
        # ]
    # )


if __name__ == "__main__":
    # select device
    device = torch.device("cuda")
    print("device: %s" % device)
    model_path = "/mnt/data/git/MiDaS-cpp/models/dpt_hybrid-midas-d889a10e.pt"
    # load network
    example_input = torch.rand(1, 3, 288, 384, dtype=torch.float32)
    #model = MidasNet(in_model_path, non_negative=True)
    model = DPTDepthModel(
            path=model_path,
            # backbone="vitb_rn50_384",
            non_negative=True,
            # enable_attention_hooks=False,
        )
    sm = torch.jit.trace(model, example_input)

    # Save the torchscript model
    # sm.save(out_model_path)
