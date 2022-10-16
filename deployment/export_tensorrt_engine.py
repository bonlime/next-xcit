# -*- coding: utf-8 -*-
import os
import torch
import argparse

# from timm.models import create_model
import sys
import onnx
from onnxsim import onnx_simplifier

sys.path.append("../classification")
from nextvit import nextvit_small
import utils

parser = argparse.ArgumentParser("Next-ViT export TensorRT engine script", add_help=False)
parser.add_argument("--batch-size", type=int, default=8, help="batch size used to export TensorRT engine.")
parser.add_argument("--image-size", type=int, default=224, help="image size used to TensorRT engine.")
parser.add_argument(
    "--model",
    type=str,
    default="nextvit_small_xca",
    choices=["nextvit_small_xca", "nextvit_small_vit", "nextvit_small_vit-orig"],
    help="model type.",
)
parser.add_argument("--datatype", type=str, default="fp16", choices=["fp16", "int8"], help="datatype of trt engine.")
parser.add_argument("--opset-version", type=str, default=13, help="the onnx opset version.")
parser.add_argument("--trtexec-path", type=str, help="path to your trtexec tool.")
parser.add_argument("--profile", type=bool, default=False, help="profile the performance of the trt engine.")
parser.add_argument(
    "--threads",
    type=int,
    default=1,
    help="number of threads for profiling. \
        (It is used when `profile` == True.)",
)
parser.add_argument(
    "--warmUp",
    type=int,
    default=10,
    help="number of warmUp for profiling. \
        (It is used when `profile` == True.)",
)
parser.add_argument(
    "--iterations",
    type=int,
    default=100,
    help="number of iterations for profiling. \
        (It is used when `profile` == True.)",
)
parser.add_argument(
    "--dumpProfile",
    type=bool,
    default=False,
    help="profile the performance of the trt engine.  \
        (It is used when `profile` == True.)",
)

parser.add_argument("--skip-onnx-export", action="store_true")

args = parser.parse_args()


def remove_bn(module: torch.nn.Module) -> None:
    for mod_name, mod in module.named_children():
        if isinstance(mod, torch.nn.BatchNorm2d):
            setattr(module, mod_name, torch.nn.Identity())
        remove_bn(mod)


def main():
    if args.model == "nextvit_small_xca":
        model = nextvit_small(attn_type="xca")
    elif args.model == "nextvit_small_vit":
        model = nextvit_small(attn_type="vit")
    elif args.model == "nextvit_small_vit-orig":
        model = nextvit_small(attn_type="vit_BNC")

    model = model.eval().requires_grad_(False)

    input_tensor = torch.zeros((args.batch_size, 3, args.image_size, args.image_size), dtype=torch.float32)
    utils.cal_flops_params_with_fvcore(model, input_tensor)

    # Merge pre bn before exporting onnx/coreml model to speedup inference.
    # all this BN could be merged into convs, so remove them for measuring speed
    remove_bn(model)

    engine_file = f"{args.model}_{args.image_size}x{args.image_size}"

    ##export and simplify onnx model
    if not args.skip_onnx_export:
        # torch.set_default_tensor_type('torch.FloatTensor')
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.onnx.export(model, input_tensor, f"{engine_file}.onnx", verbose=True)
        onnx_model = onnx.load(f"{engine_file}.onnx")
        model_simp, check = onnx_simplifier.simplify(onnx_model, check_n=0)
        onnx.save(model_simp, f"{engine_file}.onnx")
        print("Finished converting to onnx")

    if args.trtexec_path is None:
        return

    import subprocess

    ##dump trt engine
    convert_state = subprocess.call(
        f"{args.trtexec_path} --onnx={engine_file}.onnx --saveEngine={engine_file}_{args.datatype}.trt --explicitBatch --{args.datatype} --verbose",
        shell=True,
    )

    if convert_state:
        print("Convert Engine Failed. Please Check.")
    else:
        print(f"TRT Engine saved to: {engine_file}.trt .")
        if args.profile:
            subprocess.call(
                f"{args.trtexec_path} --loadEngine={engine_file}_{args.datatype}.trt --threads={args.threads} --warmUp={args.warmUp} --iterations={args.iterations} --dumpProfile={args.dumpProfile}",
                shell=True,
            )


if __name__ == "__main__":
    main()
