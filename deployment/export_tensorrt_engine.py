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

parser = argparse.ArgumentParser('Next-ViT export TensorRT engine script', add_help=False)
parser.add_argument(
    '--batch-size',
    type=int,
    default=8,
    help='batch size used to export TensorRT engine.'
)
parser.add_argument(
    '--image-size',
    type=int,
    default=224,
    help='image size used to TensorRT engine.'
)
parser.add_argument(
    '--model',
    type=str,
    default='nextvit_small',
    choices=['nextvit_small', 'nextvit_base', 'nextvit_large'],
    help='model type.'
)
parser.add_argument(
    '--datatype',
    type=str,
    default='fp16',
    choices=['fp16', 'int8'],
    help='datatype of trt engine.'
)
parser.add_argument(
    '--opset-version',
    type=str,
    default=13,
    help='the onnx opset version.'
)
parser.add_argument(
    '--trtexec-path',
    type=str,
    help='path to your trtexec tool.'
)
parser.add_argument(
    '--profile',
    type=bool,
    default=False,
    help='profile the performance of the trt engine.'
)
parser.add_argument(
    '--threads',
    type=int,
    default=1,
    help='number of threads for profiling. \
        (It is used when `profile` == True.)'
)
parser.add_argument(
    '--warmUp',
    type=int,
    default=10,
    help='number of warmUp for profiling. \
        (It is used when `profile` == True.)'
)
parser.add_argument(
    '--iterations',
    type=int,
    default=100,
    help='number of iterations for profiling. \
        (It is used when `profile` == True.)'
)
parser.add_argument(
    '--dumpProfile',
    type=bool,
    default=False,
    help='profile the performance of the trt engine.  \
        (It is used when `profile` == True.)'
)

parser.add_argument("--skip-onnx-export", action="store_true")

args = parser.parse_args()

def main():
    model = nextvit_small()
    model = model.eval().requires_grad_(False)

    input_tensor = torch.zeros((args.batch_size, 3, args.image_size, args.image_size), dtype=torch.float32)
    utils.cal_flops_params_with_fvcore(model, input_tensor)

    # model = torch.jit.script(model)

    # Merge pre bn before exporting onnx/coreml model to speedup inference.
    if hasattr(model, "merge_bn"):
        model.merge_bn()

    engine_file = f"{args.model}_{args.image_size}x{args.image_size}"

    ##export and simplify onnx model
    if not args.skip_onnx_export:
        # torch.set_default_tensor_type('torch.FloatTensor')
        # torch.set_default_tensor_type('torch.cuda.FloatTensor')
        torch.onnx.export(model, input_tensor, \
                        f"{engine_file}.onnx", verbose=True)
        onnx_model = onnx.load(f"{engine_file}.onnx")
        model_simp, check = onnx_simplifier.simplify(onnx_model, check_n = 0)
        onnx.save(model_simp, f"{engine_file}.onnx")
        print("Finished converting to onnx")

    if args.trtexec_path is None:
        return

    import subprocess

    ##dump trt engine
    convert_state = subprocess.call(
        f"{args.trtexec_path} --onnx={engine_file}.onnx --saveEngine={engine_file}_{args.datatype}.trt --explicitBatch --{args.datatype} --verbose", shell=True
    )

    if not convert_state:
        print(f"TRT Engine saved to: {engine_file}.trt .")
        if args.profile:
            subprocess.call("%s --loadEngine=%s_%s.trt --threads=%d --warmUp=%d --iterations=%d --dumpProfile=%r" %
                            (args.trtexec_path, engine_file, args.datatype, args.threads, args.warmUp, args.iterations, args.dumpProfile), shell=True)
    else:
        print('Convert Engine Failed. Please Check.')

if __name__ == '__main__':
    main()
