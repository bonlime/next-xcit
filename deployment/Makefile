.PHONY: install
install: 
	python3 -m pip install -U pip && \
	pip3 install -U onnx onnx-simplifier onnxruntime einops timm opencv-python opencv-python-headless matplotlib fvcore && \
	pip3 install git+https://github.com/bonlime/pytorch-tools.git@master && \
	pip3 install torch --extra-index-url https://download.pytorch.org/whl/cu116 && \

.PHONY: measure
measure:
	python3 export_tensorrt_engine.py --model nextvit_small_xca --trtexec-path `which trtexec` --profile --image-size 224 > xca_224.log && \
	python3 export_tensorrt_engine.py --model nextvit_small_vit --trtexec-path `which trtexec` --profile --image-size 224 > vit_224.log && \
	python3 export_tensorrt_engine.py --model nextvit_small_vit-orig --trtexec-path `which trtexec` --profile --image-size 224 > vit-orig_224.log && \
	python3 export_tensorrt_engine.py --model nextvit_small_xca --trtexec-path `which trtexec` --profile --image-size 384 > xca_384.log && \
	python3 export_tensorrt_engine.py --model nextvit_small_vit --trtexec-path `which trtexec` --profile --image-size 384 > vit_384.log && \
	python3 export_tensorrt_engine.py --model nextvit_small_vit-orig --trtexec-path `which trtexec` --profile --image-size 384 > vit-orig_384.log &&\
	python3 export_tensorrt_engine.py --model nextvit_small_xca --trtexec-path `which trtexec` --profile --image-size 512 > xca_512.log && \
	python3 export_tensorrt_engine.py --model nextvit_small_vit --trtexec-path `which trtexec` --profile --image-size 512 > vit_512.log && \
	python3 export_tensorrt_engine.py --model nextvit_small_vit-orig --trtexec-path `which trtexec` --profile --image-size 512 > vit-orig_512.log

.PHONY: print
print: 
	echo "XCA @ 224" && tail -n 8 xca_224.log | head -n 1 && \
	echo "XCA @ 384" && tail -n 8 xca_384.log | head -n 1 && \
	echo "XCA @ 512" && tail -n 8 xca_512.log | head -n 1 && \
	echo "VIT @ 224" && tail -n 8 vit_224.log | head -n 1 && \
	echo "VIT @ 384" && tail -n 8 vit_384.log | head -n 1 && \
	echo "VIT @ 512" && tail -n 8 vit_512.log | head -n 1 && \
	echo "VIT_orig @ 224" && tail -n 8 vit-orig_224.log | head -n 1 && \
	echo "VIT_orig @ 384" && tail -n 8 vit-orig_384.log | head -n 1 && \
	echo "VIT_orig @ 512" && tail -n 8 vit-orig_512.log | head -n 1
