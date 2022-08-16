## Undefended pytorch model

To train train the network
```
cd examples/undefended_pytorch_resnet
python main.py
```

To run prediction and make coverage-error plot:
```
CUDA_VISIBLE_DEVICES=0 python main.py --resume=../model_zoo/undefended_pytorch_resnet.pth.tar --evaluate
```
