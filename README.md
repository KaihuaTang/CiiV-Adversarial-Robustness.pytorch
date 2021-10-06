# Codes for "Adversarial Visual Robustness by Causal Intervention"

### Baseline for CIFAR-10
Baseline models
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/_cifar10.yaml --output_dir checkpoints/cifar10_resnet_baseline  --require_eval --phase train
```

### CiiV for CIFAR-10
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/_cifar10_ciiv.yaml --output_dir checkpoints/cifar10_resnet_ciiv  --require_eval --phase train
```

### Evaluation
Examples for evaluation
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/_cifar10_ciiv.yaml --output_dir checkpoints/cifar10_resnet_ciiv  --require_eval --phase test --load_dir checkpoints/cifar10_resnet_ciiv/epoch_XXX.pth --adv_test --adv_type TYPE_ADV_TYPE_HERE --adv_setting TYPE_DIFFERENT_SETTING_HERE
``