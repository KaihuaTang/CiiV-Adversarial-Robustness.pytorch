# Deprecated Codes

### CiiV Training
Using the yaml files in config folder to train the model, for example:
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10_ciiv.yaml --output_dir checkpoints/cifar10_ciiv  --require_eval --phase train
```
All the hyper-parameters are already in the yaml config file.

### Baseline Training
Baseline models and other AT or non-AT models all need to use another config file. Take cifar10 as an example:
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10.yaml --output_dir checkpoints/cifar10_baseline  --require_eval --phase train
```
AT settings and non-AT settings can be switched on in the corresponding config file.

### Evaluation
To evaluate the model, we need to set the phase to val or test, and indicate the loading checkpoints, e.g.,:
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/mnist.yaml --output_dir checkpoints/mnist_baseline  --require_eval --phase test   --load_dir checkpoints/mnist_baseline/epoch_99_test_model.pth
```
Note that you need to turn on the adv_train in the config file, and select the corresponding attacking methods.

### Simple Evaluation
If you want to test multiple attackers, you can change --phase to val_all, all the default attackers will be evaluated. Details of val_all is implemented in main.py, for example:
```
CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/mnist.yaml --output_dir checkpoints/mnist_baseline  --require_eval --phase val_all  --load_dir checkpoints/mnist_baseline/epoch_99_test_model.pth
```