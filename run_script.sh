CUDA_VISIBLE_DEVICES=1 python main.py --cfg config/cifar10.yaml --output_dir checkpoints/cifar10_org  --require_eval --phase test   --load_dir checkpoints/cifar10_org/epoch_99_test_model.pth   attacker_opt.adv_val True  attacker_opt.eot_iter 10  attacker_opt.attack_type EOT

CUDA_VISIBLE_DEVICES=2 python main.py --cfg config/cifar10.yaml --output_dir checkpoints/cifar10_adv_fgsm8  --require_eval --phase test   --load_dir checkpoints/cifar10_adv_fgsm8/epoch_99_test_model.pth   attacker_opt.adv_val True   attacker_opt.eot_iter 10  attacker_opt.attack_type EOT

CUDA_VISIBLE_DEVICES=3 python main.py --cfg config/cifar10.yaml --output_dir checkpoints/cifar10_adv_pgd8  --require_eval --phase test   --load_dir checkpoints/cifar10_adv_pgd8/epoch_99_test_model.pth   attacker_opt.adv_val True  attacker_opt.eot_iter 10  attacker_opt.attack_type EOT

CUDA_VISIBLE_DEVICES=1 python main.py --cfg config/cifar10.yaml --output_dir checkpoints/cifar10_adv_pgdl2_128  --require_eval --phase test   --load_dir checkpoints/cifar10_adv_pgdl2_128/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.eot_iter 10  attacker_opt.attack_type EOT

CUDA_VISIBLE_DEVICES=2 python main.py --cfg config/cifar10.yaml --output_dir checkpoints/cifar10_mixup  --require_eval --phase test   --load_dir checkpoints/cifar10_mixup/epoch_99_test_model.pth   attacker_opt.adv_val True   attacker_opt.eot_iter 10  attacker_opt.attack_type EOT

CUDA_VISIBLE_DEVICES=3 python main.py --cfg config/cifar10.yaml --output_dir checkpoints/cifar10_bpfc  --require_eval --phase test   --load_dir checkpoints/cifar10_bpfc/epoch_99_test_model.pth   attacker_opt.adv_val True   attacker_opt.eot_iter 10  attacker_opt.attack_type EOT

CUDA_VISIBLE_DEVICES=1 python main.py --cfg config/cifar10_rs.yaml --output_dir checkpoints/cifar10_rs  --require_eval --phase test   --load_dir checkpoints/cifar10_rs/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.eot_iter 10  attacker_opt.attack_type EOT

CUDA_VISIBLE_DEVICES=2 python main.py --cfg config/cifar10_final_time.yaml --output_dir checkpoints/cifar10_finalv5_loop15  --require_eval --phase test   --load_dir checkpoints/cifar10_finalv5_loop15/epoch_99_test_model.pth   attacker_opt.adv_val True  attacker_opt.eot_iter 10  attacker_opt.attack_type EOT

################### Train CIFAR-10 ####################
#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10.yaml --output_dir checkpoints/cifar10_adv_fgsm8_resnet18 --require_eval --phase train   attacker_opt.adv_train True    attacker_opt.attack_type FGSM   networks.params.m_type resnet18

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10.yaml --output_dir checkpoints/cifar10_adv_pgd8_resnet18 --require_eval --phase train    attacker_opt.adv_train True    attacker_opt.attack_type PGD    networks.params.m_type resnet18

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10.yaml --output_dir checkpoints/cifar10_adv_pgdl2_128_resnet18 --require_eval --phase train    attacker_opt.adv_train True     attacker_opt.attack_type PGDL2    networks.params.m_type resnet18

##################### Train MNIST ########################
#CUDA_VISIBLE_DEVICES=2 python main.py --cfg config/mnist.yaml --output_dir checkpoints/mnist_adv_fgsm8_resnet18 --require_eval --phase train   attacker_opt.adv_train True    attacker_opt.attack_type FGSM   networks.params.m_type resnet18

#CUDA_VISIBLE_DEVICES=2 python main.py --cfg config/mnist.yaml --output_dir checkpoints/mnist_adv_pgd8_resnet18 --require_eval --phase train    attacker_opt.adv_train True    attacker_opt.attack_type PGD    networks.params.m_type resnet18

#CUDA_VISIBLE_DEVICES=2 python main.py --cfg config/mnist.yaml --output_dir checkpoints/mnist_adv_pgdl2_128_resnet18 --require_eval --phase train    attacker_opt.adv_train True     attacker_opt.attack_type PGDL2    networks.params.m_type resnet18

########################### Save CIFAR-10 Blackbox Example ###################
#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10_final_time.yaml --output_dir checkpoints/cifar10_finalv5_loop15_resnet18  --require_eval --phase val   --load_dir checkpoints/cifar10_finalv5_loop15_resnet18/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type FGSM  networks.params.m_type resnet18  blackbox_save True

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10_final_time.yaml --output_dir checkpoints/cifar10_finalv5_loop15_resnet18  --require_eval --phase val   --load_dir checkpoints/cifar10_finalv5_loop15_resnet18/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type PGD  networks.params.m_type resnet18  blackbox_save True

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10_final_time.yaml --output_dir checkpoints/cifar10_finalv5_loop15_resnet50  --require_eval --phase val   --load_dir checkpoints/cifar10_finalv5_loop15_resnet50/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type FGSM  networks.params.m_type resnet50  blackbox_save True

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10_final_time.yaml --output_dir checkpoints/cifar10_finalv5_loop15_resnet50  --require_eval --phase val   --load_dir checkpoints/cifar10_finalv5_loop15_resnet50/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type PGD  networks.params.m_type resnet50  blackbox_save True


########################### Save MNIST Blackbox Example ###################
#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/mnist_final_time.yaml --output_dir checkpoints/mnist_finalv5_loop20_resnet18  --require_eval --phase val   --load_dir checkpoints/mnist_finalv5_loop20_resnet18/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type FGSM  networks.params.m_type resnet18  blackbox_save True

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/mnist_final_time.yaml --output_dir checkpoints/mnist_finalv5_loop20_resnet18  --require_eval --phase val   --load_dir checkpoints/mnist_finalv5_loop20_resnet18/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type PGD  networks.params.m_type resnet18  blackbox_save True

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/mnist_final_time.yaml --output_dir checkpoints/mnist_finalv5_loop20_resnet50  --require_eval --phase val   --load_dir checkpoints/mnist_finalv5_loop20_resnet50/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type FGSM  networks.params.m_type resnet50  blackbox_save True

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/mnist_final_time.yaml --output_dir checkpoints/mnist_finalv5_loop20_resnet50  --require_eval --phase val   --load_dir checkpoints/mnist_finalv5_loop20_resnet50/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type PGD  networks.params.m_type resnet50  blackbox_save True


########################### Test CIFAR-10 Blackbox Example ###################
#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10_final_time.yaml --output_dir checkpoints/cifar10_finalv5_loop15_resnet18  --require_eval --phase val   --load_dir checkpoints/cifar10_finalv5_loop15/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type FGSM  blackbox_name blackbox_resnet18_FGSM.pth  blackbox_test True

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10_final_time.yaml --output_dir checkpoints/cifar10_finalv5_loop15_resnet18  --require_eval --phase val   --load_dir checkpoints/cifar10_finalv5_loop15/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type PGD  blackbox_name blackbox_resnet18_PGD.pth  blackbox_test True

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10_final_time.yaml --output_dir checkpoints/cifar10_finalv5_loop15_resnet50  --require_eval --phase val   --load_dir checkpoints/cifar10_finalv5_loop15/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type FGSM  blackbox_name blackbox_resnet50_FGSM.pth  blackbox_test True

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/cifar10_final_time.yaml --output_dir checkpoints/cifar10_finalv5_loop15_resnet50  --require_eval --phase val   --load_dir checkpoints/cifar10_finalv5_loop15/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type PGD  blackbox_name blackbox_resnet50_PGD.pth  blackbox_test True


########################### Test MNIST Blackbox Example ###################
#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/mnist_final_time.yaml --output_dir checkpoints/mnist_finalv5_loop20_resnet18  --require_eval --phase val   --load_dir checkpoints/mnist_finalv2_augweight0.8_loop20/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type FGSM  blackbox_name blackbox_resnet18_FGSM.pth  blackbox_test True

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/mnist_final_time.yaml --output_dir checkpoints/mnist_finalv5_loop20_resnet18  --require_eval --phase val   --load_dir checkpoints/mnist_finalv2_augweight0.8_loop20/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type PGD  blackbox_name blackbox_resnet18_PGD.pth  blackbox_test True

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/mnist_final_time.yaml --output_dir checkpoints/mnist_finalv5_loop20_resnet50  --require_eval --phase val   --load_dir checkpoints/mnist_finalv2_augweight0.8_loop20/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type FGSM  blackbox_name blackbox_resnet50_FGSM.pth  blackbox_test True

#CUDA_VISIBLE_DEVICES=0 python main.py --cfg config/mnist_final_time.yaml --output_dir checkpoints/mnist_finalv5_loop20_resnet50  --require_eval --phase val   --load_dir checkpoints/mnist_finalv2_augweight0.8_loop20/epoch_99_test_model.pth   attacker_opt.adv_val True    attacker_opt.attack_type PGD  blackbox_name blackbox_resnet50_PGD.pth   blackbox_test True

