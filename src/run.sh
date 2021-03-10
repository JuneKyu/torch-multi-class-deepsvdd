# mnist
# python main.py mnist mnist_LeNet ../log/mnist_test_0 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 0;
#
# python main.py mnist mnist_LeNet ../log/mnist_test_01 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 01;
#
# python main.py mnist mnist_LeNet ../log/mnist_test_012 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 012;
#
# python main.py mnist mnist_LeNet ../log/mnist_test_0123 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 0123;
#
# python main.py mnist mnist_LeNet ../log/mnist_test_01234 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 01234;
#
# python main.py mnist mnist_LeNet ../log/mnist_test_012345 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 012345;
#
# python main.py mnist mnist_LeNet ../log/mnist_test_0123456 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 0123456;
#
# python main.py mnist mnist_LeNet ../log/mnist_test_01234567 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 01234567;
#
# python main.py mnist mnist_LeNet ../log/mnist_test_012345678 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 012345678;
#
# python main.py mnist mnist_LeNet ../log/mnist_test_0123456789 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 0123456789;
#

# cifar10
# python main.py cifar10 cifar10_LeNet ../log/cifar10_test_0 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 0;
# python main.py cifar10 cifar10_LeNet ../log/cifar10_test_01 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 01;
# python main.py cifar10 cifar10_LeNet ../log/cifar10_test_012 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 012;
# python main.py cifar10 cifar10_LeNet ../log/cifar10_test_0123 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 0123;
# python main.py cifar10 cifar10_LeNet ../log/cifar10_test_01234 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 01234;
# python main.py cifar10 cifar10_LeNet ../log/cifar10_test_012345 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 012345;
# python main.py cifar10 cifar10_LeNet ../log/cifar10_test_0123456 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 0123456;
# python main.py cifar10 cifar10_LeNet ../log/cifar10_test_01234567 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 01234567;
# python main.py cifar10 cifar10_LeNet ../log/cifar10_test_012345678 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 012345678;


# python main.py cifar10 cifar10_LeNet ../log/cifar10_test_0189 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 0189;

# python main.py cifar10 cifar10_LeNet ../log/cifar10_test_234567 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 --normal_class 234567;

# python main.py mnist mnist_LeNet ../log/mnist_test_038 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 038;
# 86.90, 8170

# python main.py mnist mnist_LeNet ../log/mnist_test_2569 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 2569;
#

# python main.py mnist mnist_LeNet ../log/mnist_test_147 ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 150 --ae_lr_milestone 50 --ae_batch_size 200 --ae_weight_decay 0.5e-3 --normal_class 147;
# 89.57, 90.29, 89.96, 86.19 



# TODO: tiny_imagenet need to be configured correctly.
# python main.py tiny_imagenet tiny_imagenet_LeNet ../log/tiny_imagenet_test_animal ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 \
#   --normal_class 0;
#
# python main.py tiny_imagenet tiny_imagenet_LeNet ../log/tiny_imagenet_test_insect ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 \
#   --normal_class 1;
#
# python main.py tiny_imagenet tiny_imagenet_LeNet ../log/tiny_imagenet_test_instruments ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 \
#   --normal_class 2;
#
# python main.py tiny_imagenet tiny_imagenet_LeNet ../log/tiny_imagenet_test_structure ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 \
#   --normal_class 3;
#
# python main.py tiny_imagenet tiny_imagenet_LeNet ../log/tiny_imagenet_test_transportation ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 \
#   --normal_class 4;

# GTSRB
# python main.py gtsrb gtsrb_LeNet ../log/gtsrb_test_speed_limits ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 \
#   --normal_class 0;
#
# python main.py gtsrb gtsrb_LeNet ../log/gtsrb_test_driving_instructions ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 \
#   --normal_class 1;
#
# python main.py gtsrb gtsrb_LeNet ../log/gtsrb_test_warnings ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 \
#   --normal_class 2;
#
# python main.py gtsrb gtsrb_LeNet ../log/gtsrb_test_directions ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 \
#   --normal_class 3;
#
# python main.py gtsrb gtsrb_LeNet ../log/gtsrb_test_special_signs ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 \
#   --normal_class 4;
#
# python main.py gtsrb gtsrb_LeNet ../log/gtsrb_test_regulations ../data --objective one-class --lr 0.0001\
#   --n_epochs 150 --lr_milestone 50 --batch_size 200 --weight_decay 0.5e-6 --pretrain True --ae_lr 0.0001\
#   --ae_n_epochs 350 --ae_lr_milestone 250 --ae_batch_size 200 --ae_weight_decay 0.5e-6 \
#   --normal_class 5;
