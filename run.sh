
#! /bin/sh


declare -a seed=("0" "1" "2")

for se in ${seed[@]}; do
python main.py --seed=$se --dataset=super_imagenet --epochs=1 --method=fedprox --num_clients=300 --gpu=6
# python main.py --seed=$se --dataset=super_imagenet --epochs=1 --method=fed_gen --num_clients=300 --gpu=5 --pi=5000 --w_bn=75 --z_dim=200
done
