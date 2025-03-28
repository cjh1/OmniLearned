# load libs
module load pytorch

# for DDP
export MASTER_ADDR=$(hostname)

# #Very Big PET  Total params: 654.88M
# cmd="omnilearned train  -o ./ --save-tag pretrain --dataset pretrain --use-pid --use-add --num-classes 201 --batch 64 --iterations 1000 --mode pretrain --use-clip --epoch 200 --wd 0.0 --num-transf 24 --base-dim 1152 --num-head 16 --feature-drop 0.1 --attn-drop 0.0 --mlp-drop 0.0"

# #Big PET Total params: 178.11M
# cmd="omnilearned train  -o ./ --save-tag pretrain --dataset pretrain --use-pid --use-add --num-classes 201 --batch 64 --iterations 1000 --mode pretrain --use-clip --epoch 200 --wd 0.0 --num-transf 12 --base-dim 768 --num-head 12 --feature-drop 0.1 --attn-drop 0.0 --mlp-drop 0.0"


#Medium PET Total params: 44.81M
cmd="omnilearned train  -o ./ --save-tag pretrain --dataset pretrain --use-pid --use-add --num-classes 201 --batch 64 --iterations 1000 --mode pretrain --use-clip --epoch 200 --wd 0.0 --num-transf 12 --base-dim 384 --num-head 6 --feature-drop 0.1 --attn-drop 0.0 --mlp-drop 0.0"


# #Small PET Total params: 0.9M
# cmd="omnilearned train  -o ./ --save-tag pretrain --dataset pretrain --use-pid --use-add --num-classes 201 --batch 64 --iterations 1000 --mode pretrain --use-clip --epoch 200 --wd 0.0 --num-transf 6 --base-dim 64 --num-head 8 --feature-drop 0.1 --attn-drop 0.0 --mlp-drop 0.0"


#TOP
#cmd="omnilearned train  -o ./ --save-tag top --dataset top  --num-classes 2 --epoch 15 --wd 0.3 --num-transf 6 --base-dim 64 --num-head 8 --feature-drop 0.0 --attn-drop 0.1 --mlp-drop = 0.1"

set -x
srun -l -u \
    bash -c "
    source export_ddp.sh
    $cmd
    "
