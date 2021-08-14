cd ..
gpus='0'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "using gpus ${gpus}"
now=$(date +"%T")
echo "Current time : $now"

datapath='dataset/YOUTUBE/all_256'
arch='MAMP'
ref_num=1
img_size=256
train_corr_radius=6
bsize=24
worker=${bsize}
lr=1e-3
epochs=33
proc_name='Train_MAMP'

python -u train.py \
    --datapath ${datapath} \
    --arch ${arch} \
    --proc_name ${proc_name} \
    --ref_num ${ref_num} \
    --train_corr_radius ${train_corr_radius} \
    --img_size ${img_size} \
    --lr ${lr} \
    --bsize ${bsize} \
    --worker ${worker} \
    --epochs ${epochs} \
    --resume '' \
#    --optical_flow_warp 0 \
#    --is_amp
