cd ..
gpus='0'
export CUDA_VISIBLE_DEVICES=${gpus}
echo "Using gpus ${gpus}"
now=$(date +"%T")
echo "Current time : $now"


resume='mamp.pt'
proc_name='Test_MAMP'
arch='MAMP'
datapath='dataset/DEMO'
memory_length=4
pad_divisible=16
test_corr_radius=12
echo "Padding divisible by "${pad_divisible}

python -u evaluate_custom.py \
  --arch ${arch} \
  --test_corr_radius ${test_corr_radius} \
  --proc_name ${proc_name} \
  --resume "ckpt/"${resume} \
  --datapath ${datapath} \
  --savepath "ckpt" \
  --pad_divisible ${pad_divisible} \
  --memory_length ${memory_length} \
  --optical_flow_warp 1 \
#  --is_amp \
