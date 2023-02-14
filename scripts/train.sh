CODES=$(pwd)
export PYTHONPATH=/mnt/petrelfs/wubohong/.conda/envs/py37:$PYTHONPATH # /path/to/python

export USE_TF=0
export USE_TORCH=1

save_path="/mnt/petrelfs/xujingjing/zhaochenyang/ckpt"  # "/path/to/save"
CKPT="/mnt/petrelfs/xujingjing/zhaochenyang/model/xlm-roberta-base" # "/path/to/pretrained/xlmr"
mkdir -p ${save_path}
data_path="/mnt/petrelfs/xujingjing/xujingjing/ted/de_en/train.merged" # /path/to/data

python $CODES/thumt/bin/trainer.py \
    --output ${save_path}/test_output \
    --input ${data_path} \
    --model xlmr_sga \
    --ptm $CKPT \
    --parameters=device_list=[0,1,2,3],train_steps=40000,update_cycle=4,batch_size=2048,save_checkpoint_steps=2000,max_length=256 \
    --hparam_set base