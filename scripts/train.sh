CODES=$(pwd)
export PYTHONPATH=/mnt/petrelfs/wubohong/.conda/envs/py37:$PYTHONPATH # /path/to/python

export USE_TF=0
export USE_TORCH=1

save_path="/path/to/save"
CKPT="/path/to/prefix_model.pt" # "/path/to/pretrained/xlmr"
data_path="/path/to/data" # /path/to/data

python $CODES/thumt/bin/trainer.py \
    --output ${save_path} \
    --input ${data_path} \
    --model xlmr_sga \
    --ptm $CKPT \
    --parameters=device_list=[0,1,2,3],train_steps=40000,update_cycle=4,batch_size=2048,save_checkpoint_steps=2000,max_length=256 \
    --hparam_set base
