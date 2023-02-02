CODES=$(pwd)
export PYTHONPATH=/mnt/petrelfs/wubohong/.conda/envs/py37:$PYTHONPATH # /path/to/python

export USE_TF=0
export USE_TORCH=1

src=$1
suffix=$2

save_path="/mnt/petrelfs/wubohong/test_clean_code_xlmr_sga/ted_de_en"  # "/path/to/save"
CKPT="/mnt/petrelfs/wubohong/PRETRAIN/xlmr_roberta_base_hf/" # "/path/to/pretrained/xlmr"
mkdir -p ${save_path}
data_path="/mnt/petrelfs/wubohong/U4Gdata/tedtalks_v2/de_XX-en_XX/train.de_XX2en_XX" # /path/to/data

python $CODES/thumt/bin/trainer.py \
    --output ${save_path}/${suffix} \
    --input ${data_path} \
    --model xlmr_sga \
    --ptm $CKPT \
    --parameters=device_list=[0,1,2,3],train_steps=40000,update_cycle=4,batch_size=2048,save_checkpoint_steps=2000,max_length=256 \
    --hparam_set base
