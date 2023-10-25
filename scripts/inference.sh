CODES=$(pwd)
export PYTHONPATH=$CODES:$PYTHONPATH

CKPT="/path/to/pretrain_ckpt"
model_path="/path/to/prefix_model.pt"
output="/path/to/output_data"
input_data="/path/to/inference_data"

python $CODES/bin/translator_for_XLMR.py \
    --input ${input_data} \
    --ptm $CKPT \
    --output ${output} \
    --model xlmr_sga \
    --half --prefix ${model_path} \
    --parameters=decode_alpha=0.0,decode_batch_size=16,prompt_length=128

python post_process_hypo.py ${output}/hypo.txt
# perl /path/to/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en < ${output}/hypo.txt > ${output}/hypo_detok.txt
# perl /path/to/mosesdecoder/scripts/tokenizer/detokenizer.perl -l en < ${output}/references.txt > ${output}/references_detok.txt

sacrebleu ${output}/references.txt < ${output}/hypo.txt --tokenize 13a