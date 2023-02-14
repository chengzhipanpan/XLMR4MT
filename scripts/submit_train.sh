srun -p NLP --gres=gpu:4 -N 1 --ntasks-per-node=1 --cpus-per-task=32 -J test_train \
    bash scripts/train.sh > logs/test.log 2>&1 &