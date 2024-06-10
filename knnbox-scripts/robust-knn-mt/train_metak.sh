export OMP_WAIT_POLICY=PASSIVE

# for it dataset, MAX_K=16
# for medical dataset, MAX_K=16
# for koran dataset, MAX_K=8, Might need to change line 186 at knnbox/combiner/robust_combiner.py to "tempe = torch.sigmoid(lambda_logit[:, :, 1:2]) * 0.2"
# for law dataset, MAX_K=8

PROJECT_PATH=$( cd -- "$( dirname -- "$ BASH_SOURCE[0]}" )" &> /dev/null && pwd )/../..
BASE_MODEL=$PROJECT_PATH/pretrain-models/wmt19.de-en/wmt19.de-en.ffn8192.pt
DATASET=it
MAX_K=16
DATA_PATH=$PROJECT_PATH/data-bin/$DATASET
SAVE_DIR=$PROJECT_PATH/save-models/combiner/robust/$DATASET
DATASTORE_LOAD_PATH=$PROJECT_PATH/datastore/vanilla/$DATASET


CUDA_VISIBLE_DEVICES=0 python $PROJECT_PATH/fairseq_cli/train.py $DATA_PATH \
--task translation \
--train-subset valid --valid-subset valid \
--best-checkpoint-metric "loss" \
--finetune-from-model $BASE_MODEL \
--optimizer adam --adam-betas '(0.9, 0.98)' --adam-eps 1e-8 --clip-norm 1.0 \
--lr 3e-4 --lr-scheduler reduce_lr_on_plateau \
--min-lr 3e-05 --label-smoothing 0.001 \
--lr-patience 5 --lr-shrink 0.5 --patience 30 --max-epoch 500 --max-update 5000 --validate-after-updates 1000 \
--criterion label_smoothed_cross_entropy_for_robust \
--no-epoch-checkpoints --no-last-checkpoints --no-save-optimizer-state \
--tensorboard-logdir $SAVE_DIR/log \
--save-dir $SAVE_DIR \
--batch-size 2 \
--update-freq 16 \
--user-dir $PROJECT_PATH/knnbox/models \
--arch "robust_knn_mt@transformer_wmt19_de_en" \
--knn-mode "train_metak" \
--knn-datastore-path $DATASTORE_LOAD_PATH \
--knn-max-k $MAX_K \
--knn-combiner-path $SAVE_DIR \
--robust-training-sigma 0.01 \
--robust-training-alpha0 1.0 \
--robust-training-beta 1000 

