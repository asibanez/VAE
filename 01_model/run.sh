#INPUT_DIR=C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/00_data/01_preprocessed
#OUTPUT_DIR=C:/Users/siban/Dropbox/CSAIL/Projects/17_VAE/00_spy_project/00_data/02_runs/00_TEST_DELETE

INPUT_DIR=/home/sibanez/Projects/02_VAE/00_data/01_preprocessed
OUTPUT_DIR=/home/sibanez/Projects/02_VAE/00_data/02_runs/00_TEST_DELETE

MODEL_FILENAME=model_0_AE_v0.py

python train_test.py \
    --input_dir=$INPUT_DIR \
    --output_dir=$OUTPUT_DIR \
    --model_filename=$MODEL_FILENAME \
    --task=Test \
    \
    --hidden_dim=16 \
    --seed=1234 \
    --use_cuda=True \
    \
    --n_epochs=10 \
    --batch_size_train=200 \
    --shuffle_train=False \
    --drop_last_train=False \
    --dev_train_ratio=1 \
    --train_toy_data=False \
    --len_train_toy_data=1000 \
    --lr=2e-4 \
    --wd=1e-6 \
    --dropout=0.2 \
    --momentum=0.9 \
    --save_final_model=True \
    --save_model_steps=True \
    --save_step_cliff=0 \
    --gpu_ids_train=0 \
    \
    --test_file=model_test.pt \
    --model_file=model.pt.9 \
    --batch_size_test=200 \
    --gpu_id_test=0 \

#read -p 'EOF'

#--model_name=nlpaueb/legal-bert-small-uncased \
#--hidden_dim=512 \

#--task=Train / Test
#--pooing=Avg / Max
#--batch_size=280 / 0,1,2,3
#--wd=1e-6
