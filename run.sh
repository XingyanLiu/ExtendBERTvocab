bert_dir_in="../PreModels/chinese_L-12_H-768_A-12"
bert_dir_out="../PreModels/chinese_L-12_H-768_A-12_extended"
n_extend=10

python extend_bert_vocab.py \
  --bert_dir_in="$bert_dir_in" \
  --bert_dir_out="$bert_dir_out" \
  --n_extend=$n_extend \
  --model_name_in="bert_model" \
  --model_name_out="bert_model" \
  --token_fmt="[extended{}]" \
  --random_seed=0
