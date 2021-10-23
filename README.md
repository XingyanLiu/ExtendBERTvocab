# Extend BERT Vocabulary

扩充BERT词表

1. 只负责修改相应TF数据流图，新增token的位置，用 “[extended0], [extended1], ...” 来填充；
2. 目前需要手动将新的 vocab.txt 文件末尾填充的 "[extended0]" 修改成需要的 token。
3. 新增的 embedding 向量随机初始化，初始化方式与原生BERT相同:
    * `tf.truncated_normal_initializer(stddev=0.02)`

## Requirements

* TensorFlow <= 1.15
* numpy

## Usage

### Parameters

| param | description | default |
|-------|-------------|---------|
bert_dir_in | 原始BERT模型 checkpoint 所在文件夹 | None, required |
bert_dir_out | 导出的词表扩充后的 BERT 模型的文件夹 | "${bert_dir_in}_extended" |
n_extend | 需要增加多少个 token | 10 |
model_name_in | 原始模型的名字 | "bert_model" |
model_name_in | 扩充后模型的名字 | "bert_model" |
token_fmt | 新增的token的格式 | "[extended{}]" |

### Example

直接运行

```shell
python extend_bert_vocab.py --bert_dir_in=./chinese_L-12_H-768_A-12 --n_extend=10
```

或者在 Shell 脚本中配置更多参数：

```shell
# run.sh
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
```
