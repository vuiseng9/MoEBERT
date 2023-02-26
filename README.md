# MoEBERT
This PyTorch package implements [*MoEBERT: from BERT to Mixture-of-Experts via
Importance-Guided Adaptation*](https://arxiv.org/abs/2204.07675) (NAACL 2022).

## Installation
* Create and activate conda environment.
``` 
conda env create -f environment.yml
```
* Install Transformers locally.
* *Note:* The code is adapted from [this codebase](https://github.com/microsoft/LoRA/blob/main/examples/NLU/README.md).
Arguments regarding *LoRA* and *adapter* can be safely ignored.
```bash
# vuiseng9's
pip install -e .

pip install \
  datasets \
  loralib==0.1.1 \
  scikit-learn # requirements in text-classification \
  tensorboard
```
## Instructions
MoEBERT targets task-specific distillation. Before running any distillation code, a pre-trained BERT model should be fine-tuned on the target task.
Path to the fine-tuned model should be passed to `--model_name_or_path`.

### Importance Score Computation
* Use `bert_base_mnli_example.sh` to compute the importance scores, 
  add a `--preprocess_importance` argument, remove the `--do_train` argument.
  ```bash
  ./scripts/score_importance.sh local # vuiseng9's
  ```
* If multiple GPUs are used to compute the importance scores, a `importance_[rank].pkl` file will be saved for each GPU. 
  Use `merge_importance.py` to merge these files.
  ```bash
  python ../../merge_importance.py --num_files=4 # vuiseng9's
  ```
* To use the pre-computed importance scores, pass the file name to `--moebert_load_importance`.
  ```bash
  ./scripts/train-moebert-mnli.sh local # vuiseng9's
  ```
### Knowledge Distillation
* For GLUE tasks, see `examples/text-classification/run_glue.py`.
* For question answering tasks, see `examples/question-answering/run_qa.py`.
* Run `bash bert_base_mnli_example.sh` as an example.
* The codebase supports different routing strategies: *gate-token*, *gate-sentence*, *hash-random* and *hash-balance*. 
  Choices should be passed to `--moebert_route_method`.
  * To use *hash-balance*, a balanced hash list needs to be pre-computed using `hash_balance.py`. 
    Path to the saved hash list should be passed to `--moebert_route_hash_list`.
  * Add a load balancing loss by setting `--moebert_load_balance` when using trainable gating mechanisms.
  * The sentence-based gating mechanism (*gate-sentence*) is advantageous for inference because it 
    induces significantly less communication overhead compared with token-level routing methods.
    
