# MultiLog
### Lightweight Multi-task Learning Method for System Log Anomaly Detection

Multilog is a semi-supervised learning method combining multi-task learning for log anomaly detection. Three training objectives are predicting the next log template ID, predicting masked template IDs, distance minimization between the encoded [CLS] vector and the center vector. We apply a simple yet effective PCA algorithm to reduce the embedding vector dimension. 

Main techniques in this paper:
- Pretrained language model SBERT
- Dimension reduction
- Transformer encoder
- Multitask-learning

Evaluation datasets: HDFS, BGL, Thunderbird 10M

## How to use the repository
### Clone the repository
```bash
git clone https://github.com/tuananhphamds/MultiLog.git
cd MultiLog
```

### Prepare experiment environment (GPU is needed)
- Python: 3.8, 3.9
- Tensorflow: 2.8.0, 2.9.0

### Prepare data
Download processed dataset from here: https://bit.ly/3INmDDD
Unzip to datasets folder in MultiLog

### Run the code
```bash
python run/train.py --model multilog --dataset hdfs_no_parser
```
Supported datasets: 
- hdfs_spell
- hdfs_drain
- hdfs_no_parser
- bgl_spell
- bgl_drain
- bgl_no_parser
- tbird_spell
- tbird_drain
- tbird_no_parser

Supported models:
- deeplog
- loganomaly
- logbert
- n_bert (Next training objective only)
- pn_bert (Next + Mask training objectives)
- multilog (Next + Mask + Distance training objectives)

Evaluation results will be saved in `results` folder. Make sure `results` is created.
Check `best_results.pkl` file for F1, threshold, Precision, Recall, TP, TN, FN, FP.

Paper results:
![image](https://github.com/tuananhphamds/MultiLog/assets/60763901/9a0452e8-d697-47d0-a1dc-2ce689f8f7e3)

Reevaluate Thunderbird dataset
![image](https://github.com/tuananhphamds/MultiLog/assets/60763901/d54fa485-ac4d-4dae-b614-9f4041f429d9)
