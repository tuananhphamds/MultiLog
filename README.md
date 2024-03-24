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
|                    |    HDFS   |  BGL  | Thunderbird |  Average  |
|--------------------|:---------:|:-----:|:-----------:|:---------:|
| _Supervised_       |           |       |             |           |
| LogRobust          |   99.00   |   -   |      -      |     -     |
| HitAnomaly         |   98.00   | 92.00 |      -      |     -     |
| LightLog           |   97.00   | 97.20 |      -      |     -     |
| NeuralLog          |   98.00   | 98.00 |    96.00    |   97.33   |
| _Semi-suprervised_ |           |       |             |           |
| DeepLog*           |   95.05   | 97.86 |    84.64    |   92.51   |
| LogAnomaly*        |   95.39   | 97.85 |    82.80    |   92.01   |
| LogBERT*           |   87.17   | 99.01 |    98.88    |   95.02   |
| LAnoBERT           |   96.45   | 87.49 |  **99.90**  |   94.61   |
| MultiLog(Next)     | **98.08** | 98.03 |    82.79    |   92.97   |
| MultiLog(Mask)     |   93.33   | 98.93 |    98.65    | **96.97** |
| MultiLog(Center)   |   72.05   | 83.26 |    79.52    |   78.28   |


Reevaluation results on Thunderbird dataset
| Original | Scenario 1 | Scenario 2 |
|:--------:|:----------:|:----------:|
|   82.79  |    99.02   |    99.84   |

Dimension reduction for SBERT embedding is inspired by: [dimensionreduction](https://github.com/UKPLab/sentence-transformers/blob/master/examples/training/distillation/dimensionality_reduction.py)
