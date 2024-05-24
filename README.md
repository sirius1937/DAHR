# DAHR: Depth Aware Hierarchical Replay For Continual Knowledge Based Question Answering (COLING 2024)



### Introduction

We propose a replay algorithm for knowledge question answering, which better preserves previously learned knowledge. This method is intuitive, simple but effective. Our work was published in COLING 2024.



<img src="./resource/main_figure.png" alt="main_fig" style="zoom:67%;" />

We share this document and code for the following purposes:

- Provide complete and reproducible code and experimental results to ensure the effectiveness of our method. The experimental part of the article cannot fully describe the implementation details of our code.
- Expanded several sets of ablation experiments, completed our conclusions, and corrected minor issues or incomplete parts in the paper.



### Experimental environment

Tesla V100-SXM2-32GB * 2

Python 3.10



#### Experiment setting

| Parameters      | Val           |
| --------------- | ------------- |
| Model           | Electra-small |
| Learning_rate   | 5e-5          |
| Learning_epochs | 5             |
| Random_seed     | 42            |

##### Experiment Command

| Command                                                      | Meaning                           |
| ------------------------------------------------------------ | --------------------------------- |
| python main.py  --replay_method DAHR_md  --save_log_path ./final/br2clu100_42_dep --branch 2 --cluster_size 100 --use_nvidia 0-1 --batch_size 50 --train_epochs 5 --default_times 1 --choice_cnt 40 --random_seed 42 --desc LASTsed42 | Main_result                       |
| python main.py  --replay_method DAHR_md  --save_log_path ./final/br3clu100_42_dep --branch 3 --cluster_size 100 --use_nvidia 0-1 --batch_size 50 --train_epochs 5 --default_times 1 --choice_cnt 40 --random_seed 42 --desc LASTsed42 | Ablation for 'branch = 3'         |
| python main.py  --replay_method DAHR_md  --save_log_path ./final/br4clu100_42_dep --branch 4 --cluster_size 100 --use_nvidia 0-1 --batch_size 50 --train_epochs 5 --default_times 1 --choice_cnt 40 --random_seed 42 --desc LASTsed42 | Ablation for 'branch = 4'         |
| python main.py  --replay_method DAHR_md  --save_log_path ./final/br2clu200_42_dep --branch 2 --cluster_size 200 --use_nvidia 0-1 --batch_size 50 --train_epochs 5 --default_times 1 --choice_cnt 40 --random_seed 42 --desc LASTsed42 | Ablation for 'cluster_size = 200' |
| python main.py  --replay_method DAHR_md  --save_log_path ./final/br2clu400_42_dep --branch 2 --cluster_size 400 --use_nvidia 0-1 --batch_size 50 --train_epochs 5 --default_times 1 --choice_cnt 40 --random_seed 42 --desc LASTsed42 | Ablation for 'cluster_size = 400' |
| python main.py  --replay_method rd  --save_log_path ./final/br2clu100_rd --branch 2 --cluster_size 100 --use_nvidia 0-1 --batch_size 50 --train_epochs 5 --default_times 1 --choice_cnt 40 --random_seed 42 --desc LASTsed42 | RMR_baseline                      |
| python main.py  --replay_method DAHR_md  --save_log_path ./final/br2clu100_42_dep --branch 2 --cluster_size 100 --use_nvidia 0-1 --batch_size 50 --train_epochs 5 --default_times 1 --choice_cnt 40 --random_seed 42 --desc LASTsed42 --sample_percent 0.1 | Main_0.1                          |
| python main.py  --replay_method rd  --save_log_path ./final/br2clu100_rd --branch 2 --cluster_size 100 --use_nvidia 0-1 --batch_size 50 --train_epochs 5 --default_times 1 --choice_cnt 40 --random_seed 42 --desc LASTsed42  --sample_percent 0.1 | RMR_0.1                           |

