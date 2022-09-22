# DeMis: Data-efficient Misinformation Detection using RL
Resources for misinformation detection on Twitter. This repo is the official resource of the following paper.
- *DeMis: Data-efficient Misinformation Detection using Reinforcement Learning*, ECML-PKDD 2022.
- [[Paper](https://drive.google.com/file/d/1oQL5R5YiaO3Wdj6o7Nqd7BVAN2kSMxN8/view?usp=sharing)][[Slide](https://drive.google.com/file/d/1S9UUctw6rHw28FOk6zv1zaimLojcONIv/view?usp=sharing)]

<img width="540" alt="overview-model" src="https://user-images.githubusercontent.com/15230011/191144467-604bcdd8-a21a-4391-a85e-245225a67c6b.png">

## 📚 Data Sets
The data sets about COVID-19 misinformation on Twitter presented in [our paper](https://drive.google.com/file/d/1oQL5R5YiaO3Wdj6o7Nqd7BVAN2kSMxN8/view?usp=sharing) are available below.

- COMYTH (weather & home-remedies) - [[Datasets](https://portals.mdi.georgetown.edu/public/misinformation-detection)]
- COVIDLies - [[Paper](https://aclanthology.org/2020.nlpcovid19-2.11/)]

<img width="540" alt="image" src="https://user-images.githubusercontent.com/15230011/191144727-37843f6d-67ac-4180-8670-1b39558142fe.png">

## 🚀 Pre-trained Models
We release our models for misinformation detection on Twitter trained using DeMis method. There are three models trained on three COVID-19 misinformation data sets separately. All trained misinformation detection models are available on my [Google Drive](https://drive.google.com/drive/folders/1czX7oh_pvQaYsY3w-RjPZrZEE9v_d8dW?usp=sharing) 🤗 so you can download models via PyTorch and use it for prediction right away!!!

- [DeMis-COMYTH-W](https://drive.google.com/file/d/1x7AAP7aw9KzPtz0JC8T_XYCpV2dFMs9e/view?usp=sharing) (trained on COVID-weather data)
- [DeMis-COMYTH-H](https://drive.google.com/file/d/19n02CFvEbQJ2hRL9noU3vVLCIqY-WtjW/view?usp=sharing) (trained on COVID-home-remedies data)
- [DeMis-COVIDLies](https://drive.google.com/file/d/14Hc5IhYqKI5fxNkLZqgnQ8KwO-waGtSu/view?usp=sharing) (trained on COVIDLies data)

## ⚙️ Usage
We tested in `pytorch v1.10.1` and `transformers v4.18.0`.

### Usage 1: Preprocessing tweets
Specify the input and output filepaths in the shell script `run_tweet_preprocessing.sh` and run the following command.
```shell
sh scripts/run_tweet_preprocessing.sh
```

### Usage 2: Run the detection model for classification
Specify the model path, input and output filepaths in the shell script `run_detector.sh`. Note that you can download the models from the section above and try running it with the following command, or you can train a new detection model using DeMis in the next section.
```shell
sh scripts/run_detector.sh
```

### Usage 3: Train detector using DeMis
1. Run script `run_weak_annotator.sh` to generate weak labels based on similarity between unlabeled tweets and claims. Note that this will also generate similarity matrix so we can reuse it for other myth themes by adding `similarity_matrix_filepath` argument to the shell script.
> **WARNING**: This step can take up to an hour depending on the size of unlabeled data.
```shell
sh scripts/run_weak_annotator.sh
```
2. Run `run_get_balanced_weak_labeled_data.sh` to build a balanced set of weak labeled data.
```shell
sh scripts/run_get_balanced_weak_labeled_data.sh
```
3. Run `run_sim_sentence_embedding_strong_labeled.sh` to generate similarity matrix between strong-labeled tweets and claims. No need to generate similarity matrix for unlabeled tweets since it is already done in the step 1.
```shell
sh scripts/run_sim_sentence_embedding_strong_labeled.sh
```
4. Run `run_build_info_for_RL_state.sh` to build state information. Both stong and weak labeled data must have been proceesed.
```shell
sh scripts/run_build_info_for_RL_state.sh
```
5. Run `run_generate_sentence_embeddings.sh` to create sentence embedding matrix for calculating max/min similarity during the state creation as an element of state info.
```shell
sh scripts/run_generate_sentence_embeddings.sh
```
6. Run `train_demis.sh` to train the model.
> **WARNING**: With the actual setting described in the paper, this step can take a few hours to finish, therefore, the setting in the script is used for testing purpose only.
```shell
sh scripts/train_demis.sh
```

## ✏️ Citation
If you feel our paper and resources are useful, please consider citing our work! 🙏
```bibtex
@inproceedings{kawintiranon2022demis,
  title     = {DeMis: Data-efficient Misinformation Detection using Reinforcement Learning},
  author    = {Kawintiranon, Kornraphop and Singh, Lisa},
  booktitle = {Proceedings of the European Conference on Machine Learning and Principles and Practice of Knowledge Discovery in Databases (ECML-PKDD)},
  year      = {2022},
  publisher = {Springer}
}
```

##  🛠 Throubleshoots
[Create an issue here](https://github.com/GU-DataLab/misinformation-detection-DeMis/issues) if you have any issues loading models or data sets.
