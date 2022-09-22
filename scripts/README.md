# Usage and Commands

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
5. Run `run_sim_sentence_embedding_weak_labeled.sh` to create sentence embedding matrix for weak labeled data in order to calculate max/min similarity during the state creation as an element of state info.
```shell
sh scripts/run_sim_sentence_embedding_weak_labeled.sh
```
6. Run `run_train_demis.sh` to train the model.
> **WARNING**: With the actual setting described in the paper, this step can take a few hours to finish, therefore, the setting in the script is used for testing purpose only.
```shell
sh scripts/run_train_demis.sh
```