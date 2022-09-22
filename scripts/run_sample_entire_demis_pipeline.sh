#!/bin/sh

# See README for the description.

# 1. Generate weak labels
sh scripts/run_weak_annotator.sh
# 2. Make weak labeled data balanced
sh scripts/run_get_balanced_weak_labeled_data.sh
# 3. Get sentence embedding for string labeled data
sh scripts/run_sim_sentence_embedding_strong_labeled.sh
# 4. Prepare state information
sh scripts/run_build_info_for_RL_state.sh
# 5. Generate sentence embedding for weak labels
sh scripts/run_sim_sentence_embedding_weak_labeled.sh
# 6. train the detection model using DeMis method
sh scripts/run_train_demis.sh