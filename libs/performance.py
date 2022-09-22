###################################################
# Performance Class
#
# A class contains methods for evaluation and
# getting performance results
#
###################################################

__author__ = "Kornraphop Kawintiranon"
__email__ = "kk1155@georgetown.com"

import collections
import numpy as np
import pandas as pd

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_fscore_support


def accuracy_precision_recall_fscore_support(
    Y_test, Y_pred, n_labels=None, average=None, labels=None,
    pos_label=None, use_minority_as_pos_label=True):
    """
    Calculate performance for given actual and predicted labels

    Args:
        Y_test:
            A list of actual labels
        Y_pred:
            A list of predicted labels

    Return:
        A list of performance including accuracy, precision, recall, f1
    """

    a = accuracy_score(Y_test, Y_pred)

    if not labels:
        labels = list(set(Y_test).union(set(Y_pred)))
    labels.sort()

    if n_labels is None:
        n_labels = len(labels)
    else:
        if n_labels != len(labels):
            raise ValueError(f"`n_labels` mismatched. Found Y_test:{set(Y_test)} and  y_pred: {set(Y_pred)}")

    if n_labels <= 2:
        p, r, f, _ = precision_recall_fscore_support(
            Y_test, Y_pred, average=average, labels=labels)

        # Minority class as positive label
        if pos_label is not None and use_minority_as_pos_label:
            pos_label = collections.Counter(Y_test).most_common()[-1][0]

            p = p[pos_label]
            r = r[pos_label]
            f = f[pos_label]

        if pos_label:
            p = p[pos_label]
            r = r[pos_label]
            f = f[pos_label]
    else:
        p, r, f, _ = precision_recall_fscore_support(
            Y_test, Y_pred, average=average, labels=labels)

    result = [a, p, r, f]
    result = [
        x.tolist() if type(x).__module__==np.__name__ else x for x in result]

    return result


def get_average_result(result_df_list):
    columns = ['model', 'accuracy', 'precision', 'recall', 'f1']
    avg_result_df = pd.DataFrame(columns=columns)
    model_names = list(result_df_list[0]['model'])  # all model names

    ######################
    #   Average result   #
    ######################
    #
    # Iterate thorugh each model
    for idx, model_name in enumerate(model_names):
        sum_a, sum_p, sum_r, sum_f = 0, 0, 0, 0  # Start with zero

        # Iterate through all
        for result_df in result_df_list:
                model_result_df = result_df[result_df['model'] == model_name]
                sum_a += float(model_result_df['accuracy'])
                sum_p += float(model_result_df['precision'])
                sum_r += float(model_result_df['recall'])
                sum_f += float(model_result_df['f1'])

        # Get average result from a model
        avg_a = sum_a / len(result_df_list)
        avg_p = sum_p / len(result_df_list)
        avg_r = sum_r / len(result_df_list)
        avg_f = sum_f / len(result_df_list)

        # Append result of a model
        avg_result_df.loc[idx] = [model_name, avg_a, avg_p, avg_r, avg_f]

    return avg_result_df

def main():
    # Test accuracy_precision_recall_fscore_support()
    A = [1,2,1,3]
    B = [1,2,1,2]

    print(accuracy_precision_recall_fscore_support(A, B))

if __name__ == "__main__":
    main()