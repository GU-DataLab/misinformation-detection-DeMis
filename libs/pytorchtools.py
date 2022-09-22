"""
    Additional tools for pytorch

    Inspired by https://github.com/Bjarten/early-stopping-pytorch
"""

import time
from numpy.core.fromnumeric import mean
import tqdm
import torch
import warnings

from loguru import logger
import numpy as np

from libs.performance import accuracy_precision_recall_fscore_support


class EarlyStoppingByKen:
    """Early stops the training if the specified measurement doesn't improve after a given patience.
    
        Note that improvement could be either increasing nor decreasing
    """

    def __init__(self, patience=7, verbose=False, delta=0, checkpoint_path=None, improve_measure=-1):
        """
        Args:
            patience (int):
                How long to wait after last time validation loss improved.
                Default: 7
            verbose (bool):
                If True, prints a message for each validation loss improvement. 
                Default: False
            delta (float):
                Minimum change in the monitored quantity to qualify as an improvement.
                Default: 0
            checkpoint_path (str):
                Path for the checkpoint to be saved to.
                Default: None
            improve_measure (int):
                Indicate how to compare scores to measure improvement.
                Negative value means lower score is better while positive value means higher score is better.
                Default: -1

                Eg. loss (-1) or accuracy (+1)
        """

        if patience <= 0:
            raise ValueError("patience must be positive integer")

        self.patience = abs(patience)
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_val = np.Inf if improve_measure < 0 else -np.Inf
        self.delta = delta
        self.checkpoint_path = checkpoint_path
        self.improve_measure = improve_measure
        self.call_number = 0

    def __call__(self, val, model, checkpoint_path=None):
        self.call_number += 1

        if checkpoint_path != None:  # update checkpoint path
            self.checkpoint_path = checkpoint_path

        score = val * (self.improve_measure / abs(self.improve_measure))

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(
                    f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val, model)
            self.counter = 0

    def save_checkpoint(self, val, model):
        '''Saves model when validation score is better.'''
        if self.checkpoint_path != None and len(self.checkpoint_path) > 0:
            if self.verbose:
                print(
                    f'Validation is better ({self.best_val:.6f} --> {val:.6f}).  Saving model ...')
            torch.save(model.state_dict(), self.checkpoint_path)
            self.best_val = val


def train_one_epoch(
    model, train_dataloader, optimizer, criterion, device,
    eval_pos_label_only=True, verbose=True, all_possible_labels=[0, 1]):
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if device.type == 'cuda':
    #     model.cuda()
    model.to(device)
    model.train()
    train_sum_loss = 0.0
    train_sum_correct = 0
    if eval_pos_label_only:
        train_sum_precision = 0.0
        train_sum_recall = 0.0
        train_sum_f1 = 0.0
    else:
        train_sum_precision = {c: 0.0 for c in all_possible_labels}
        train_sum_recall = {c: 0.0 for c in all_possible_labels}
        train_sum_f1 = {c: 0.0 for c in all_possible_labels}
    counter = 0
    total = 0
    prog_bar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader)) if verbose else enumerate(train_dataloader)
    for i, batch in prog_bar:
        counter += 1
        # print(batch)
        data, mask, target = batch
        if device.type == 'cuda':
            data, mask, target = data.to(device), mask.to(device), target.to(device)
            # # Exception for text input
            # if not isinstance(data, tuple):
            #     data = data.to(device)
            # target = target.to(device)
        total += target.size(0)
        optimizer.zero_grad()
        outputs = model(data, mask)
        loss = criterion(outputs, target)
        train_sum_loss += loss.item()
        _, preds = torch.max(outputs.data, 1)
        train_sum_correct += (preds == target).sum().item()
        loss.backward()
        optimizer.step()

        # Compute performance
        # zero_division = "warn" if verbose else 0 
        _, p, r, f = accuracy_precision_recall_fscore_support(
            target.tolist(), preds.tolist(), labels=[0, 1])
        if eval_pos_label_only:
            train_sum_precision += p[minority_class] if isinstance(p, list) else p
            train_sum_recall += r[minority_class] if isinstance(r, list) else r
            train_sum_f1 += f[minority_class] if isinstance(f, list) else f
        else:
            for c in all_possible_labels:
                train_sum_precision[c] += p[c] if isinstance(p, list) else p
                train_sum_recall[c] += r[c] if isinstance(r, list) else r
                train_sum_f1[c] += f[c] if isinstance(f, list) else f

    if criterion:
        train_loss = train_sum_loss / counter
    train_accuracy = 100. * train_sum_correct / total
    if eval_pos_label_only:
        train_precision = train_sum_precision / counter
        train_recall = train_sum_recall / counter
        train_f1 = train_sum_f1 / counter
        results = {
            "precision": train_precision,
            "recall": train_recall,
            "f1": train_f1,
        }

    else:
        train_precision = {c: train_sum_precision[c] / counter for c in all_possible_labels}
        train_recall = {c: train_sum_recall[c] / counter for c in all_possible_labels}
        train_f1 = {c: train_sum_f1[c] / counter for c in all_possible_labels}
        results = {
            c: {
                "precision": train_precision[c],
                "recall": train_recall[c],
                "f1": train_f1[c]
            } for c in all_possible_labels
        }
    results["accuracy"] = train_accuracy
    if criterion:
        results["loss"] = train_loss

    return results
    # return train_loss, train_accuracy


def validate_one_epoch(model, test_dataloader, criterion, device, verbose=True):
    raise NotImplementedError("Use 'evaluate_model' instead.")
#     model.eval()
#     val_sum_loss = 0.0
#     val_sum_correct = 0
#     counter = 0
#     total = 0
#     prog_bar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)) if verbose else enumerate(test_dataloader)
#     with torch.no_grad():
#         for i, batch in prog_bar:
#             counter += 1
#             data, target = batch
#             if device.type == 'cuda':
#                 data, target = data.to(device), target.to(device)
#             total += target.size(0)
#             outputs = model(data)
#             loss = criterion(outputs, target)

#             val_sum_loss += loss.item()
#             _, preds = torch.max(outputs.data, 1)
#             val_sum_correct += (preds == target).sum().item()

#         val_loss = val_sum_loss / counter
#         val_accuracy = 100. * val_sum_correct / total
#         return val_loss, val_accuracy


def evaluate_model(
    model, test_dataloader, criterion=None, device=None,
    eval_pos_label_only=True, get_predictions=False, verbose=True,
    get_prediction_probs=False, all_possible_labels=[0, 1], minority_class=None):

    if minority_class is None:
        minority_class = 1
    
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type == 'cuda':
        model.cuda()
    
    if get_prediction_probs:
        my_softmax = torch.nn.Softmax(dim=1)

    model.eval()
    
    # Store results
    val_sum_correct = 0
    val_sum_loss = 0.0
    if eval_pos_label_only:
        val_sum_precision = 0.0
        val_sum_recall = 0.0
        val_sum_f1 = 0.0
    else:
        val_sum_precision = {c: 0.0 for c in all_possible_labels}
        val_sum_recall = {c: 0.0 for c in all_possible_labels}
        val_sum_f1 = {c: 0.0 for c in all_possible_labels}

    predictions = []
    prediction_probs = []
    counter = 0
    total = 0
    prog_bar = tqdm.tqdm(enumerate(test_dataloader), total=len(test_dataloader)) if verbose else enumerate(test_dataloader)
    with torch.no_grad():
        for i, batch in prog_bar:
            counter += 1
            data, mask, target = batch
            if device.type == 'cuda':
                data, mask, target = data.to(device), mask.to(device), target.to(device)
                # # Exception for text input
                # if not isinstance(data, tuple):
                #     data = data.to(device)
                # target = target.to(device)
            total += target.size(0)
            outputs = model(data, mask)
            if criterion:
                loss = criterion(outputs, target)

            val_sum_loss += loss.item()
            _, preds = torch.max(outputs.data, 1)
            val_sum_correct += (preds == target).sum().item()

            if get_predictions:
                predictions.append(preds)
            if get_prediction_probs:
                # Consider prob of class 1 (positive class)
                b_output_softmax = my_softmax(outputs.data)
                b_predict_probs = torch.index_select(
                    b_output_softmax.to("cpu"), 1, torch.tensor([1]))
                prediction_probs.append(b_predict_probs)

            # zero_division = "warn" if verbose else 0 
            _, p, r, f = accuracy_precision_recall_fscore_support(
                target.tolist(), preds.tolist(), labels=all_possible_labels)

            if eval_pos_label_only:
                val_sum_precision += p[minority_class] if isinstance(p, list) else p
                val_sum_recall += r[minority_class] if isinstance(r, list) else r
                val_sum_f1 += f[minority_class] if isinstance(f, list) else f
            else:
                for c in all_possible_labels:
                    val_sum_precision[c] += p[c] if isinstance(p, list) else p
                    val_sum_recall[c] += r[c] if isinstance(r, list) else r
                    val_sum_f1[c] += f[c] if isinstance(f, list) else f

        if criterion:
            val_loss = val_sum_loss / counter
        val_accuracy = 100. * val_sum_correct / total

        if eval_pos_label_only:
            val_precision = val_sum_precision / counter
            val_recall = val_sum_recall / counter
            val_f1 = val_sum_f1 / counter
        else:
            val_precision = {c: val_sum_precision[c] / counter for c in all_possible_labels}
            val_recall = {c: val_sum_recall[c] / counter for c in all_possible_labels}
            val_f1 = {c: val_sum_f1[c] / counter for c in all_possible_labels}

    if eval_pos_label_only:
        results = {
            "precision": val_precision,
            "recall": val_recall,
            "f1": val_f1
        }
    else:
        results = {
            c: {
                "precision": val_precision[c],
                "recall": val_recall[c],
                "f1": val_f1[c]
            } for c in all_possible_labels
        }
    results["accuracy"] = val_accuracy

    if criterion:
        results["loss"] = val_loss

    # print(torch.cuda.memory_allocated())
    # print(torch.cuda.memory_reserved())
    # torch.cuda.empty_cache()
    # print(torch.cuda.memory_allocated())
    # print(torch.cuda.memory_reserved())

    if get_predictions:
        # Flatten the predictions and true values for aggregate Matthew's evaluation on the whole dataset
        flat_predictions = [item.item() for sublist in predictions for item in sublist]
        # flat_predictions = np.argmax(flat_predictions, axis=1).flatten()
        results["predictions"] = flat_predictions
    if get_prediction_probs:
        flat_prediction_probs = [item.item() for sublist in prediction_probs for item in sublist]
        results["prediction_probs"] = flat_prediction_probs

    return results


def train_model(model, train_dataloader, val_dataloader, num_epochs, optimizer,
                criterion, stop_when_metric_change=None, baseline_matric_value=None,
                load_best_model=True, early_stop=-3, device=None, eval_pos_label_only=True,
                best_model_measure="loss", verbose=True, all_possible_labels=[0, 1],
                eval_warning=None, minority_class=None):
    # Ref: https://debuggercafe.com/using-learning-rate-scheduler-and-early-stopping-with-pytorch/

    """
    Args:
        stop_when_metric_change (str): A metric to be considered for stopping the
            training. This is designed to avoid zero reward in RL. Must be used
            together with `baseline_matric_value`.
        baseline_matric_value (float): A baseline metric value to be considered
            for stopping the training. This is designed to avoid zero reward in RL.
            Must be used together with `stop_when_metric_change`.
        load_best_model (bool): Whether load the best model or not. Default is True.
            If False, the latest model will be the last trained model, not the best
            model. Best model means the best model with the least loss.
        early_stop (int): Whether to use early stopping. Negative value means using
            loss as a criteria to stop. Positive value means the accuracy is used.
            Zero or None value means no early stopping.
        minority_class (int): The minority class that the model will focus on optimizing.
    """

    if minority_class is None:
        minority_class = 1

    if eval_warning:
        warnings.filterwarnings(eval_warning)

    if (stop_when_metric_change and baseline_matric_value is None) \
        or (stop_when_metric_change is None and baseline_matric_value):
        raise ValueError(
            f"stop_when_metric_change and baseline_matric_value "
            f"must be used together to perform the task. Found "
            f"stop_when_metric_change: {stop_when_metric_change} "
            f"and baseline_matric_value: {baseline_matric_value}")

    if early_stop:
        early_stop = int(early_stop)

    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # if device.type == 'cuda':
    #     model.cuda()

    # Initialize early stopper
    if early_stop != None:
        early_stopper = EarlyStoppingByKen(
            patience=abs(early_stop),
            verbose=False,
            delta=0,
            checkpoint_path=None,
            improve_measure=int(early_stop/abs(early_stop)))  # -1 for loss, 1 for accuracy

    # Keep performance results
    if eval_pos_label_only:
        train_results = {
            "precision": [],
            "recall": [],
            "f1": []
        }
        val_results = {
            "precision": [],
            "recall": [],
            "f1": []
        }
    else:
        train_results = {
            c: {
                "precision": [],
                "recall": [],
                "f1": []
            } for c in all_possible_labels
        }
        val_results = {
            c: {
                "precision": [],
                "recall": [],
                "f1": []
            } for c in all_possible_labels
        }
    train_results["loss"] = []
    train_results["accuracy"] = []
    val_results["loss"] = []
    val_results["accuracy"] = []

    sum_val_loss = 0
    sum_val_acc = 0
    counter = 0
    start = time.time()

    # To select best model
    best_val_epoch_loss = float("inf")
    best_val_epoch_f1 = float("-inf")
    if load_best_model:
        # TODO: save the original as the initial best model
        # val_original_results = evaluate_model(
        #     model, val_dataloader, criterion=criterion, device=device,
        #     eval_pos_label_only=eval_pos_label_only, verbose=verbose,
        #     all_possible_labels=all_possible_labels
        # )
        best_val_epoch_idx = 0
        best_model_params = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                best_model_params[name] = param.detach().clone()
    # best_optimizer_params = {}
    # for name, param in optimizer.named_parameters():
    #     if param.requires_grad:
    #         best_optimizer_params[name] = param.detach().clone()

    prog_bar = tqdm.trange(num_epochs, desc="Train model") if verbose else range(num_epochs)
    for epoch in prog_bar:
        train_epoch_results = train_one_epoch(
            model, train_dataloader, optimizer=optimizer,
            criterion=criterion, device=device, verbose=False,
            eval_pos_label_only=eval_pos_label_only,
            all_possible_labels=all_possible_labels
        )
        val_epoch_results = evaluate_model(
            model, val_dataloader, criterion=criterion, device=device,
            eval_pos_label_only=eval_pos_label_only, verbose=False,
            all_possible_labels=all_possible_labels
        )

        # Train results
        train_results["loss"].append(train_epoch_results["loss"])
        train_results["accuracy"].append(train_epoch_results["accuracy"])
        if eval_pos_label_only:
            train_results["precision"].append(train_epoch_results["precision"])
            train_results["recall"].append(train_epoch_results["recall"])
            train_results["f1"].append(train_epoch_results["f1"])
        else:
            for c in all_possible_labels:
                train_results[c]["precision"].append(train_epoch_results[c]["precision"])
                train_results[c]["recall"].append(train_epoch_results[c]["recall"])
                train_results[c]["f1"].append(train_epoch_results[c]["f1"])

        # Validation results
        val_results["loss"].append(val_epoch_results["loss"])
        val_results["accuracy"].append(val_epoch_results["accuracy"])
        if eval_pos_label_only:
            val_results["precision"].append(val_epoch_results["precision"])
            val_results["recall"].append(val_epoch_results["recall"])
            val_results["f1"].append(val_epoch_results["f1"])
        else:
            for c in all_possible_labels:
                val_results[c]["precision"].append(val_epoch_results[c]["precision"])
                val_results[c]["recall"].append(val_epoch_results[c]["recall"])
                val_results[c]["f1"].append(val_epoch_results[c]["f1"])

        sum_val_loss += val_epoch_results["loss"]
        sum_val_acc += val_epoch_results["accuracy"]
        counter += 1

        if best_model_measure not in ["loss", "f1"]:
            raise ValueError(f"`best_val_epoch_f1` must be loss/f1. "
                             f"Found {best_model_measure}.")
        # Consider F1 of class-1
        epoch_loss = val_epoch_results["loss"]
        epoch_f1 = val_epoch_results["f1"] if eval_pos_label_only else val_epoch_results[minority_class]["f1"]
        epoch_precision = val_epoch_results["precision"] if eval_pos_label_only else val_epoch_results[minority_class]["precision"]
        epoch_recall = val_epoch_results["recall"] if eval_pos_label_only else val_epoch_results[minority_class]["recall"]

        # Consider if the best loss
        if best_model_measure == "loss" and epoch_loss <= best_val_epoch_loss:
            best_val_epoch_loss = epoch_loss
            best_val_epoch_idx = epoch

            # Store best model
            if load_best_model:
                best_model_params = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        best_model_params[name] = param.detach().clone()
            # for name, param in optimizer.named_parameters():
            #     if param.requires_grad:
            #         best_optimizer_params[name] = param.detach().clone()

        # Consider if the best f1
        elif best_model_measure == "f1" and (epoch_f1 >= best_val_epoch_f1) or \
            (epoch_f1 == best_val_epoch_f1 and epoch_loss < best_val_epoch_loss):
            best_val_epoch_f1 = epoch_f1
            best_val_epoch_idx = epoch

            # Store best model
            if load_best_model:
                best_model_params = {}
                for name, param in model.named_parameters():
                    if param.requires_grad:
                        best_model_params[name] = param.detach().clone()

        # Call early stop to verify new value but not save
        if early_stop:
            if early_stop > -1:
                early_stopper(val=sum_val_acc / counter, model=None, checkpoint_path=None)
            else:
                early_stopper(val=sum_val_loss / counter, model=None, checkpoint_path=None)

            # If need to do early stop
            if early_stopper.early_stop:
                break

        if verbose:
            print(f"Train Loss: {train_epoch_results['loss']:.4f}, Train Acc: {train_epoch_results['accuracy']:.2f}")
            print(f"Val Loss: {val_epoch_results['loss']:.4f}, Val Acc: {val_epoch_results['accuracy']:.2f}")
            print(f"Val Precision: {epoch_precision:.4f}")
            print(f"Val Recall: {epoch_recall:.4f}")
            print(f"Val F1: {epoch_f1:.4f}")

        if early_stop and early_stopper.early_stop:
            logger.debug(f"Early stop at epoch: {early_stopper.call_number}")
            if verbose:
                print(f"Train Loss: {train_epoch_results['loss']:.4f}, Train Acc: {train_epoch_results['accuracy']:.2f}")
                print(f"Val Loss: {val_epoch_results['loss']:.4f}, Val Acc: {val_epoch_results['accuracy']:.2f}")

        # Stop when metric change to avoid zero reward for RL
        # Now only support accuracy and F1
        if stop_when_metric_change:
            if stop_when_metric_change in ["accuracy", "f1"]:
                tmp_score = val_epoch_results[stop_when_metric_change] if eval_pos_label_only else val_epoch_results[minority_class][stop_when_metric_change]
                if abs(baseline_matric_value - tmp_score) > 1e-6:
                    logger.info(f"Stop when metric change at epoch: {epoch}")
                    break
            else:
                raise ValueError(f"stop_when_metric_change is incorrect."
                                 f" Found {stop_when_metric_change}")

    # Restore the saved best model
    if load_best_model:
        with torch.no_grad():
            for name, current_weight in model.named_parameters():
                current_weight.copy_(best_model_params[name])
        # for name, current_weight in optimizer.named_parameters():
        #     current_weight.copy_(best_optimizer_params[name])

    end = time.time()
    if verbose:
        print(f"Training time: {(end-start)/60:.3f} minutes")

    # If not loaded the best model then the result of the saved model
    # is the latest model we have trained. The result is the last element.
    if not load_best_model:
        best_val_epoch_idx = -1

    # Pack results together
    if eval_pos_label_only:
        results = {
            # Train results
            "train_loss": train_results["loss"][best_val_epoch_idx],
            "train_accuracy": train_results["accuracy"][best_val_epoch_idx],
            "train_precision": train_results["precision"][best_val_epoch_idx],
            "train_recall": train_results["recall"][best_val_epoch_idx],
            "train_f1": train_results["f1"][best_val_epoch_idx],
            "train_history": train_results,
            # Validation results
            "val_loss": val_results["loss"][best_val_epoch_idx],
            "val_accuracy": val_results["accuracy"][best_val_epoch_idx],
            "val_precision": val_results["precision"][best_val_epoch_idx],
            "val_recall": val_results["recall"][best_val_epoch_idx],
            "val_f1": val_results["f1"][best_val_epoch_idx],
            "val_history": val_results
        }
    else:
        results = {
            c: {
                # Train results
                "train_precision": train_results[c]["precision"][best_val_epoch_idx],
                "train_recall": train_results[c]["recall"][best_val_epoch_idx],
                "train_f1": train_results[c]["f1"][best_val_epoch_idx],
                # Validation results
                "val_precision": val_results[c]["precision"][best_val_epoch_idx],
                "val_recall": val_results[c]["recall"][best_val_epoch_idx],
                "val_f1": val_results[c]["f1"][best_val_epoch_idx],
            } for c in all_possible_labels
        }
        results["train_loss"] = train_results["loss"][best_val_epoch_idx]
        results["train_accuracy"] = train_results["accuracy"][best_val_epoch_idx]
        results["val_loss"] = val_results["loss"][best_val_epoch_idx]
        results["val_accuracy"] = val_results["accuracy"][best_val_epoch_idx]
        results["train_history"] = train_results
        results["val_history"] = val_results

    # Change back to default if changed beat the beginning of this function
    if eval_warning:
        warnings.filterwarnings("default")  # "error", "ignore", "always", "default", "module" or "once"

    return results


class AscentFunction(torch.autograd.Function):
    # Ref: https://discuss.pytorch.org/t/gradient-ascent-and-gradient-modification-modifying-optimizer-instead-of-grad-weight/62777/5
    @staticmethod
    def forward(ctx, input):
        return input

    @staticmethod
    def backward(ctx, grad_input):
        return -grad_input


def make_ascent(loss):
    return AscentFunction.apply(loss)


# Test
if __name__ == "__main__":
    early_stopper = EarlyStoppingByKen(
        patience=3, verbose=False, delta=0, checkpoint_path=None, improve_measure=1)

    # arr = [0, 1, 0, 1, 3, 4, 5, 6, 7, 8, 9, 10]
    arr = [10, 9, 10, 9, 10, 9, 8, 7, 6, 5, 4, 3]

    for val in arr:
        early_stopper(val=val, model=None)
        print(val, early_stopper.early_stop)
        if early_stopper.early_stop:
            break
