from typing import Optional, Any
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch

##########################################################################################
def torch_status():
    print("torch_version: ", torch.__version__)
    print("torch CUDA version: ", torch.version.cuda)
    print("torch CUDA available: ", torch.cuda.is_available())
    print("torch number of GPU: ", torch.cuda.device_count())


##########################################################################################
"""
    GPU Setup
"""


def gpu_setup(use_gpu: bool = True):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    if torch.cuda.is_available() and use_gpu:
        print('cuda available with GPU:', torch.cuda.get_device_name(0))
        device = torch.device("cuda")
    else:
        print('cuda not available')
        device = torch.device("cpu")
    return device


#########################################################################################
def check_directory(directory_path: str) -> None:
    """
    Checks and creats a directory_path if does not exist
    :param directory_path:
    :return:
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        print(f"Directory created: {directory_path}")

##########################################################################################
def epoch_info() -> pd.DataFrame:

    df = pd.DataFrame(
        columns=[
            "model",
            "graph",
            "epoch_number",
            "epoch_time",
            "training_loss",
            "validation_loss",
            "test_loss",
            "training_accuracy",
            "validation_accuracy",
            "test_accuracy",
        ]
    )

    return df

##########################################################################################
def append_df(df: pd.DataFrame, row: dict):
    """

    :param df:
    :param row:
    :return:
    """

    inds = len(df.index)
    for ky, val in row.items():
        df.loc[inds, ky] = val
    return df


#############################################################################################
def accuracy(pred: torch.tensor, true_label: torch.tensor, verbose: bool = False):
    """

    :param pred: prediction
    :param true_label: true labels
    :param verbose:
    :return:
    """

    pred = pred.argmax(dim=1)
    correct = (pred == true_label).sum()
    acc = int(correct) / len(true_label)

    if verbose:
        print(f'Accuracy: {acc:.4f}')
    return acc


##########################################################################################
def split_info() -> pd.DataFrame:

    df = pd.DataFrame(
        columns=[
            "model",
            "graph",
            "training_loss",
            "validation_loss",
            "test_loss",
            "training_accuracy",
            "validation_accuracy",
            "test_accuracy",
        ]
    )


    return df

##########################################################################################
def plot_epoch(
        df: pd.DataFrame,
        graph_name: str,
        model_mode: str,
        root_dir: str,
        col: str = "loss",
        keep: bool = True,
        show: bool = True,
        image_type="png",
):

    if col == "loss":
        cols = ["training_loss", "validation_loss", "test_loss"]
        train_opt = df["training_loss"].min()
        train_pos = df["epoch_number"][df["training_loss"] == train_opt].values[0]
        val_opt = df["validation_loss"].min()
        val_pos = df["epoch_number"][df["validation_loss"] == val_opt].values[0]
        test_opt = df["test_loss"].min()
        test_pos = df["epoch_number"][df["test_loss"] == test_opt].values[0]

    elif col == "accuracy":
        cols = ["training_accuracy", "validation_accuracy", "test_accuracy"]
        train_opt = df["training_accuracy"].max()
        train_pos = df["epoch_number"][df["training_accuracy"] == train_opt].values[0]
        val_opt = df["validation_accuracy"].max()
        val_pos = df["epoch_number"][df["validation_accuracy"] == val_opt].values[0]
        test_opt = df["test_accuracy"].max()
        test_pos = df["epoch_number"][df["test_accuracy"] == test_opt].values[0]

    else:
        raise ValueError('col in plot_epoch is not defined...')

    plt.figure(figsize=(15, 12)) #width height
    plt.plot(df["epoch_number"], df[cols[0]], color='b', label='training set')
    plt.plot(df["epoch_number"], df[cols[1]], color='g', label='validation set')
    plt.plot(df["epoch_number"], df[cols[2]], color='r', label='test set')

    def annotate_best(label, position, value, color):
        plt.annotate(
            f'{label}: {value:.2f}',
            xy=(position, value),
            xytext=(position, value + (0.1 if col == "loss" else -0.1) * value),
            arrowprops=dict(arrowstyle="->", color=color),
            fontsize=9,
            ha='center',
            color=color
        )

    annotate_best('train', train_pos, train_opt, 'b', )
    annotate_best('val', val_pos, val_opt, 'g')
    annotate_best('test', test_pos, test_opt, 'r')

    plt.title(
        f"{graph_name} - {model_mode} {col} \n "
        f"best train: ep#: {train_pos}, value: {np.round(train_opt, 2)} \n  "
        f"best val: ep#: {val_pos},     value: {np.round(val_opt, 3)} \n"
        f"best test: ep#: {test_pos},   value: {np.round(test_opt, 3)} "
    )
    plt.xlabel("epoch")
    plt.ylabel(col)
    plt.legend()

    if keep:
        check_directory(f"{root_dir}Result\\{graph_name}\\")
        cap = f"{root_dir}Result\\{graph_name}\\{graph_name}_{model_mode}_{col}.{image_type}"
        plt.savefig(cap)
    if show:
        plt.show()
    plt.close()
