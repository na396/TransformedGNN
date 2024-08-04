import time
from utils import *
import torch.nn.functional as F


#############################################################
def train(
        model,
        optimizer, 
        num_epoch, 
        graph,
        verbose=True
):

    # perform training phase on the input data
    # mdl: model
    # opt: optimizer
    # num_epoch: number of epoch
    # data: input semi-supervised graph

    t = time.time()
    epoch_df = epoch_info()

    model.train()
    for epoch in range(num_epoch):

        optimizer.zero_grad()
        out = model(graph)

        loss_train = F.nll_loss(out[graph.final_train_mask], graph.y[graph.final_train_mask])
        acc_train = accuracy(out[graph.final_train_mask], graph.y[graph.final_train_mask])

        loss_val = F.nll_loss(out[graph.final_val_mask], graph.y[graph.final_val_mask])
        acc_val = accuracy(out[graph.final_val_mask], graph.y[graph.final_val_mask])

        loss_test = F.nll_loss(out[graph.final_test_mask], graph.y[graph.final_test_mask])
        acc_test = accuracy(out[graph.final_test_mask], graph.y[graph.final_test_mask])

        row = {
            "model": model.name,
            "graph": graph.name,
            "epoch_number": epoch+1,
            "epoch_time":time.time() - t,
            "training_loss": loss_train.cpu().detach().numpy(),
            "training_accuracy": acc_train,
            "validation_loss":loss_val.cpu().detach().numpy(),
            "validation_accuracy": acc_val,
            "test_loss":loss_test.cpu().detach().numpy(),
            "test_accuracy":acc_test,
        }

        epoch_df = append_df(df=epoch_df, row=row)

        if verbose:
           print(epoch_df.loc[:,
                 ["epoch_number",
                  "epoch_time",
                  "training_loss",
                  "validation_loss",
                  "test_loss",
                  "training_accuracy",
                  "validation_accuracy",
                  "test_accuracy"
                  ]
                 ].iloc[-1, :]
        )
           print("\n")

        loss_train.backward()
        optimizer.step()

    return model, optimizer, epoch_df


#############################################################
def test(model, graph, verbose=True):

    split_df = epoch_info()

    model.eval()
    out = model(graph)

    pred = model(graph).argmax(dim=1)
    acc_model = (pred == graph.y).sum()/len(graph.y)

    loss_train = F.nll_loss(out[graph.final_train_mask], graph.y[graph.final_train_mask])
    acc_train = accuracy(out[graph.final_train_mask], graph.y[graph.final_train_mask])

    loss_val = F.nll_loss(out[graph.final_val_mask], graph.y[graph.final_val_mask])
    acc_val = accuracy(out[graph.final_val_mask], graph.y[graph.final_val_mask])

    loss_test = F.nll_loss(out[graph.final_test_mask], graph.y[graph.final_test_mask])
    acc_test = accuracy(out[graph.final_test_mask], graph.y[graph.final_test_mask])

    row = {
        "model": model.name,
        "graph": graph.name,
        "model_accuracy": acc_model.item(),
        "training_loss": loss_train.cpu().detach().numpy(),
        "training_accuracy": acc_train,
        "validation_loss": loss_val.cpu().detach().numpy(),
        "validation_accuracy": acc_val,
        "test_loss": loss_test.cpu().detach().numpy(),
        "test_accuracy": acc_test
    }
    split_df = append_df(df=split_df, row=row)

    if verbose:
        print(split_df.iloc[-1, :])
        print("\n")

    return model, split_df

#############################################################
def train_earlystoping(
        model,
        optimizer,
        num_epoch,
        graph,
        verbose=True
):

    # perform training phase on the input data
    # mdl: model
    # opt: optimizer
    # num_epoch: number of epoch
    # data: input semi-supervised graph

    t = time.time()
    epoch_df = epoch_info()

    ploss_val = float('inf')

    model.train()
    for epoch in range(num_epoch):

        optimizer.zero_grad()
        out = model(graph)

        loss_train = F.nll_loss(out[graph.final_train_mask], graph.y[graph.final_train_mask])
        acc_train = accuracy(out[graph.final_train_mask], graph.y[graph.final_train_mask])

        loss_val = F.nll_loss(out[graph.final_val_mask], graph.y[graph.final_val_mask])
        acc_val = accuracy(out[graph.final_val_mask], graph.y[graph.final_val_mask])

        loss_test = F.nll_loss(out[graph.final_test_mask], graph.y[graph.final_test_mask])
        acc_test = accuracy(out[graph.final_test_mask], graph.y[graph.final_test_mask])

        if ploss_val < loss_val:
            print(f'best validation loss at epoch: {epoch+1}')
            print(f'difference in loss {ploss_val - loss_val}')
            break

        ploss_val = loss_val

        row = {
            "model": model.name,
            "graph": graph.name,
            "epoch_number": epoch+1,
            "epoch_time":time.time() - t,
            "training_loss": loss_train.cpu().detach().numpy(),
            "training_accuracy": acc_train,
            "validation_loss":loss_val.cpu().detach().numpy(),
            "validation_accuracy": acc_val,
            "test_loss":loss_test.cpu().detach().numpy(),
            "test_accuracy":acc_test,
        }

        epoch_df = append_df(df=epoch_df, row=row)

        if verbose:
           print(epoch_df.loc[:,
                 ["epoch_number",
                  "epoch_time",
                  "training_loss",
                  "validation_loss",
                  "test_loss",
                  "training_accuracy",
                  "validation_accuracy",
                  "test_accuracy"
                  ]
                 ].iloc[-1, :]
        )
           print("\n")

        loss_train.backward()
        optimizer.step()

    return model, optimizer, epoch_df


