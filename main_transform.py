"""


# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

"""
globals().clear()

######################################################################################################################## libraries
import gc
import json

import torch
from classes import DynamicGCNconv
from graph_data import GraphData
from train_test import train, test
from transform import (
    standard_normal,
    standardization,
    krylov_reEmbed,
    restricted_generalized_eigenvectors,
    divide_by_col_norm
)
from utils import gpu_setup, plot_epoch, epoch_info, check_directory

######################################################################################################################## params
with open('params.json') as f:
    params = json.load(f)

######################################################################################################################## setup

torch.cuda.empty_cache()
gc.collect()
device = gpu_setup(params['use_gpu'])

if device.type == 'cuda':
    torch.cuda.manual_seed(123)
    torch.cuda.manual_seed_all(123)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


######################################################################################################################## graphs
torch.manual_seed(222)

# graph_names = ["Cora", "CiteSeer", "PubMed", "WikiCs", "ogbn-arxiv", "ogbn-products"]
graph_names = [
    "Cora",
    "CiteSeer",
    "PubMed",
    "ogbn-arxiv"
]

############################################################
for graph_name in graph_names:

    ############ graph
    graph_data = GraphData(
        graph_name=graph_name,
        path=params['path'],
        graph_data=params['graph_data'],
    )
    graph = graph_data.get_graph()
    graph = graph.to(device)
    graph2 = graph.__copy__()

    for use_transform in [
        # 'original',
        # 'original_standard',
        # 'original_norm2',
        # 'normal',
        # 'normal_standard',
        'generalized_eigenvectors',
        'row_normalized_generalized_eigenvectors'
    ]:
        print("###############################################")
        print(f'{graph_name} {use_transform}: staring...')

        graph = graph2.__copy__()

        if use_transform == 'original':
            model_mode = 'original'

        elif use_transform == 'original_standard':
            graph = standardization(graph=graph)
            model_mode = 'original_standard'

        elif use_transform == 'original_norm2':
            graph = divide_by_col_norm(graph=graph, device=device)
            model_mode = 'original_norm2'

        elif use_transform == 'normal':
            graph = standard_normal(graph=graph, device=device)
            model_mode = 'normal'

        elif use_transform == 'normal_standard':
            graph = standard_normal(graph=graph, device=device)
            graph = standardization(graph=graph)
            model_mode = 'standardized_normal'

        elif use_transform == 'generalized_eigenvectors':
            graph = standardization(graph)
            graph = restricted_generalized_eigenvectors(
                graph=graph,
                device=device,
                normalize=False
            )
            model_mode = 'generalized_eigenvectors'

        elif use_transform == 'row_normalized_generalized_eigenvectors':
            graph = standardization(graph)
            graph = restricted_generalized_eigenvectors(
                graph=graph,
                device=device,
                normalize=True
            )
            model_mode = 'row_normalized_generalized_eigenvectors'

        else:
            raise ValueError(
                (
                    f"use_transform not in "
                    f"['original', 'normal', 'normal_standard', 'generalized_eigenvectors', "
                    f"'ranked_generalized_eigenvectors']"
                )
            )

        ######################################################
        ######################################################
        ############## Original Model ########################
        ######################################################
        ######################################################

        model = DynamicGCNconv(
            in_dim=graph.num_node_features,
            hidden_dims=tuple(params['model_params']['hidden_dim']),
            out_dim=graph.num_classes,
            transform_mode=model_mode,
            mutate_x_epoch=False,
            add_relu=params['model_params']['add_relu'],
            bias=params['model_params']['bias'],
            init=params['model_params']['init'],
            normalize=params['model_params']['normalize'],
            add_self_loops=params['model_params']['add_self_loops'],
            dropout=params['model_params']['dropout']
        ).to(device)
        print(f'model: {model.name}')

        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['hyperparam_param']['learning_rate'],
            weight_decay=params['hyperparam_param']['weight_decay'],
        )

        ############ training
        print("entering training phase...")
        model, optimizer, local_epoch_df = train(
            model=model,
            optimizer=optimizer,
            num_epoch=params['hyperparam_param']['number_epochs'],
            graph=graph,
            verbose=True
        )

        plot_epoch(
            df=local_epoch_df,
            graph_name=graph.name,
            model_mode=model_mode,
            root_dir=params['path'],
            col="loss",
            keep=True,
            show=False,
            image_type='png'
        )
        plot_epoch(
            df=local_epoch_df,
            graph_name=graph.name,
            model_mode=model_mode,
            root_dir=params['path'],
            col="accuracy",
            keep=True,
            show=False,
            image_type='png'
        )

        ############ test
        print("entering test phase...")

        print(f'model: {model.name}')
        model, local_split_df = test(
            model=model,
            graph=graph,
            verbose=True)

        saved = {
            'model': model,
            'optimizer': optimizer,
            'epoch_df': local_epoch_df,
            'spit_df': local_split_df
        }

        cap = f'{params["path"]}Result\\{graph.name}\\'
        check_directory(cap)
        torch.save(saved, f'{cap}{graph.name}_GCNconv_{model_mode}.pt')

        print(f'{graph_name} {use_transform}: Done!')
        print("###############################################")
        print("\n\n")
