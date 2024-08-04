"""
    This class represent the graph object in torch geometric


"""
import os
import copy
from collections import Counter
import math

import torch

from torch_geometric.data.data import Data
from torch_geometric.datasets import WikiCS
from torch_geometric.datasets import Planetoid
from ogb.nodeproppred import PygNodePropPredDataset

from torch_geometric.utils import remove_self_loops
from torch_geometric.utils import to_undirected
from torch_geometric.utils import index_to_mask
from torch_geometric.utils import to_scipy_sparse_matrix
from torch_geometric.utils import from_scipy_sparse_matrix

import numpy as np
from scipy.sparse.csgraph import connected_components
from utils import check_directory


class GraphData:

    def __init__(
            self,
            graph_name: str,
            path: str,
            graph_data: dict,
    ):

        train_val = graph_data['train_val']
        final_split_method = graph_data['final_split_method']
        remove_self_loop = graph_data['remove_self_loop']
        directed_to_undirected = graph_data['directed_to_undirected']
        reduce_to_largest_cc = graph_data['reduce_to_largest_cc']
        sel_dim_in_multiple_trains = graph_data['sel_dim_in_multiple_trains']
        verbose = graph_data['verbose']

        if path is None:
            path = os.getcwd()

        self.path = f'{path}'
        self.remove_self_loop = remove_self_loop
        self.directed_to_undirected = directed_to_undirected
        self.reduce_to_largest_cc = reduce_to_largest_cc
        self.graph_name = graph_name
        self.sel_dim_in_multiple_trains = sel_dim_in_multiple_trains
        self.train_ratio = train_val['train_ratio'] if train_val['train_ratio'] is not None else None
        self.val_ratio = train_val['val_ratio'] if train_val['val_ratio'] is not None else None
        self.train_num = train_val['train_num'] if train_val['train_num'] is not None else None
        self.val_num = train_val['val_num'] if train_val['val_num'] is not None else None
        self.final_split_method = final_split_method
        self.verbose = verbose

        print(f'graph: {graph_name}')
        
        graph = self.graph_initialization()
        graph = self.graph_preprocessing(graph=graph)
        graph = self.add_split_type(graph=graph)
        graph.path = path
        graph.name = self.graph_name
        self.summary(graph)
        self.graph = graph
        print('\n')

    ##########################################################################################
    def __copy__(self):
        return copy.deepcopy(self)

    ##########################################################################################
    def graph_initialization(self) -> Data:
        """

        :return:
        """
        check_directory(f'{self.path}DataSets\\')
        path = f'{self.path}DataSets\\'
        if self.graph_name in ['Cora', 'CiteSeer', 'PubMed']:
            graphs = Planetoid(name=self.graph_name, root=path)
            graph = graphs[0]

        if self.graph_name == "WikiCs":
            graphs = WikiCS(root=path)
            graph = graphs[0]

        if self.graph_name in ["ogbn-arxiv", "ogbn-products"]:
            graphs = PygNodePropPredDataset(name=self.graph_name, root=path)
            graph = self._ogb_create_mask(graph=graphs[0], idx_split=graphs.get_idx_split())

        else:
            ValueError(f'graph_name={self.graph_name} is not defined')

        graph = self._add_attribute(graph=graph)
        graph.name = self.graph_name

        return graph

    ##########################################################################################
    @staticmethod
    def _ogb_create_mask(graph: Data, idx_split) -> Data:
        """

        :param graph:
        :param split_idx:
        :return:
        """

        train_mask, val_mask, test_mask = idx_split["train"], idx_split["valid"], idx_split["test"]
        mask_orig = torch.arange(start=0, end=graph.x.shape[0])
        graph.train_mask = torch.isin(mask_orig, train_mask)
        graph.val_mask = torch.isin(mask_orig, val_mask)
        graph.test_mask = torch.isin(mask_orig, test_mask)

        return graph

    ##########################################################################################
    @staticmethod
    def _add_attribute(graph: Data) -> Data:
        """

        :param graph:
        :return:
        """
        if not hasattr(graph, "num_nodes"):
            graph.num_nodes = graph.x.shape[0]

        if not hasattr(graph, "num_edges"):
            graph.num_edges = graph.edge_index.shape[1]

        if not hasattr(graph, "num_classes"):
            graph.classes = torch.unique(graph.y)
            graph.num_classes = len(graph.classes)

        if graph.y.dim() > 1:
            graph.y = torch.squeeze(graph.y)

        return graph

    ##########################################################################################
    @staticmethod
    def _update_attribute(graph: Data) -> Data:
        """

        :param graph:
        :return:
        """
        graph.num_nodes = graph.x.shape[0]
        graph.num_edges = graph.edge_index.shape[1]
        graph.classes = torch.unique(graph.y)
        graph.num_classes = len(graph.classes)

        if graph.y.dim() > 1:
            graph.y = torch.squeeze(graph.y)

        return graph

    ##########################################################################################
    @staticmethod
    def summary(graph: Data) -> None:

        if hasattr(graph, 'final_train_mask'):
            n = graph.final_train_mask.size(1) if graph.final_train_mask.dim()>1 else 1
        else:
            n = 0

        print((
            f"graph {graph.name} \n"
            f"# nodes: {graph.num_nodes} \n"
            f"# edges: {graph.num_edges} \n"
            f"edge_attr {graph.edge_attr if hasattr(graph, 'edge_attr') else 'not provided'} \n"
            f"# classes: {graph.num_classes} \n"
            f"classes: {graph.classes} \n"
            f"has_isolated_nodes: {graph.has_isolated_nodes()} \n"
            f"is_directed {graph.is_directed()} \n"
            f"is_undirected {graph.is_undirected()} \n"
            f"is_coalesced {graph.is_coalesced()} \n"
            f"# different train-val-test split: {n}"

        ))

    ##########################################################################################
    def graph_preprocessing(self, graph: Data) -> Data:
        
        if self.verbose:
            print(f'graph cleaning on {graph.name}')
        x, edge_index, edge_attr, num_nodes = graph.x, graph.edge_index, graph.edge_attr, graph.num_nodes
    
        if self.remove_self_loop and graph.has_self_loops():
            if self.verbose:
                print("removing self loops...")
            edge_index, edge_attr = remove_self_loops(edge_index=edge_index, edge_attr=edge_attr)
    
        if self.directed_to_undirected and graph.is_directed():
            if self.verbose:
                print("converting to undirected ")
            edge_index = to_undirected(edge_index=edge_index)

        if self.reduce_to_largest_cc:
            if self.verbose:
                print('finding largest connected component')
            graph = self._largest_cc(
                graph=graph,
                num_nodes=num_nodes,
                edge_index=edge_index,
                edge_attr=edge_attr,
            )

        return graph

    ##########################################################################################
    def add_split_type(self, graph: Data) -> Data:

        graph = self.per_class_split(graph=graph)
        graph = self.all_class_split(graph=graph)

        if self.final_split_method == 'original':
            graph.final_train_mask = graph.train_mask
            graph.final_val_mask = graph.val_mask
            graph.final_test_mask = graph.test_mask

        elif self.final_split_method == 'per_class':
            graph.final_train_mask = graph.train_mask_per_class
            graph.final_val_mask = graph.val_mask_per_class
            graph.final_test_mask = graph.test_mask_per_class

        elif self.final_split_method == 'all_classes':
            graph.final_train_mask = graph.train_mask_all_classes
            graph.final_val_mask = graph.val_mask_all_classes
            graph.final_test_mask = graph.test_mask_all_classes

        else:
            raise ValueError(f'self.final_split_method not in ["original", "per_class", "all_classes"]')

        return graph

    ##########################################################################################
    def _largest_cc(
            self,
            graph: Data,
            num_nodes: int,
            edge_index: torch.tensor,
            edge_attr,
    ) -> Data:

        sel_dim_in_multiple_trains = self.sel_dim_in_multiple_trains
        adj = to_scipy_sparse_matrix(edge_index=edge_index, num_nodes=num_nodes).tocsr()

        cc = connected_components(adj, directed=graph.is_directed())[1]
        cc = np.array(cc)
        freq = Counter(cc)
        cc_label = max(freq, key=freq.get)
        nodes = np.where(cc == cc_label)[0]
        adj = adj[:, nodes][nodes, :]

        edge_index, _ = from_scipy_sparse_matrix(adj)
        nodes = torch.tensor(nodes)
        index_2_mask = index_to_mask(nodes)

        graph.edge_attr = edge_attr
        graph.edge_index = edge_index
        graph.x = graph.x[nodes]
        graph.y = graph.y[nodes]
        graph.ind_to_mask = index_2_mask

        if graph.train_mask.dim() > 1 and sel_dim_in_multiple_trains is None:
            sel_dim_in_multiple_trains = torch.randint(0, graph.train_mask.shape[1], (1,))

        if graph.train_mask.dim() == 1:
            graph.train_mask = graph.train_mask[nodes]
            graph.val_mask = graph.val_mask[nodes]

        else:
            graph.train_multiple_masks = graph.train_mask[nodes, :]
            graph.train_mask = graph.train_mask[nodes, sel_dim_in_multiple_trains]

            graph.val_multiple_masks = graph.val_mask[nodes, :]
            graph.val_mask = graph.val_mask[nodes, sel_dim_in_multiple_trains]

        if graph.test_mask.dim() == 1:
            graph.test_mask = graph.test_mask[nodes]
        else:
            graph.test_multiple_masks = graph.test_mask[nodes, :]
            graph.test_mask = graph.test_mask[nodes, sel_dim_in_multiple_trains]

        graph = self._update_attribute(graph=graph)

        return graph

    ##########################################################################################
    def _in_single_class_split(
            self,
            graph: Data,
            label: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """

        :param graph: Data object
        :param labels:
        :param train_ratio: ratio for train set out of train val sets
        :param val_ratio: ratio for val set together
        :param train_num: number of datapoints for train set
        :param val_num: number of datapoints for validation set
        :return: graph with added split sets
        """

        label_inds = (graph.y == label).nonzero(as_tuple=True)[0]
        n = len(label_inds)
        suffle_inds = torch.randperm(n)
        label_inds = label_inds[suffle_inds]

        if self.train_ratio is not None:
            self.train_num = math.ceil(n * self.train_ratio)
            self.val_num = math.ceil(n * self.val_ratio)

        perm_train = label_inds[0:self.train_num]
        perm_val = label_inds[self.train_num:self.train_num + self.val_num]
        perm_test = label_inds[self.train_num + self.val_num:]

        return perm_train, perm_val, perm_test

    ##########################################################################################
    def per_class_split(
            self,
            graph: Data,
    ) -> Data:
        """

        :param graph:
        :param train_ratio:
        :param val_ratio:
        :param train_num:
        :param val_num:
        :return:
        """

        ################################################################
        train_all = torch.tensor([])
        val_all = torch.tensor([])
        test_all = torch.tensor([])

        for label in torch.unique(graph.y):
            perm_train, perm_val, perm_test = self._in_single_class_split(
                                               graph=graph,
                                               label=label,
            )
            train_all = torch.cat((train_all, perm_train))
            val_all = torch.cat((val_all, perm_val))
            test_all = torch.cat((test_all, perm_test))

        indices = torch.arange(0, graph.num_nodes)
        train_mask_per_class = torch.isin(indices, train_all)
        val_mask_per_class = torch.isin(indices, val_all)
        test_mask_per_class = torch.isin(indices, test_all)

        graph.train_mask_perClass = train_mask_per_class
        graph.val_mask_per_class = val_mask_per_class
        graph.test_mask_per_class = test_mask_per_class

        if self.verbose:
            print("train_mask_per_class size: ", train_mask_per_class.size())
            print("val_mask_per_class size: ", val_mask_per_class.size())
            print("test_mask_per_class size: ", test_mask_per_class.size())

        return graph

    ##########################################################################################
    def all_class_split(
            self,
            graph: Data,
    ) -> Data:
            """

            :param graph:
            :param train_ratio:
            :param val_ratio:
            :param train_num:
            :param val_num:
            :return:
            """

            if self.train_ratio is not None:
                self.train_num = int(graph.num_nodes * self.train_ratio)
                self.val_num = int(graph.num_nodes * self.val_ratio)

            perm = torch.randperm(graph.num_nodes)
            perm_train = perm[0:self.train_num]
            perm_val = perm[self.train_num:self.train_num + self.val_num]
            perm_test = perm[self.train_num + self.val_num:]

            indices = torch.arange(0, graph.num_nodes)
            train_mask_all_classes = torch.isin(indices, perm_train)
            val_mask_all_classes = torch.isin(indices, perm_val)
            test_mask_all_classes = torch.isin(indices, perm_test)

            graph.train_mask_all_classes = train_mask_all_classes
            graph.val_mask_all_classes = val_mask_all_classes
            graph.test_mask_all_classes = test_mask_all_classes

            if self.verbose:
                print("train_mask_all_classes size: ", train_mask_all_classes.size())
                print("val_mask_all_classes size: ", val_mask_all_classes.size())
                print("test_mask_all_classes size: ", test_mask_all_classes.size())

            return graph

    ##########################################################################################
    def get_graph(self):
        """
        returns the graph attribute of the object
        :return:
        """
        return self.graph
