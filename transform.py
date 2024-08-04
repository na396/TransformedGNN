""" Main Module for Transformation """
from typing import Optional, Any
import time
import warnings

import numpy as np
from numpy.linalg import matrix_rank

from scipy.linalg import eig, eigh, qr
from scipy.sparse import eye, diags, spdiags
from scipy.sparse.linalg import eigsh

import torch
from torch.optim import SparseAdam
from torch.utils.data import DataLoader

from torch_sparse import SparseTensor, sum as sparsesum, mul
from torch_scatter import scatter_add

from torch_geometric.data.data import Data
from torch_geometric.utils import (
    degree,
    to_scipy_sparse_matrix,
    add_remaining_self_loops,
    from_scipy_sparse_matrix
)

# import dgl
# from dgl.nn.pytorch import DeepWalk

from utils import check_directory

############################
def spectral_embedding(
        graph: Data,
        mode: str,
        num_sel_first_col: int = None,
        drop_first_col: bool = True,
        load_cache: bool = True,
        save_cache: bool = True,
        vectorized: bool = True,
) -> Data:
    """

    :param vectorized:
    :param save_cache:
    :param graph:
    :param num_sel_first_col:
    :param drop_first_col:
    :param load_cache:
    :param mode:
    :return:
    """
    t = time.time()
    if mode not in ['symmetric', 'non-symmetric']:
        raise ValueError(f'{mode} not in ["symmetric", "non-symmetric"]')

    print(f'{mode} spectral embedding')

    check_directory(f'{graph.path}\\Cache\\{graph.name}')
    cache_eigval = f'{graph.path}\\Cache\\{graph.name}\\{mode}_eigenvalues_embedding.pt'
    cache_eigvec = f'{graph.path}\\Cache\\{graph.name}\\{mode}_eigenvectors_embedding.pt'

    k = graph.num_nodes - 1 if num_sel_first_col is None else num_sel_first_col + 1

    x, edge_index = graph.x, graph.edge_index
    deg = degree(edge_index[0])
    A = to_scipy_sparse_matrix(edge_index=edge_index).tocsr()
    D = spdiags(1 / deg.sqrt(), 0, graph.num_nodes, graph.num_nodes)

    DA = D.dot(A)
    L = DA.dot(D)

    if load_cache:
        try:
            X = torch.load(cache_eigval, map_location=torch.device('cpu'))
            Y = torch.load(cache_eigvec, map_location=torch.device('cpu'))
        except:
            print(f'spectral embedding for {graph.name} was not stored')
            X, Y = eigsh(A=L, k=k, which='LM')

    else:
        X, Y = eigsh(A=L, k=k, which='LM')

    if save_cache:
        torch.save(X, cache_eigval)
        torch.save(Y, cache_eigvec)

    X = torch.tensor(X)
    Y = torch.tensor(Y)

    Xs = X.sort(descending=True)
    X = Xs.values
    Y = Y[:, Xs.indices]

    if (torch.round(X, decimals=4) == 1).sum() != 1:
        raise ValueError('eigenvalues==1 is more than one')

    if mode == "symmetric":  # row_wise multiplication each row of Y with 1/sqrt(deg)
        if drop_first_col:
            Y = Y[:, 1:Y.size(1)]
            X = X[1:]
        graph.embedding_vectors = Y
        graph.embedding_values = X
        print(f'total time: {time.time() - t}')

        return graph

    D = torch.sparse.spdiags(1 / deg.sqrt(), torch.tensor([0]), (graph.num_nodes, graph.num_nodes))
    Y = torch.matmul(D, Y)
    Y = torch.sub(Y, torch.matmul(deg, Y) / deg.sum())

    if vectorized:
        Ydeg = Y * deg.view(-1, 1)  # elementwise multiplication vector column of Y with vector deg
        denum = torch.sqrt(
            torch.sum(Ydeg * Y, dim=0))  # dot product of each vector column of Ydeg with vector column of Y
        Y = torch.div(Y, denum)  # elementwise division of each vector column of Y with corresponding scalar of denum
    else:
        for j in range(Y.size(1)):
            x = Y[:, j]
            Y[:, j] = x / torch.sqrt(torch.matmul(torch.mul(x, deg), x))

    if drop_first_col:
        Y = Y[:, 1:Y.size(1)]
        X = X[1:]

    constant = torch.matmul(torch.mul(Y[:, 0], deg), Y[:, 0])
    if torch.round(constant, decimals=5) != 1:
        warnings.warn("constant condition does not hold! ")

    graph.embedding_vectors = Y
    graph.embedding_values = X

    print(f'total time: {time.time() - t}')
    return graph


############################
# def deepwalk_embedding(
#         graph: Data,
#         embedding_dim: int = 128,
#         learning_rate: float = 0.01,
#         n_epoch: int = 100,
#         batch_size: int = 128,
#         load_cache: bool = True,
#         save_cache: bool = True,
# ) -> Data:
#     """
#
#     :param save_cache:
#     :param load_cache:
#     :param batch_size:
#     :param learning_rate:
#     :param embedding_dim:
#     :type n_epoch: object
#     :type graph: object
#     """
#     print(f"deepwalk embedding")
#
#     check_directory(f'{graph.path}\\Cache\\{graph.name}')
#     cache_dwalk = f'{graph.path}\\Cache\\{graph.name}\\deepwalk_embedding.pt'
#
#     t = time.time()
#
#     if load_cache:
#         try:
#             dw = torch.load(cache_dwalk, map_location=torch.device('cpu'))
#             return dw
#         except:
#             print(f'deepwalk embedding for {graph.name} was not stored')
#
#     g = to_dgl(graph=graph)
#     mdl = DeepWalk(g=g, emb_dim=embedding_dim)
#     dataloader = DataLoader(
#         torch.arange(g.num_nodes()),
#         batch_size=batch_size,
#         shuffle=True,
#         collate_fn=mdl.sample
#     )
#     opt = SparseAdam(mdl.parameters(), lr=learning_rate)
#
#     for epoch in range(n_epoch):
#         for batch_walk in dataloader:
#             loss = mdl(batch_walk)
#             opt.zero_grad()
#             loss.backward()
#             opt.step()
#
#     if save_cache:
#         torch.save(mdl.node_embed.weight.detach(), cache_dwalk)
#
#     graph.embedding_vecotrs = mdl.node_embed.weight.detach()
#     graph.embedding_values = None
#
#     print(f"total time: {time.time() - t}")
#     return graph
#
#
# ############################
# def to_dgl(graph: Data):
#     """
#     Converts Data Object to dgl object
#     :param graph:
#     :return:
#     """
#
#     row, col = graph.edge_index
#     g = dgl.graph((row, col))
#
#     assert graph.train_mask_final.shape[0] == g.num_nodes(), "Size mismatch for train_mask_final"
#     assert graph.val_mask_final.shape[0] == g.num_nodes(), "Size mismatch for val_mask_final"
#     assert graph.test_mask_final.shape[0] == g.num_nodes(), "Size mismatch for test_mask_final"
#
#     g.ndata["train_mask"] = graph.train_mask
#     g.ndata["label"] = graph.y
#     g.ndata["val_mask"] = graph.val_mask
#     g.ndata["test_mask"] = graph.test_mask
#     g.ndata['x'] = graph.x
#
#     if hasattr(graph, "edge_attr") and graph.edge_attr != None:
#         g.edata = graph.edge_attr
#
#     return g


############################
# def embedding_method(
#         graph: Data,
#         method: str,
#         spectral_params: Optional[dict[Any, Any]] = None,
#         deepwalk_params: Optional[dict[Any, Any]] = None,
# ) -> Data:
#     if method in ['symmetric', 'non-symmetric']:
#         graph = spectral_embedding(
#             graph=graph,
#             mode=method,
#             num_sel_first_col=spectral_params['num_sel_first_col'],
#             drop_first_col=spectral_params['drop_first_col'],
#             load_cache=spectral_params['load_cache'],
#             vectorized=spectral_params['vectorized'],
#         )
#
#     elif method == 'deepwalk':
#         graph = deepwalk_embedding(
#             graph=graph,
#             embedding_dim=deepwalk_params['embedding_dim'],
#             learning_rate=deepwalk_params['learning_rate'],
#             n_epoch=deepwalk_params['n_epoch'],
#             batch_size=deepwalk_params['batch_size'],
#             load_cache=deepwalk_params['load_cache'],
#         )
#
#     return graph


############################

############################

############################
def edge_index_normalization(
         graph: Data,
         add_self_loops: bool = True,
         fill_value: float = 1.0,
         dtype: torch.dtype = torch.float32,
):
    """

    :param graph:
    :param add_self_loops:
    :param fill_value:
    :param dtype:
    :return:
    """
    edge_index, num_nodes = graph.edge_index, graph.num_nodes
    edge_weight = graph.edge_weight if hasattr(graph, "edge_attr") else None

    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype, device=edge_index.device)

    if add_self_loops:
        edge_index, tmp_edge_weight = add_remaining_self_loops(
         edge_index=edge_index,
         edge_attr=edge_weight,
         fill_value=fill_value,
         num_nodes=num_nodes
        )
        assert tmp_edge_weight is not None
        edge_weight = tmp_edge_weight

    row, col = edge_index[0], edge_index[1]
    deg = scatter_add(input=edge_weight, dim=col, index=0, src=num_nodes)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)

    return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]


############################
def sparse_tensor_normalizing(
        edge_index: SparseTensor,
        add_self_loops: bool = True,
        fill_value: float = 1.0,
        dtype: torch.dtype = torch.float32,

):
    """

    :param edge_index:
    :param add_self_loops:
    :param fill_value:
    :param dtype:
    :return:
    """

    adj_t = edge_index

    if not adj_t.has_value():
        adj_t = adj_t.fill_value(fill_value)

    if add_self_loops:
        adj_t = adj_t.fill_diag(fill_value, dtype=dtype)

    deg = sparsesum(adj_t, dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt.masked_fill_(deg_inv_sqrt == float('inf'), 0)
    adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
    adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))

    return adj_t


############################
def standard_normal(graph: Data, device) -> Data:
    """
    :param x:
    :return:
    """
    x = graph.x

    n1 = x.size()
    trans = torch.randn(size=(x.size(1), x.size(1)), device=device)
    x = x @ trans
    n2 = x.size()
    graph.x = x

    print(f'standard_normal: {n1} => {n2} ')
    check_zero_sum(x)
    check_zero_std(x)

    graph = update_attribute(graph=graph)
    return graph


############################
def standardization(graph: Data) -> Data:
    """

    :param graph:
    :return:
    """
    x = graph.x
    n1 = x.size()

    means = x.mean(dim=0, keepdim=True)
    stds = x.std(dim=0, keepdim=True)
    x = (x - means) / stds
    graph.x = x

    n2 = x.size()
    print(f'standardization: {n1} => {n2} ')
    check_isnan(x)

    graph = update_attribute(graph=graph)
    return graph


############################
def check_isnan(x):
    ans = torch.isnan(x).sum()
    print(f'number of na after standardization: {ans}')

############################
def check_zero_sum(x):
    ans = torch.sum(x, 0)
    inds = (ans == 0).nonzero()

    if inds.size(0) ==0:
        print(f'x has no zero sum columns')
    else:
        print(f'number of zero col: {inds.size(0)} \n zero col indices: {inds}')

############################
def check_zero_std(x):

    ans = torch.std(x, dim=0, keepdim=True)
    ans = torch.abs(ans)
    print(f'smallest std value: {ans.min()}')
    inds = (ans == 0).nonzero()
    if inds.size(0) ==0:
        print(f'x has no zero std columns')
    else:
        print(f'number of zero std: {inds.size(0)} \n zero stdc ol indices: {inds}')


############################
def krylov_reEmbed(graph: Data, device):
    # X: node features matrix
    # The embedding dimension is determined by X.shape[1]

    t1 = time.time()
    # Build Laplacians
    edge_index = graph.edge_index
    X = graph.x
    dt_x = X.dtype
    n1 = X.size()

    X = X.cpu().detach().numpy()
    X_rank = matrix_rank(X)
    print(f'X rank: {X_rank}')
    print(f'X dims: {X.shape}')

    G = to_scipy_sparse_matrix(edge_index=edge_index).tocsr()
    n = G.shape[0]
    d = G.sum(axis=0).A.flatten()

    # Symmetric normalized adjacency and Laplacian
    diagonal_matrix = diags(1 / np.sqrt(d), 0)
    snG = diagonal_matrix @ G @ diagonal_matrix
    snL = eye(n) - snG

    # Form small eigenvalue problem for snL using node features X
    M = np.linalg.pinv(X) @ snL @ X


    # Use eig instead of eigs
    D, V = eig(M)

    idx = D.argsort()[::-1]
    D = D[idx]
    V = V[:, idx]
    V = X.dot(V)  # All eigenvectors
    lambda_vals = np.diag(D)  # All eigenvalues

    dim = X.shape[1]  # Set dim based on the second dimension of X

    x = diags(np.sqrt(1 / d), 0) @ eye(n) @ V[:, :dim]
    lambda_vals = lambda_vals[0:dim]

    # Apply d-orthogonalization
    x = d_orthogonalization(x, d)

    x = torch.from_numpy(x).to(device)
    x = x.type(dt_x)
    graph.x = x
    lambda_vals = torch.from_numpy(lambda_vals).to(device)
    graph.lambda_vals = lambda_vals
    lambda_vals = torch.sort(torch.real(lambda_vals), descending=False)

    n2 = x.size()
    t2 = time.time()

    print(f'krylov_reEmbed: {n1} => {n2} ')
    print(f'krylov_reEmbed time: {t2 - t1}')
    print(f'smallest lambda_vals: {lambda_vals[0:3]}')

    graph = update_attribute(graph=graph)

    return graph


############################
def d_orthogonalization(x, d):

    drps = []

    n = x.shape[0]
    for i in range(x.shape[1]):
        c = -np.dot(d, x[:, i]) / np.dot(d, np.ones(n))
        x[:, i] = x[:, i] + c * np.ones(n)
        sq_val = x[:, i].transpose() @ Bfun(x[:, i], d)
        if sq_val <= 0:
            drps.append(i)
        x[:, i] = x[:, i] / np.sqrt(sq_val)

    x = np.delete(x, drps, 1)
    return x


############################
def update_attribute(graph: Data) -> Data:
    """

    :param graph:
    :return:
    """
    num_nodes1 = graph.num_nodes
    num_edges1 = graph.num_edges
    classes1 = graph.num_classes
    num_classes1 = graph.num_classes
    feature_nodedim1 = graph.x.shape[1]

    graph.num_nodes = graph.x.shape[0]
    graph.num_edges = graph.edge_index.shape[1]
    graph.classes = torch.unique(graph.y)
    graph.num_classes = len(graph.classes)

    num_nodes2 = graph.num_nodes
    num_edges2 = graph.num_edges
    classes2 = graph.num_classes
    num_classes2 = graph.num_classes
    feature_nodedim2 = graph.x.shape[1]

    print((
        f"update attribute:"
        f"num_nodes: {num_nodes1} => {num_nodes2} \n"
        f"num_edges: {num_edges1} => {num_edges2} \n"
        f"classes: {classes1} => {classes2} \n"
        f"num_classes: {num_classes1} => {num_classes2} \n"
        f"node_features:{feature_nodedim1} => {feature_nodedim2} \n"
    ))

    return graph


############################
def restricted_generalized_eigenvectors(
        graph: Data,
        device,
        normalize: bool = False
):

    t1 = time.time()
    X = graph.x
    dt_x = X.dtype
    n1 = X.size()
    rnk = matrix_rank(X)

    print(f'X rank: {rnk}')
    print(f'X dims: {n1}')

    # Ensure X is orthonormal
    X = X.cpu().detach().numpy()
    Q, R = qr(X, mode='economic')
    Q = Q[:, :rnk]

    G = to_scipy_sparse_matrix(edge_index=graph.edge_index).tocsr()
    n = G.shape[0]
    L = eye(n) - G
    d = G.sum(axis=1).A.flatten()

    # Project L and K onto the subspace spanned by X
    L_proj = Q.T @ L @ Q

    dTQ = d @ Q
    K_proj = (Q.T * d) @ Q - np.outer(dTQ, dTQ) / np.sum(d)

    # Solve the generalized eigenvalue problem for (L_proj, K_proj)
    lambda_v, x = eigh(L_proj, K_proj)

    # Transform eigenvectors back to the original space
    x = Q @ x
    x = d_standardization(x, d)
    if normalize:
        x = devide_by_norm(x)

    # Sort eigenvalues and eigenvectors in descending order
    idx = lambda_v.argsort()[::-1]
    lambda_v = lambda_v[idx]
    x = x[:, idx]

    x = torch.from_numpy(x).to(device)
    x = x.type(dt_x)
    lambda_v = torch.from_numpy(lambda_v).to(device)

    graph.x = x
    graph.lambda_v = lambda_v

    n2 = x.size()
    t2 = time.time()

    print(f'restricted_generalized_eigenvectors: {n1} => {n2} ')
    print(f'restricted_generalized_eigenvectors time: {t2 - t1}')
    print(f'smallest lambda_vals: {lambda_v[0:3]}')

    graph = update_attribute(graph=graph)

    return graph

############################
def devide_by_norm(x):
    ans = x.sum(axis=1)
    inds = np.where(ans==0)
    x[inds, :] = 1
    x = x / np.linalg.norm(x, ord=None, axis=1, keepdims=True)
    x[inds, :] = 0
    return x

############################
def Bfun(x, d):
    sd = d.sum()
    return d * x - d * ((d @ x) / sd)


############################
def d_standardization(x, d):
    n = x.shape[0]
    for i in range(x.shape[1]):
        c = -np.dot(d, x[:, i]) / np.dot(d, np.ones(n))
        x[:, i] = x[:, i] + c * np.ones(n)
        x[:, i] = x[:, i] / np.sqrt(x[:, i].transpose() @ Bfun(x[:, i], d))
    return x


############################
# def topological_overlap_matrix(graph: Data) -> Data:
#
#     adj = to_scipy_sparse_matrix(edge_index=graph.edge_index)
#     adj.setdiag(0)
#     deg_row = np.asarray(adj.sum(axis=1))
#     deg_col = np.asarray(adj.sum(axis=0))
#
#     mat_row = np.tile(deg_row, adj.shape[1])
#     mat_col = np.tile(deg_col, (adj.shape[0], 1))
#
#     deg_min = np.minimum(mat_row, mat_col)
#
#     numerator = np.add(matrix_power(adj, 2), adj)
#     denominator = np.add(deg_min, np.subtract(1, adj.toarray()))
#     tom = np.divide(numerator.toarray(), denominator)
#     np.fill_diagonal(tom, 0)
#
#     edge_index, _ = from_scipy_sparse_matrix(coo_array(tom))
#     graph.edge_index = edge_index
#
#     print((
#         f"graph {graph.name} \n"
#         f"# nodes: {graph.num_nodes} \n"
#         f"# edges: {graph.num_edges} \n"
#         f"edge_attr {graph.edge_attr if hasattr(graph, 'edge_attr') else 'not provided'} \n"
#         f"# classes: {graph.num_classes} \n"
#         f"classes: {graph.classes} \n"
#         f"has_isolated_nodes: {graph.has_isolated_nodes()} \n"
#         f"is_directed {graph.is_directed()} \n"
#         f"is_undirected {graph.is_undirected()} \n"
#         f"is_coalesced {graph.is_coalesced()} \n"
#     ))
#
#     return graph


############################
# def deep_overlap_matrix(graph: Data) -> Data:
#
#     adj = to_scipy_sparse_matrix(edge_index=graph.edge_index)
#     adj.setdiag(0)
#
#     deg_row = np.asarray(adj.sum(axis=1))
#     deg_col = np.asarray(adj.sum(axis=0))
#     mat_row = np.tile(deg_row, adj.shape[1])
#     mat_col = np.tile(deg_col, (adj.shape[0], 1))
#     deg_min = np.minimum(mat_row, mat_col)
#     del deg_row, deg_col, mat_row, mat_col
#
#     deg_row = np.power(np.asarray(adj.sum(axis=1)), 2)
#     deg_col = np.power(np.asarray(adj.sum(axis=0)), 2)
#     mat_row = np.tile(deg_row, adj.shape[1])
#     mat_col = np.tile(deg_col, (adj.shape[0], 1))
#     deg_min2 = np.minimum(mat_row, mat_col)
#
#     numerator = np.add(adj, matrix_power(adj, 2))
#     numerator = np.add(numerator, matrix_power(adj, 3))
#
#     denominator = np.add(deg_min2, deg_min)
#     denominator = np.add(denominator, np.subtract(1, adj.toarray()))
#     dom = np.divide(numerator.toarray(), denominator)
#     np.fill_diagonal(dom, 0)
#
#     edge_index, _ = from_scipy_sparse_matrix(coo_array(dom))
#     graph.edge_index = edge_index
#
#     print((
#         f"graph {graph.name} \n"
#         f"# nodes: {graph.num_nodes} \n"
#         f"# edges: {graph.num_edges} \n"
#         f"edge_attr {graph.edge_attr if hasattr(graph, 'edge_attr') else 'not provided'} \n"
#         f"# classes: {graph.num_classes} \n"
#         f"classes: {graph.classes} \n"
#         f"has_isolated_nodes: {graph.has_isolated_nodes()} \n"
#         f"is_directed {graph.is_directed()} \n"
#         f"is_undirected {graph.is_undirected()} \n"
#         f"is_coalesced {graph.is_coalesced()} \n"
#     ))
#
#     return graph

############################
