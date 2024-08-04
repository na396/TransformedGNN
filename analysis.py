from analysis_utils import *

cora = read_df('Cora')
citeseer = read_df('CiteSeer')
pubmed = read_df('PubMed')
wikics = read_df('WikiCs')
arxiv = read_df('ogbn-arxiv')

cora_k = every_k_row_df(graph_name='Cora', cols=cols, k=10)
citeseer_k = every_k_row_df(graph_name='CiteSeer', cols=cols, k=10)
pubmed_k = every_k_row_df(graph_name='PubMed', cols=cols, k=10)
wikics_k = every_k_row_df(graph_name='WikiCs', cols=cols, k=10)
arxiv_k = every_k_row_df(graph_name='ogbn-arxiv', cols=cols, k=10)

###################### cora
plt_df(
    graph_name='Cora',
    df=cora,
    cols=cols,
    eval_col='validation_loss',
    epoch_col='epoch_number',
    keep=True,
    show=False,
)


plt_df(
    graph_name='Cora',
    df=cora_k,
    cols=plt_cols,
    eval_col='mean_validation_loss',
    epoch_col='k_epoch',
    keep=True,
    show=False,
)

############### citeseer
plt_df(
    graph_name='CiteSeer',
    df=citeseer,
    cols=cols,
    eval_col='validation_loss',
    epoch_col='epoch_number',
    keep=True,
    show=False,
)


plt_df(
    graph_name='CiteSeer',
    df=citeseer_k,
    cols=plt_cols,
    eval_col='mean_validation_loss',
    epoch_col='k_epoch',
    keep=True,
    show=False,
)

############### pubmed
plt_df(
    graph_name='PubMed',
    df=pubmed,
    cols=cols,
    eval_col='validation_loss',
    epoch_col='epoch_number',
    keep=True,
    show=False,
)


plt_df(
    graph_name='PubMed',
    df=pubmed_k,
    cols=plt_cols,
    eval_col='mean_validation_loss',
    epoch_col='k_epoch',
    keep=True,
    show=False,
)


############### wikics
plt_df(
    graph_name='WikiCs',
    df=wikics,
    cols=cols,
    eval_col='validation_loss',
    epoch_col='epoch_number',
    keep=True,
    show=False,
)


plt_df(
    graph_name='WikiCs',
    df=wikics_k,
    cols=plt_cols,
    eval_col='mean_validation_loss',
    epoch_col='k_epoch',
    keep=True,
    show=False,
)


############### arxiv
plt_df(
    graph_name='ogbn-arxiv',
    df=arxiv,
    cols=cols,
    eval_col='validation_loss',
    epoch_col='epoch_number',
    keep=True,
    show=False,
)


plt_df(
    graph_name='ogbn-arxiv',
    df=arxiv_k,
    cols=plt_cols,
    eval_col='mean_validation_loss',
    epoch_col='k_epoch',
    keep=True,
    show=False,
)