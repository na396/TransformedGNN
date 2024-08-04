import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import torch
from utils import check_directory

path= "Y:\\Root\\Study\\PhD - All\\Contributions\\Paper 5 - GNN - TrainableFeatures\\"
cols = [
    'training_loss',
    'validation_loss',
    'test_loss',
    'training_accuracy',
    'validation_accuracy',
    'test_accuracy'
]

plt_cols = [
    "mean_training_loss",
    "mean_validation_loss",
    "mean_test_loss",
    "mean_training_accuracy",
    "mean_validation_accuracy",
    "mean_test_accuracy",
]


def read_df(graph_name):
    original = torch.load(f'{path}Result\\{graph_name}\\{graph_name}_GCNconv_original.pt',
                          map_location=torch.device('cpu'))['epoch_df']
    original['model_name'] = f'{graph_name}_original'

    normal = torch.load(f'{path}Result\\{graph_name}\\{graph_name}_GCNconv_normal.pt',
                        map_location=torch.device('cpu'))['epoch_df']
    normal['model_name'] = f'{graph_name}_normal'

    stan_normal = torch.load(f'{path}Result\\{graph_name}\\{graph_name}_GCNconv_standardized_normal.pt',
                             map_location=torch.device('cpu'))['epoch_df']
    stan_normal['model_name'] = f'{graph_name}_standardized_normal'

    df = pd.concat([original, normal, stan_normal], ignore_index=True)
    df = df.convert_dtypes()
    return df


def every_k_row(df, k, cols):

    mean_col = [f'mean_{col}' for col in cols]
    mean_col.append('k_epoch')
    tdic = {col: float for col in mean_col}
    out = pd.DataFrame(columns=list(tdic.keys())).astype(tdic)

    num_groups = len(df) // k + (1 if len(df) % k != 0 else 0)

    for i in range(num_groups):
        start_idx = i * k
        end_idx = start_idx + k
        subset = df.iloc[start_idx:end_idx]
        temp = pd.DataFrame({f'mean_{col}': [subset[col].mean()] for col in cols})
        temp['k_epoch'] = i
        out = pd.concat([out, temp], ignore_index=True)

    return out


def every_k_row_df(graph_name, cols, k=10):
    df = read_df(graph_name)

    new_mean = [f'mean_{col}' for col in cols]
    models = df['model_name'].unique()
    tdic = {col: float for col in new_mean}
    schema = {'graph': str, 'model_name': str, 'k_epoch': int}
    kdf = pd.DataFrame(columns=['graph', 'model_name', 'k_epoch']+new_mean).astype(tdic|schema)

    for model in models:
        tdf = every_k_row(
            df=df[df['model_name'] == model],
            k=k,
            cols=cols
        )
        tdf['model_name'] = model
        tdf['graph'] = graph_name
        tdf = tdf[kdf.columns]

        kdf = pd.concat([kdf, tdf], ignore_index=True)

    return kdf


def best_epoch(df_model, eval_col='mean_validation_loss', epoch_col='k_epoch'):
    val_opt = df_model[eval_col].min()
    val_pos = df_model[epoch_col][df_model[eval_col] == val_opt].values[0]

    return val_opt, val_pos


def single_plot(df, col, epoch_col, bestepoch, graph_name, keep, show):

    mode = 'every10row' if epoch_col == 'k_epoch' else ' everyrow'

    models = df['model_name'].unique()
    plt.figure(figsize=(15, 12))  # width height
    plt.rcParams.update({'font.size': 25})

    for model in models:
        df_model = df[df['model_name'] == model]
        df_model = df_model[df_model[epoch_col] <= bestepoch[model]]

        val_pos = bestepoch[model]
        val_opt = df_model[col][df_model[epoch_col] == val_pos].values[0]

        plt.plot(df_model[epoch_col], df_model[col], label=f'{model}')

        plt.annotate(
            f'{val_opt:.2f}',
            xy=(val_pos, val_opt),
            xytext=(val_pos, val_opt + (0.1 if 'loss' in col else -0.1) ),
            arrowprops=dict(arrowstyle="->"),
            fontsize=20,
            ha='center',
        )

    plt.title(f"{graph_name} - {col} \n ", fontsize=40)
    plt.xlabel(epoch_col, fontsize=30)
    plt.ylabel(col, fontsize=30)
    plt.legend(fontsize=25)

    if keep:
        check_directory(f"{path}Result\\{graph_name}\\per_col\\")
        cap = f"{path}Result\\{graph_name}\\per_col\\{graph_name}_{col}_{mode}.png"
        plt.savefig(cap)
    if show: plt.show()
    plt.close()


def plt_df(
        graph_name,
        df,
        cols,
        eval_col='mean_validation_loss',
        epoch_col='k_epoch',
        keep=True,
        show=True,
):

    models = df['model_name'].unique()

    bestepoch = {}
    for model in models:
        df_model = df[df['model_name'] == model]
        val_opt, val_pos = best_epoch(
            df_model=df_model,
            eval_col=eval_col,
            epoch_col=epoch_col,
        )
        bestepoch[model] = val_pos

    for col in cols:
        single_plot(
            df=df,
            col=col,
            epoch_col=epoch_col,
            bestepoch=bestepoch,
            graph_name=graph_name,
            keep=keep,
            show=show,
        )