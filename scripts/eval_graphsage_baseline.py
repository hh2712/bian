import os
import torch
from utils.dgraphfin import DGraphFin
from torch_geometric.loader import NeighborSampler
import torch_geometric.transforms as T
from argparse import ArgumentParser
import mlflow
import mlflow.pytorch
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np


def parameter_parser():
    parser = ArgumentParser(description="Run the code.")
    parser.add_argument("--device",
                        default="6",
                        help="gpu device num")
    args = parser.parse_args()
    return args


def eval_model(args):
    args.dataset = "dgraphfin"
    args.expr_name = "graphsage_128h"
    args.epochs = 200
    args.num_features = 17
    args.hidden_size = 128
    args.out_size = 2
    args.batch_size = 512 * 16
    args.learning_rate = 0.003
    args.num_layers = 2
    args.dropout = 0
    args.device = "cuda:7"

    dataset = DGraphFin("/data/huhy/datasets/", name="DGraphFin", transform=T.ToSparseTensor())
    data = dataset[0]
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x

    data.n_id = torch.arange(data.x.size(0))
    data.train_nodes = data.train_mask
    data.valid_nodes = data.valid_mask
    data.test_nodes = data.test_mask
    data.train_mask = torch.isin(data.n_id, data.train_mask)
    data.valid_mask = torch.isin(data.n_id, data.valid_mask)
    data.test_mask = torch.isin(data.n_id, data.test_mask)
    row, col, _ = data.adj_t.to_symmetric().t().coo()
    data.edge_index = torch.stack([row, col], dim=0)

    layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=args.batch_size)

    model_path = os.path.join("params", args.dataset, args.expr_name)
    # model = Model(args.num_features, args.hidden_size, args.out_size, args.num_layers, args.dropout,
    #               batchnorm=False).to(args.device)
    run_id = "ce856d2be98f4bb0929ca8c8e97d0980"

    model = mlflow.pytorch.load_model("runs:/" + run_id + '/model')
    model.eval()

    out = model.embeds(data.x, layer_loader, args.device)
    y_pred = out[data.test_nodes].cpu().detach().numpy()
    y_true = data.y[data.test_nodes].squeeze(1).numpy()

    result = {"y_true": y_true, "y_pred": y_pred}
    result_path = os.path.join("results", "graphsage")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    np.save(result_path + "/preds.npy", result)

    # fpr, tpr, threshold = roc_curve(y_true, y_pred, pos_label=1)
    # plt.plot(fpr, tpr, marker = 'o')
    # plt.draw()
    # plt.savefig("fig/graphsage_roc_curve.png")


if __name__ == "__main__":
    args = parameter_parser()
    eval_model(args)