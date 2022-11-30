import os
import numpy as np
import torch
from utils.dgraphfin import DGraphFin
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import LineGraph
from torch_scatter import scatter_add
from argparse import ArgumentParser
from torchmetrics import Accuracy, AUROC, Precision, Recall
import mlflow
import mlflow.pytorch


def parameter_parser():
    parser = ArgumentParser(description="Run the code.")
    parser.add_argument("--dataset",
                        default="dgraphfin")
    parser.add_argument("--expr_name",
                        default="node_edge_aggr")
    parser.add_argument("--device",
                        default="5",
                        help="gpu device num")
    parser.add_argument("--epochs",
                        type=int,
                        default=200)
    parser.add_argument("--hidden_size",
                        type=int,
                        default=128)
    parser.add_argument("--batch_size",
                        type=int,
                        default=512)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.003)
    parser.add_argument("--num_layers",
                        type=int,
                        default=2)
    parser.add_argument("--dropout",
                        type=float,
                        default=0)
    args = parser.parse_args()

    return args


def eval_fn(args, model, data_loader, mode="valid"):
    test_acc = Accuracy()
    test_auroc = AUROC(num_classes=args.out_size)
    test_precision = Precision(num_classes=args.out_size)
    test_recall = Recall(num_classes=args.out_size)
    model.eval()
    outs = []
    preds = []
    labels = []
    linegraph = LineGraph()
    for batch in data_loader:
        batch = batch.to(args.device)
        mask = batch.valid_mask if mode == "valid" else batch.test_mask
        clone_batch = batch.clone()
        lg = linegraph(clone_batch)
        idx = batch.edge_index
        src = torch.ones_like(idx)
        H = scatter_add(src, idx, dim=0).nonzero().transpose(1, 0).to(args.device)
        et = lg.edge_timestamp.unsqueeze(1).float()
        out = model(batch.x, et, H, batch.edge_index, lg.edge_index)
        out = out.exp()
        out = out[mask][:args.batch_size].cpu().detach()
        pred = out.argmax(dim=-1)
        label = batch.y[mask][:args.batch_size].squeeze(1).cpu().detach()
        labels.append(label)
        preds.append(pred)
        outs.append(out)
    outs = torch.vstack(outs)
    labels = torch.cat(labels, dim=-1)
    preds = torch.cat(preds, dim=-1)

    result = {"y_true": labels.cpu().detach().numpy(), "y_pred": outs[:, 1].cpu().detach().numpy()}
    result_path = os.path.join("results", "cat_node_edge_agg")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    np.save(result_path + "/result.npy", result)

    acc = test_acc(preds, labels)
    prec = test_precision(preds, labels)
    recall = test_recall(preds, labels)
    auroc = test_auroc(outs, labels)
    return acc, auroc, prec, recall


def log_params(params):
    mlflow.log_params(params)


def log_metrics(**kwargs):
    mlflow.log_metrics(kwargs)


def eval_model(args):
    args.device= "cuda:" +args.device
    args.expr_name = "_".join([args.expr_name, str(args.hidden_size) +'h'])
    args.out_size = 2
    args.timestamp_size = 1
    g_cpu = torch.Generator()
    args.seed = g_cpu.seed()

    dataset = DGraphFin("/data/huhy/datasets/", name="DGraphFin")
    data = dataset[0]
    x = data.x
    x = (x - x.mean(0)) / x.std(0)
    data.x = x
    args.num_features = x.size(1)

    data.n_id = torch.arange(data.x.size(0))
    data.train_nodes = data.train_mask
    data.valid_nodes = data.valid_mask
    data.test_nodes = data.test_mask
    data.train_mask = torch.isin(data.n_id, data.train_mask)
    data.valid_mask = torch.isin(data.n_id, data.valid_mask)
    data.test_mask = torch.isin(data.n_id, data.test_mask)


    test_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=args.batch_size, directed=False, input_nodes=data.test_nodes)

    run_id = "6af9340dadc144a68470c18088a2613d"
    args.run_id = run_id
    model = mlflow.pytorch.load_model("runs:/" + run_id + '/model').to(args.device)

    # model.load_state_dict(torch.load(os.path.join(model_path, "weight.pth")))
    # model = mlflow.pytorch.load_model(mlflow.get_artifact_uri("model"))
    test_acc, test_auroc, test_prec, test_recall = eval_fn(args, model, test_loader, mode="test")
    print (test_acc, test_auroc, test_prec, test_recall)


if __name__ == "__main__":
    args = parameter_parser()
    eval_model(args)