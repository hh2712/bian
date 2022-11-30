import os
import torch
from utils.dgraphfin import DGraphFin
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
import numpy as np
from argparse import ArgumentParser
from torchmetrics import Accuracy, AUROC, Precision, Recall
import mlflow
import mlflow.pytorch


def parameter_parser():
    parser = ArgumentParser(description="Run the code.")
    parser.add_argument("--device",
                        default="6",
                        help="gpu device num")
    args = parser.parse_args()
    return args


def eval_fn(args, model, data_loader, mode="valid", data=None):
    test_acc = Accuracy(num_classes=args.out_size)
    test_auroc = AUROC(num_classes=args.out_size)
    test_precision = Precision(num_classes=args.out_size, ignore_index=0)
    test_recall = Recall(num_classes=args.out_size, ignore_index=0)
    model.eval()
    outs = []
    preds = []
    labels = []

    for batch in data_loader:
        batch = batch.to(args.device)
        mask = batch.valid_mask if mode == "valid" else batch.test_mask

        batch = batch.to(args.device)

        x = batch.x
        edge_index = batch.edge_index

        out = model(x, edge_index)
        out = out[mask][:args.batch_size].cpu().detach()
        pred = out.argmax(dim=-1)
        label = batch.y[mask][:args.batch_size].squeeze(1).cpu().detach()

        labels.append(label)
        preds.append(pred)
        outs.append(out)
    outs = torch.vstack(outs).cpu().detach()
    labels = torch.cat(labels, dim=-1)
    preds = torch.cat(preds, dim=-1)

    result = {"y_true": labels.cpu().detach().numpy(), "y_pred": outs[:, 1].cpu().detach().numpy()}
    result_path = os.path.join("results", "amnet")
    if not os.path.exists(result_path):
        os.mkdir(result_path)
    np.save(result_path + "/result.npy", result)

    acc = test_acc(preds, labels)
    prec = test_precision(preds, labels)
    recall = test_recall(preds, labels)
    auroc = test_auroc(outs, labels)
    return acc, auroc, prec, recall


def log_params(args):
    mlflow.log_param("expr_name", args.expr_name)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("hidden_size", args.hidden_size)
    mlflow.log_param("seed", args.seed)


def log_metrics(**kwargs):
    mlflow.log_metrics(kwargs)


def eval_model(args):
    args.dataset = "dgraphfin"
    args.expr_name = "amnet_128h"
    args.epochs = 200
    args.num_features = 17
    args.hidden_size = 128
    args.out_size = 2
    args.batch_size = 512*16
    args.learning_rate = 5e-4
    args.lr_f = 5e-2
    args.weight_decay = 1e-5
    args.num_layers = 2
    args.dropout=0
    args.order_of_filter = 5
    args.num_filters = 2
    args.beta = 1
    args.device = "cuda:5"

    g_cpu = torch.Generator()
    args.seed = g_cpu.seed()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    mlflow.set_experiment(args.dataset)
    mlflow.set_tracking_uri("http://0.0.0.0:12007")
    run = mlflow.active_run()
    if run:
        print("Active run_id: {}".format(run.info.run_id))
        mlflow.end_run()
    with mlflow.start_run():
        model_path = os.path.join("params", args.dataset, args.expr_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        log_params(args)
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

        run_id = "79222bcd8f974ca5a938e1afdded76f9"
        model = mlflow.pytorch.load_model("runs:/" + run_id + '/model').to(args.device)

        test_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=args.batch_size, directed=False, input_nodes=data.test_nodes)
        test_acc, test_auroc, test_prec, test_recall = eval_fn(args, model, test_loader, mode="test")
        print(test_acc, test_auroc, test_prec, test_recall)
        log_metrics(aucroc=test_auroc.item(),
                    accuracy=test_acc.item(),
                    precision=test_prec.item(),
                    recall=test_recall.item())
    mlflow.end_run()


if __name__ == "__main__":
    args = parameter_parser()
    eval_model(args)
