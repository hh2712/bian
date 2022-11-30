import os
import torch
import torch.nn.functional as F
from utils.dataset_reader import read_dataset
from torch_geometric.loader import NeighborLoader
from torch_geometric.transforms import LineGraph
from torch_scatter import scatter_add
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from model_zoo import NodeEdgeAggregatorV2 as Model
from torchmetrics import Accuracy, AUROC, Precision, Recall
import mlflow
import mlflow.pytorch
from time import time


def parameter_parser():
    parser = ArgumentParser(description="Run the code.")
    parser.add_argument("--dataset",
                        default="dgraphfin")
    parser.add_argument("--dataset_dir",
                        type=str)
    parser.add_argument("--expr_name",
                        default="node_edge_aggr")
    parser.add_argument("--device",
                        default="6",
                        help="gpu device num")
    parser.add_argument("--epochs",
                        type=int,
                        default=200)
    parser.add_argument("--time_encoder",
                        type=str,
                        default="harmonic")
    parser.add_argument("--hidden_size",
                        type=int,
                        default=128)
    parser.add_argument("--batch_size",
                        type=int,
                        default=8096)
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
    test_precision = Precision(num_classes=args.out_size, ignore_index=0)
    test_recall = Recall(num_classes=args.out_size, ignore_index=0)
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
        out = out[mask][:args.batch_size].cpu().detach()
        out = out.exp()
        pred = out.argmax(dim=-1)
        label = batch.y[mask][:args.batch_size].squeeze(1).cpu().detach()
        labels.append(label)
        preds.append(pred)
        outs.append(out)
    outs = torch.vstack(outs)
    labels = torch.cat(labels, dim=-1)
    preds = torch.cat(preds, dim=-1)

    # result = {"y_true": labels.cpu().detach().numpy(), "y_pred": preds.cpu().detach().numpy()}
    # result_path = os.path.join("results", args.run_id, "result.npy")
    # np.save(result_path, result)

    acc = test_acc(preds, labels)
    prec = test_precision(preds, labels)
    recall = test_recall(preds, labels)
    auroc = test_auroc(outs, labels)
    return acc, auroc, prec, recall


def log_params(params):
    mlflow.log_params(params)


def log_metrics(**kwargs):
    mlflow.log_metrics(kwargs)


def train(args):
    args.device= "cuda:" +args.device
    args.expr_name = "_".join([args.expr_name, str(args.hidden_size) +'h'])
    args.out_size = 2
    args.timestamp_size = 1
    g_cpu = torch.Generator()
    args.seed = g_cpu.seed()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    mlflow.set_experiment(args.dataset.lower())
    mlflow.set_tracking_uri("http://0.0.0.0:12007")
    run = mlflow.active_run()
    if run:
        print("Active run_id: {}".format(run.info.run_id))
        mlflow.end_run()
    with mlflow.start_run():
        model_path = os.path.join("params", args.dataset, args.expr_name)
        if not os.path.exists(model_path):
            os.makedirs(model_path)

        data = read_dataset(args.dataset, args.dataset_dir)
        args.num_features = data.x.size(1)

        train_loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=args.batch_size, directed=False, shuffle=True, input_nodes=data.train_nodes)
        valid_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=args.batch_size, directed=False, input_nodes=data.valid_nodes)
        test_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=args.batch_size, directed=False, input_nodes=data.test_nodes)

        log_path = os.path.join("../log", args.dataset, args.expr_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        log_params(args.__dict__)
        writer = SummaryWriter(log_dir=log_path)
        model = Model(args).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-7)
        linegraph = LineGraph()

        max_aucroc = 0
        max_ep = 0
        for ep in range(args.epochs):
            model.train()
            total_loss = 0
            i = 0
            start_time = time()
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                batch = batch.to(args.device)
                clone_batch = batch.clone()
                lg = linegraph(clone_batch)
                idx = batch.edge_index
                src = torch.ones_like(idx)
                H = scatter_add(src, idx, dim=0).nonzero().transpose(1, 0).to(args.device)
                et = lg.edge_timestamp.unsqueeze(1).float()
                out = model(batch.x, et, H, batch.edge_index, lg.edge_index)
                # loss = F.nll_loss(out[batch.train_mask], batch.y.squeeze(1)[batch.train_mask], weight=torch.tensor([0.0126, 0.9873]).to(args.device))
                loss = F.nll_loss(out[batch.train_mask], batch.y.squeeze(1)[batch.train_mask])
                loss.backward()
                optimizer.step()
                total_loss += loss
            avg_loss = total_loss / (i + 1)
            val_acc, val_auroc, val_prec, val_recall = eval_fn(args, model, valid_loader, mode="valid")
            # print(val_acc, val_auroc, val_prec, val_recall)
            writer.add_scalar("average loss", avg_loss, ep + 1)
            writer.add_scalar("val acc", val_acc, ep + 1)
            writer.add_scalar("val auc", val_auroc, ep + 1)
            if (val_auroc > max_aucroc) and ep > 50:
                # torch.save(model.state_dict(), os.path.join(model_path, "weight.pth"))
                mlflow.pytorch.log_model(model, "model")
                max_aucroc = val_auroc
                max_ep = ep
            if (ep - max_ep >= 10):
                break
            print("take {}s for epoch {}".format(round(time() - start_time, 2), ep + 1))
        mlflow.pytorch.log_model(model, "model")
        # model.load_state_dict(torch.load(os.path.join(model_path, "weight.pth")))
        # model = mlflow.pytorch.load_model(mlflow.get_artifact_uri("model"))
        test_acc, test_auroc, test_prec, test_recall = eval_fn(args, model, test_loader, mode="test")
        print (test_acc, test_auroc, test_prec, test_recall)
        log_metrics(aucroc=test_auroc.item(),
                    accuracy=test_acc.item(),
                    precision=test_prec.item(),
                    recall=test_recall.item())
    mlflow.end_run()


if __name__ == "__main__":
    args = parameter_parser()
    train(args)