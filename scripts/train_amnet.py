import os
import torch
import torch.nn.functional as F
from utils.dataset_reader import read_dataset
from torch_geometric.loader import NeighborLoader
import torch_geometric.transforms as T
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from model_zoo import AMNet as Model
from torchmetrics import Accuracy, AUROC, Precision, Recall
import mlflow
import mlflow.pytorch
from time import time
from utils import Evaluator


def parameter_parser():
    parser = ArgumentParser(description="Run the code.")
    parser.add_argument("--dataset",
                        default="dgraphfin")
    parser.add_argument("--device",
                        default="6",
                        help="gpu device num")
    parser.add_argument("--dataset_dir",
                        type=str)
    parser.add_argument("--expr_name",
                        default="amnet")
    parser.add_argument("--epochs",
                        type=int,
                        default=200)
    parser.add_argument("--hidden_size",
                        type=int,
                        default=128)
    parser.add_argument("--batch_size",
                        type=int,
                        default=512 * 16)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=5e-4)
    parser.add_argument("--lr_f",
                        type=float,
                        default=5e-2)
    parser.add_argument("--weight_decay",
                        type=float,
                        default=1e-5)
    parser.add_argument("--num_layers",
                        type=int,
                        default=2)
    parser.add_argument("--dropout",
                        type=float,
                        default=0)
    parser.add_argument("--order_of_filter",
                        type=int,
                        default=5)
    parser.add_argument("--num_filters",
                        type=int,
                        default=2)
    parser.add_argument("--beta",
                        type=int,
                        default=1)

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


def train(args):
    args.device = "cuda:" + args.device
    g_cpu = torch.Generator()
    args.seed = g_cpu.seed()
    args.out_size = 2
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

        log_params(args)
        data = read_dataset(args.dataset, args.dataset_dir, transform=T.ToSparseTensor())
        args.num_features = data.x.size(1)
        row, col, _ = data.adj_t.to_symmetric().t().coo()
        data.edge_index = torch.stack([row, col], dim=0)
        train_loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=args.batch_size, directed=False, shuffle=True, input_nodes=data.train_nodes)
        valid_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=args.batch_size, directed=False, input_nodes=data.valid_nodes)
        test_loader = NeighborLoader(data, num_neighbors=[-1], batch_size=args.batch_size, directed=False, input_nodes=data.test_nodes)

        evaluator = Evaluator("auc")

        log_path = os.path.join("../log", args.dataset, args.expr_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        writer = SummaryWriter(log_dir=log_path)
        # model = Model(args.num_features, args.hidden_size, args.out_size, args.num_layers, args.dropout, batchnorm=False).to(args.device)
        model = Model(args.num_features, args.hidden_size, args.out_size, args.order_of_filter, args.num_filters).to(args.device)

        optimizer = torch.optim.Adam([
            dict(params=model.filters.parameters(), lr=args.lr_f),
            dict(params=model.lin, lr=args.learning_rate, weight_decay=args.weight_decay),
            dict(params=model.attn, lr=args.learning_rate, weight_decay=args.weight_decay)]

        )

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

                x = batch.x
                edge_index = batch.edge_index
                y = batch.y
                out, margin_loss = model(x, edge_index, label = ((y==1).squeeze(1).nonzero().squeeze(), (y==0).squeeze(1).nonzero().squeeze()))
                # loss = F.nll_loss(out[batch.train_mask], batch.y.squeeze(1)[batch.train_mask], weight=torch.tensor([0.0126, 0.9873]).to(args.device))
                cls_loss = F.cross_entropy(out[batch.train_mask], batch.y.squeeze(1)[batch.train_mask])
                loss = cls_loss + args.beta * margin_loss
                loss.backward()
                optimizer.step()
                total_loss += loss
            avg_loss = total_loss / (i + 1)
            val_acc, val_auroc, val_prec, val_recall = eval_fn(args, model, valid_loader, mode="valid")
            print(val_acc, val_auroc, val_prec, val_recall, avg_loss.cpu().item())
            writer.add_scalar("average loss", avg_loss, ep + 1)
            writer.add_scalar("val acc", val_acc, ep + 1)
            if (val_auroc > max_aucroc):
                # torch.save(model.state_dict(), os.path.join(model_path, "weight.pth"))
                mlflow.pytorch.log_model(model, "model")
                max_aucroc = val_auroc
                max_ep = ep
            if (ep - max_ep >= 10):
                break
            print("take {}s for epoch {}".format(round(time() - start_time, 2), ep))
        # model.load_state_dict(torch.load(os.path.join(model_path, "weight.pth")))

        # model = mlflow.pytorch.load_model(mlflow.get_artifact_uri("model"))
        mlflow.pytorch.log_model(model, "model")
        # eval_results, val_loss, _ = eval_fn(layer_loader, model, data, evaluator, args, mode="valid")
        # print(eval_results["acc"], eval_results["auc"], eval_results["prec"], eval_results["recall"])

        test_acc, test_auroc, test_prec, test_recall = eval_fn(args, model, test_loader, mode="test")
        print(test_acc, test_auroc, test_prec, test_recall)
        log_metrics(aucroc=test_auroc.item(),
                    accuracy=test_acc.item(),
                    precision=test_prec.item(),
                    recall=test_recall.item())
    mlflow.end_run()


if __name__ == "__main__":
    args = parameter_parser()
    train(args)
