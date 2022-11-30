import os
import torch
import torch.nn.functional as F
from utils.dataset_reader import read_dataset
from torch_geometric.loader import NeighborSampler
import torch_geometric.transforms as T
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from model_zoo.baseline import SAGE_NeighSampler as Model
import mlflow
import mlflow.pytorch
from time import time
from utils.evaluator import Evaluator


def parameter_parser():
    parser = ArgumentParser(description="Run the code.")
    parser.add_argument("--dataset",
                        default="dgraphfin")
    parser.add_argument("--dataset_dir",
                        type=str)
    parser.add_argument("--expr_name",
                        default="graphsage")
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
                        default=32)
    parser.add_argument("--batch_size",
                        type=int,
                        default=1024)
    parser.add_argument("--learning_rate",
                        type=float,
                        default=0.01)
    parser.add_argument("--num_layers",
                        type=int,
                        default=2)
    parser.add_argument("--dropout",
                        type=float,
                        default=0)
    parser.add_argument("--transform",
                        action="store_true",
                        default=False)
    args = parser.parse_args()
    return args


# def eval_fn(args, model, data_loader, mode="valid", data=None):
#     test_acc = Accuracy(num_classes=args.out_size)
#     test_auroc = AUROC(num_classes=args.out_size)
#     test_precision = Precision(num_classes=args.out_size, ignore_index=0)
#     test_recall = Recall(num_classes=args.out_size, ignore_index=0)
#     model.eval()
#     outs = []
#     preds = []
#     labels = []
#
#     for batch in data_loader:
#         # batch = batch.to(args.device)
#         # mask = batch.valid_mask if mode == "valid" else batch.test_mask
#
#         batch_size, n_id, adjs = batch
#         edge_index = [adj.to(args.device) for adj in adjs]
#         x = data.x[n_id].to(args.device)
#         y = data.y[n_id[:batch_size]].squeeze(1)
#         out = model(x, edge_index)
#         # out = out[mask][:args.batch_size].cpu().detach()
#         pred = out.argmax(dim=-1).cpu().detach()
#         # label = batch.y[mask][:args.batch_size].squeeze(1).cpu().detach()
#         label=y
#         labels.append(label)
#         preds.append(pred)
#         outs.append(out)
#     outs = torch.vstack(outs).cpu().detach()
#     labels = torch.cat(labels, dim=-1)
#     preds = torch.cat(preds, dim=-1)
#
#     acc = test_acc(preds, labels)
#     prec = test_precision(preds, labels)
#     recall = test_recall(preds, labels)
#     auroc = test_auroc(outs, labels)
#     return acc, auroc, prec, recall


@torch.no_grad()
def eval_fn(layer_loader, model, data, evaluator, args, mode = "valid", no_conv=False):
    # data.y is labels of shape (N, )
    model.eval()

    out = model.inference(data.x, layer_loader, args.device)
    #     out = model.inference_all(data)
    y_pred = out.exp()  # (N,num_classes)

    losses, eval_results = dict(), dict()

    node_id = data.get(mode + "_nodes")
    node_id = node_id.to(args.device)
    loss = F.nll_loss(out[node_id].cpu(), data.y[node_id].squeeze(1)).item()
    eval_results = evaluator.eval(data.y[node_id].squeeze(1), y_pred[node_id])

    return eval_results, loss, y_pred


def log_params(args):
    mlflow.log_param("expr_name", args.expr_name)
    mlflow.log_param("batch_size", args.batch_size)
    mlflow.log_param("hidden_size", args.hidden_size)
    mlflow.log_param("seed", args.seed)


def log_metrics(**kwargs):
    mlflow.log_metrics(kwargs)


def train(args):
    args.device = "cuda:" + args.device
    args.out_size = 2
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

        log_params(args)

        data = read_dataset(args.dataset, args.dataset_dir, transform= T.ToSparseTensor() if args.transform else None)
        args.num_features = data.x.size(1)

        # train_loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=args.batch_size, directed=False, shuffle=True, input_nodes=data.train_nodes)
        # valid_loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=args.batch_size, directed=False, input_nodes=data.valid_nodes)
        # test_loader = NeighborLoader(data, num_neighbors=[10, 5], batch_size=args.batch_size, directed=False, input_nodes=data.test_nodes)

        train_loader = NeighborSampler(data.adj_t, node_idx=data.train_nodes, sizes=[10, 5], batch_size=args.batch_size, shuffle=True)
        layer_loader = NeighborSampler(data.adj_t, node_idx=None, sizes=[-1], batch_size=args.batch_size)
        evaluator = Evaluator("auc")

        log_path = os.path.join("../log", args.dataset, args.expr_name)
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        writer = SummaryWriter(log_dir=log_path)
        model = Model(args.num_features, args.hidden_size, args.out_size, args.num_layers, args.dropout, batchnorm=False).to(args.device)
        optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=5e-7)

        max_aucroc = 0
        max_ep = 0
        for ep in range(args.epochs):
            model.train()
            total_loss = 0
            i = 0
            start_time = time()
            for i, batch in enumerate(train_loader):
                optimizer.zero_grad()
                # batch = batch.to(args.device)

                batch_size, n_id, adjs = batch
                edge_index = [adj.to(args.device) for adj in adjs]
                x = data.x[n_id].to(args.device)
                y = data.y[n_id[:batch_size]].squeeze(1).to(args.device)
                out = model(x, edge_index)
                # loss = F.nll_loss(out[batch.train_mask], batch.y.squeeze(1)[batch.train_mask], weight=torch.tensor([0.0126, 0.9873]).to(args.device))
                # loss = F.nll_loss(out[batch.train_mask], batch.y.squeeze(1)[batch.train_mask])

                loss = F.nll_loss(out, y)
                loss.backward()
                optimizer.step()
                total_loss += loss
            avg_loss = total_loss / (i + 1)
            # val_acc, val_auroc, val_prec, val_recall = eval_fn(args, model, valid_loader, mode="valid", data=data)
            # eval_results, val_loss, _ = eval_fn(layer_loader, model, data, evaluator, args, mode = "valid")
            # print(eval_results["acc"], eval_results["auc"], eval_results["prec"], eval_results["recall"])
            # writer.add_scalar("average loss", avg_loss, ep + 1)
            # writer.add_scalar("val acc", eval_results["acc"], ep + 1)
            # if (eval_results["auc"] > max_aucroc):
            #     # torch.save(model.state_dict(), os.path.join(model_path, "weight.pth"))
            #     mlflow.pytorch.log_model(model, "model")
            #     max_aucroc = eval_results["auc"]
            #     max_ep = ep
            # if (ep - max_ep >= 10):
            #     break
            print("take {}s for epoch {}".format(round(time() - start_time, 2), ep))
        # model.load_state_dict(torch.load(os.path.join(model_path, "weight.pth")))

        # model = mlflow.pytorch.load_model(mlflow.get_artifact_uri("model"))
        mlflow.pytorch.log_model(model, "model")
        eval_results, val_loss, _ = eval_fn(layer_loader, model, data, evaluator, args, mode="valid")
        print(eval_results["acc"], eval_results["auc"], eval_results["prec"], eval_results["recall"])

        eval_results, avg_loss, _ = eval_fn(layer_loader, model, data, evaluator, args, mode = "test")
        print(eval_results["acc"], eval_results["auc"], eval_results["prec"], eval_results["recall"])
        log_metrics(aucroc=eval_results["auc"],
                    accuracy=eval_results["acc"],
                    precision=eval_results["prec"],
                    recall=eval_results["recall"])
    mlflow.end_run()


if __name__ == "__main__":
    args = parameter_parser()
    train(args)
