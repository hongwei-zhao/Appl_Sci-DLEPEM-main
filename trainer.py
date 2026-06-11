import sys
import logging
import copy
import time

import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os
import numpy as np
import torch.distributed as dist


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])
    args["local_time"] = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())
    # Get GPU model information
    gpu_names = []
    for dev in device:
        if dev.isdigit():  # Check if it is a GPU device index
            idx = int(dev)
            if idx < torch.cuda.device_count():
                gpu_names.append(torch.cuda.get_device_name(idx))
    args['gpu_models'] = gpu_names  # Add GPU model list to args

    args['local_time'] = time.strftime("%Y-%m-%d-%H:%M:%S", time.localtime())

    for seed in seed_list:
        args["seed"] = seed
        _train(args)


def _train(args):

    init_cls = 0 if args ["init_cls"] == args["increment"] else args["init_cls"]
    logs_name = "logs/{}/{}/{}/{}".format(args["model_name"],args["dataset"], init_cls, args['increment'])
    
    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["backbone_type"],
    )
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
    )

    _set_random(args["seed"])
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        args,
    )
    
    args["nb_classes"] = data_manager.nb_classes # update args
    args["nb_tasks"] = data_manager.nb_tasks
    model = factory.get_model(args["model_name"], args)

    cnn_curve, nme_curve, route_curve, lora_expert_curve = {"top1": [], "top5": []}, {"top1": [], "top5": []}, {
        "top1": [], "top5": []}, {"top1": [], "top5": []}
    cnn_matrix = []
    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )

        model.incremental_train(data_manager)

        cnn_accy, nme_accy, route_accy, lora_expert_accy = model.eval_task_lora_expert()

        model.after_task()

        cnn_keys = [key for key in cnn_accy["grouped"].keys() if '-' in key]
        cnn_values = [cnn_accy["grouped"][key] for key in cnn_keys]
        cnn_matrix.append(cnn_values)

        logging.info("CNN: {}".format(cnn_accy["grouped"]))
        logging.info("Route: {}".format(route_accy["grouped"]))
        logging.info("LoRA_Expert: {}".format(lora_expert_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])
        route_curve["top1"].append(route_accy["top1"])
        lora_expert_curve["top1"].append(lora_expert_accy["top1"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("Route top1 curve: {}".format(route_curve["top1"]))
        logging.info("LoRA expert top1 curve: {}\n".format(lora_expert_curve["top1"]))
        logging.info("Average Accuracy (CNN): {:.2f}".format(sum(cnn_curve["top1"]) / len(cnn_curve["top1"])))
        logging.info("Average Accuracy (Route): {:.2f}".format(sum(route_curve["top1"]) / len(route_curve["top1"])))
        logging.info("Average Accuracy (LoRA expert):{:.2f}\n".format(
            sum(lora_expert_curve["top1"]) / len(lora_expert_curve["top1"])))

    print(f"\n{'=' * 80}")
    print(
        "Finished {}_init{}_inc{}: {}  ".format(
            args["dataset"],
            args["init_cls"],
            args["increment"],
            args["backbone_type"],
        )
    )
    print("Average Accuracy (Top1): {}".format(round(sum(cnn_curve["top1"]) / len(cnn_curve["top1"]), 2)))
    print("PD: {:.2f}".format(cnn_curve["top1"][0] - cnn_curve["top1"][-1]))
    if len(cnn_matrix) > 0:
        np_acctable = np.zeros([task + 1, task + 1])
        for idxx, line in enumerate(cnn_matrix):
            idxy = len(line)
            np_acctable[idxx, :idxy] = np.array(line)
        print("Accuracy Matrix (CNN):")
        print(np_acctable)

    print(f"{'=' * 80}\n")
def _set_device(args):
    """
    Set device (CPU/GPU) for training.
    """
    device_type = args["device"]
    gpus = []

    for device in device_type:
        if device == -1:
            device = torch.device("cpu")
        else:
            device = torch.device("cuda:{}".format(device))

        gpus.append(device)

    args["device"] = gpus


def _set_random(seed=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):
    for key, value in args.items():
        logging.info("{}: {}".format(key, value))