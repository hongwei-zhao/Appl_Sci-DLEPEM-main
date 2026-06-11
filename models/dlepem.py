import copy
import logging
import os
import time

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from utils.inc_net import DLEPEMNet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy
import timm
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler as DS

EPSILON = 1e-8
class Learner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)

        self._network = DLEPEMNet(args, True)
        self.cls_mean = dict()
        self.cls_cov = dict()
        self.cls2task = dict()

        self.batch_size = args["batch_size"]
        self.init_lr = args["init_lr"]
        self.router_lr = args["router_lr"]
        self.router_epochs = args["router_epochs"]
        self.weight_decay = args["weight_decay"] if args["weight_decay"] is not None else 0.0005
        self.min_lr = args["min_lr"] if args["min_lr"] is not None else 1e-8
        self.args = args
        # self.ensemble = args["ensemble"]

        for n, p in self._network.backbone.named_parameters():
            if 'lora' not in n and 'head' not in n:
                p.requires_grad = False

        total_params = sum(p.numel() for p in self._network.backbone.parameters())
        logging.info(f'{total_params:,} model total parameters.')
        total_trainable_params = sum(p.numel() for p in self._network.backbone.parameters() if p.requires_grad)
        logging.info(f'{total_trainable_params:,} model training parameters.')

        self.router_model = timm.create_model(args['backbone_type'].rsplit('_', 1)[0], pretrained=True,
                                              num_classes=0).to(self._device).eval()
        # print("Router Model: ", args['backbone_type'].rsplit('_', 1)[0])
        self.router_cls_mean = []

    def replace_fc(self):
        model = self._network.to(self._device)
        model.eval()
        embedding_list = []
        label_list = []
        with torch.no_grad():
            for i, batch in enumerate(self.train_loader_for_protonet):
                (_, data, label) = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model(data, adapter_id=self._cur_task, train=False)['features']
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = np.unique(self.train_dataset.labels)
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.fc.weight.data[class_index] = proto
        return model

    def after_task(self):
        self._known_classes = self._total_classes
        if 'q' in self.args['lora_positions']:
            self._network.backbone.old_router_q_lora = copy.deepcopy(self._network.backbone.router_q_lora)
        if 'k' in self.args['lora_positions']:
            self._network.backbone.old_router_k_lora = copy.deepcopy(self._network.backbone.router_k_lora)
        if 'v' in self.args['lora_positions']:
            self._network.backbone.old_router_v_lora = copy.deepcopy(self._network.backbone.router_v_lora)
        if 'mlp' in self.args['lora_positions']:
            self._network.backbone.old_router_mlp_lora = copy.deepcopy(self._network.backbone.router_mlp_lora)

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(self._cur_task)

        for i in range(self._known_classes, self._total_classes):
            self.cls2task[i] = self._cur_task

        self._network.update_fc(self._total_classes)
        logging.info("Learning on {}-{}".format(self._known_classes, self._total_classes))

        # Dataloader settings
        self.data_manager = data_manager
        self.train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="train"
        )

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.args.get("num_workers", 4),
            # sampler=train_sampler,
            shuffle=True  # Auto-shuffle when not distributed
        )

        # Test set settings (keep full data)
        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes),
            source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            num_workers=self.args.get("num_workers", 4),
            shuffle=False
        )

        # Prototypical network training data
        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train", mode="test"
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet,
            batch_size=self.batch_size,
            num_workers=self.args.get("num_workers", 4),
            shuffle=False
        )

        self._train(self.train_loader, self.test_loader)

    def _train(self, train_loader, test_loader):
        self._network.backbone.to(self._device)

        optimizer = self.get_optimizer(self._network.backbone)
        scheduler = self.get_scheduler(optimizer)

        router_optimizer = self.get_optimizer(self._network.backbone)
        router_scheduler = self.get_scheduler(router_optimizer)
        
        self._init_train(train_loader, test_loader, optimizer, scheduler)
        self.replace_fc()
        
        self._network.backbone.adapter_update()
        # Router update
        self.router_model_update(train_loader, router_optimizer, router_scheduler)

    def get_optimizer(self, model):
        base_params = [p for name, p in model.named_parameters() if 'cur' in name and p.requires_grad]
        base_fc_params = [p for name, p in model.named_parameters() if 'lora' not in name and p.requires_grad]
        router_params = [p for name, p in model.named_parameters() if 'router' in name  and p.requires_grad]

        base_params = {'params': base_params, 'lr': self.init_lr, 'weight_decay': self.weight_decay}
        base_fc_params = {'params': base_fc_params, 'lr': self.init_lr * 0.1, 'weight_decay': self.weight_decay}
        router_params = {'params': router_params, 'lr': self.router_lr, 'weight_decay': self.weight_decay}

        network_params = [base_params, base_fc_params,router_params]

        if self.args['optimizer'] == 'sgd':
            optimizer = optim.SGD(
                network_params,
                momentum=0.9,
            )
        elif self.args['optimizer'] == 'adam':
            optimizer = optim.Adam(
                network_params,
            )

        elif self.args['optimizer'] == 'adamw':
            optimizer = optim.AdamW(
                network_params,
            )

        return optimizer

    def get_scheduler(self, optimizer):
        if self.args["scheduler"] == 'cosine':
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer, T_max=self.args['tuned_epoch'],
                                                             eta_min=self.min_lr)
        elif self.args["scheduler"] == 'steplr':
            scheduler = optim.lr_scheduler.MultiStepLR(optimizer=optimizer, milestones=self.args["init_milestones"],
                                                       gamma=self.args["init_lr_decay"])
        elif self.args["scheduler"] == 'constant':
            scheduler = None

        return scheduler

    def router_model_update(self, train_loader, router_optimizer, router_scheduler):
        # Get sampling method
        router_update_method = self.args["router_train_method"].lower()
        if router_update_method == 'prototype':
            self.router_model_update_by_prototype(self.data_manager)
        elif router_update_method == 'knn':
            # knn sampling, save embedding to self._network.example
            self.router_model_update_by_knn(self.train_loader)
        elif router_update_method == 'ema_prototype':
            self.router_model_update_by_ema_prototype(self.data_manager)
        elif router_update_method == 'prototype_ensemble':
            if self._cur_task == 0:
                if 'q' in self.args['lora_positions']:
                    self._network.backbone.router_q_lora = copy.deepcopy(self._network.backbone.cur_q_lora)
                if 'k' in self.args['lora_positions']:
                    self._network.backbone.router_k_lora = copy.deepcopy(self._network.backbone.cur_k_lora)
                if 'v' in self.args['lora_positions']:
                    self._network.backbone.router_v_lora = copy.deepcopy(self._network.backbone.cur_v_lora)
                if 'mlp' in self.args['lora_positions']:
                    self._network.backbone.router_mlp_lora = copy.deepcopy(self._network.backbone.cur_mlp_lora)
            if self._cur_task > 0:
                self.router_model_update_by_prototype_ensemble(train_loader, router_optimizer, router_scheduler)
            self._update_router_cls_mean(self.data_manager)  # Update router prototype
        else:
            print("No need to update router")
    def _init_train(self, train_loader, test_loader, optimizer, scheduler):
        prog_bar = tqdm(range(self.args['tuned_epoch']))
        for _, epoch in enumerate(prog_bar):
            self._network.backbone.train()

            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device,
                                                                                         non_blocking=True)
                output = self._network(inputs, adapter_id=self._cur_task, train=True)
                logits = output["logits"][:, :self._total_classes]
                logits[:, :self._known_classes] = float('-inf')

                loss = F.cross_entropy(logits, targets.long())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            if scheduler:
                scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_accy {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args['tuned_epoch'],
                losses / len(train_loader),
                train_acc,
            )
            prog_bar.set_description(info)

        logging.info(info)

    def router_model_update_by_prototype_ensemble(self, train_loader, optimizer, scheduler):
        # self.router_model.eval() # get frozen_prototype
        prog_bar = tqdm(range(self.args['router_epochs']))

        for _, epoch in enumerate(prog_bar):
            losses = 0.0
            train_acc = 0.0
            temperature = 1.0
            alpha = self.args['alpha']
            for i, (_, inputs, targets) in enumerate(train_loader):
                inputs, targets = inputs.to(self._device, non_blocking=True), targets.to(self._device,
                                                                                         non_blocking=True)
                router_embedding = self._network.backbone.forward_features_router(inputs)['features']
                old_router_embedding = self._network.backbone.forward_features_old_router(inputs)['features']

                # KL
                router_probs = F.softmax(router_embedding / temperature, dim=-1)
                old_router_probs = F.softmax(old_router_embedding.detach() / temperature, dim=-1)
                lora_probs = F.softmax(self._network.fc.weight[targets].detach() / temperature, dim=-1)

                # Calculate KL divergence loss
                plasticity_fd_loss = F.kl_div(router_probs.log(), lora_probs, reduction='batchmean')
                stable_fd_loss = F.kl_div(router_probs.log(), old_router_probs, reduction='batchmean')

                loss = alpha * plasticity_fd_loss + (1 - alpha) * stable_fd_loss

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

            if scheduler:
                scheduler.step()
            # train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)

            info = "Router {}, Epoch {}/{} => Loss {:.3f}".format(
                self._cur_task,
                epoch + 1,
                self.args['router_epochs'],
                losses / len(train_loader),
            )
            prog_bar.set_description(info)
        logging.info(info)

    @torch.no_grad()
    def router_model_update_by_prototype(self, data_manager):
        router_cls_mean = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )

            idx_loader = DataLoader(
                idx_dataset,
                batch_size=self.batch_size,
                num_workers=self.args.get("num_workers", 4),
                shuffle=True,
            )
            vectors, _ = self._extract_vectors_by_sample_mean(idx_loader)  # Different sampling methods for different prototype centers
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            # mean = mean / np.linalg.norm(mean)  # Vector standardization
            mean = torch.tensor(mean, device=self._device)
            router_cls_mean.append(mean.unsqueeze(0))
        # Convert list to tensor
        router_cls_mean_tensor = torch.cat(router_cls_mean, dim=0)
        if len(self.router_cls_mean) == 0:
            self.router_cls_mean = router_cls_mean_tensor
        else:
            self.router_cls_mean = torch.cat((self.router_cls_mean, router_cls_mean_tensor), dim=0).to(self._device)
    @torch.no_grad()
    def _update_router_cls_mean(self, data_manager):
        # Set backbone to eval mode
        self._network.backbone.eval()
        router_cls_mean = []

        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )

            idx_loader = DataLoader(
                idx_dataset,
                batch_size=self.batch_size,
                num_workers=self.args.get("num_workers", 4),
                shuffle=True,
            )

            # Extract original feature vectors for constructing class centers
            frozen_vectors, _ = self._extract_vectors_by_sample_mean(idx_loader)
            frozen_vectors_norm = (frozen_vectors.T / (np.linalg.norm(frozen_vectors.T, axis=0) + EPSILON)).T
            frozen_vectors_mean = torch.tensor(np.mean(frozen_vectors_norm, axis=0), device=self._device)

            # Calculate the mean of feature vectors obtained through router_model
            vectors = []
            for _, _inputs, _targets in idx_loader:
                # Directly call backbone.forward_features_router, no need to consider .module
                _features = self._network.backbone.forward_features_router(_inputs.to(self._device))["features"]
                vectors.append(_features)
            vectors = torch.cat(vectors, dim=0)
            lora_vectors_mean = vectors.mean(dim=0).to(self._device)

            # Concatenate the two feature vector means (dimension can be adjusted as needed)
            mean = torch.cat((frozen_vectors_mean.unsqueeze(0), lora_vectors_mean.unsqueeze(0)), dim=1)
            router_cls_mean.append(mean)

        # Merge all class center lists into a tensor, and update router_cls_mean
        router_cls_mean_tensor = torch.cat(router_cls_mean, dim=0)
        if len(self.router_cls_mean) == 0:
            self.router_cls_mean = router_cls_mean_tensor.to(self._device)
        else:
            self.router_cls_mean = torch.cat((self.router_cls_mean, router_cls_mean_tensor), dim=0).to(self._device)

    def router_model_sample_by_prototype(self, data_manager):
        router_cls_mean = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size, num_workers=self.args.get("num_workers", 4), shuffle=False
            )
            vectors, _ = self._extract_vectors_by_sample_mean(idx_loader)  # Different sampling methods for different prototype centers
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            # mean = mean / np.linalg.norm(mean)  # Vector standardization
            mean = torch.tensor(mean, device=self._device)
            router_cls_mean.append(mean.unsqueeze(0))
        # Convert list to tensor
        router_cls_mean_tensor = torch.cat(router_cls_mean, dim=0)
        if len(self.router_cls_mean) == 0:
            self.router_cls_mean = router_cls_mean_tensor
        else:
            self.router_cls_mean = torch.cat((self.router_cls_mean, router_cls_mean_tensor), dim=0).to(self._device)

    @torch.no_grad()
    def router_model_sample_by_ema_prototype(self, data_manager):
        self._network.backbone.eval()
        self._network.backbone.router_ema = True
        router_cls_mean = []
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=self.batch_size, num_workers=self.args.get("num_workers", 4), shuffle=False
            )

            frozen_vectors, _ = self._extract_vectors_by_sample_mean(idx_loader)  # Different sampling methods for different prototype centers
            frozen_vectors_norm = (frozen_vectors.T / (np.linalg.norm(frozen_vectors.T, axis=0) + EPSILON)).T
            frozen_vectors_mean = torch.tensor(np.mean(frozen_vectors_norm, axis=0), device=self._device)

            vectors = []
            for _, _inputs, _targets in idx_loader:
                _vectors = self._network.backbone.forward_features_router(_inputs.to(self._device), adapter_id=0)["features"]
                vectors.append(_vectors)
            vectors = torch.cat(vectors, dim=0)
            lora_vectors_mean = vectors.mean(dim=0).to(self._device)

            mean = torch.cat((frozen_vectors_mean.unsqueeze(0), lora_vectors_mean.unsqueeze(0)), dim=1)
            router_cls_mean.append(mean)
        # Convert list to tensor
        router_cls_mean_tensor = torch.cat(router_cls_mean, dim=0)
        if len(self.router_cls_mean) == 0:
            self.router_cls_mean = router_cls_mean_tensor
        else:
            self.router_cls_mean = torch.cat((self.router_cls_mean, router_cls_mean_tensor), dim=0).to(self._device)
        self._network.backbone.router_ema = False

