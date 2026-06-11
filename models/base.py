import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist
import time

EPSILON = 1e-8
batch_size = 64

class BaseLearner(object):
    def __init__(self, args):
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]
        self.args = args

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim
    
    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def tsne(self,showcenters=False,Normalize=False):
        import umap
        import matplotlib.pyplot as plt
        print('now draw tsne results of extracted features.')
        tot_classes=self._total_classes
        test_dataset = self.data_manager.get_dataset(np.arange(0, tot_classes), source='test', mode='test')
        valloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers = 16)
        vectors, y_true = self._extract_vectors(valloader)
        if showcenters:
            fc_weight=self._network.fc.proj.cpu().detach().numpy()[:tot_classes]
            print(fc_weight.shape)
            vectors=np.vstack([vectors,fc_weight])
        
        if Normalize:
            vectors = vectors / np.linalg.norm(vectors, axis=1, keepdims=True)

        embedding = umap.UMAP(n_neighbors=5,
                      min_dist=0.3,
                      metric='correlation').fit_transform(vectors)
        
        if showcenters:
            clssscenters=embedding[-tot_classes:,:]
            centerlabels=np.arange(tot_classes)
            embedding=embedding[:-tot_classes,:]
        scatter=plt.scatter(embedding[:,0],embedding[:,1],c=y_true,s=20,cmap=plt.cm.get_cmap("tab20"))
        plt.legend(*scatter.legend_elements())
        if showcenters:
            plt.scatter(clssscenters[:,0],clssscenters[:,1],marker='*',s=50,c=centerlabels,cmap=plt.cm.get_cmap("tab20"),edgecolors='black')
        
        plt.savefig(str(self.args['model_name'])+str(tot_classes)+'tsne.pdf')
        plt.close()


    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes, self.args["init_cls"], self.args["increment"])
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )

        return ret

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        return cnn_accy, nme_accy

    def incremental_train(self):
        pass

    def _train(self):
        pass
    
    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")  # [nb_classes, N]
        scores = dists.T  # [N, nb_classes], choose the one with the smallest distance

        return np.argsort(scores, axis=1)[:, : self.topk], y_true  # [N, topk]

    def eval_task_lora_expert(self):
        # start_time = time.time()
        self._network.eval()
        self.router_model.eval()
        start_time = time.time()
        # Calculate classification accuracy
        y_pred, y_true = self._eval_classify_accuracy(self.test_loader)
        test_time = round(time.time() - start_time, 2)
        # print('Inference time: for task {} : {}'.format(self._cur_task, test_time))

        # Calculate router accuracy
        route_pred, route_true = self._eval_router_accuracy(self.test_loader)

        # Calculate classification accuracy of lora_expert
        lora_expert_pred, lora_expert_true = self._eval_lora_expert_accuracy_by_targets(self.test_loader)

        cnn_accy = self._evaluate(y_pred, y_true)
        route_accy = self._evaluate(np.expand_dims(route_pred, axis=1), route_true)  # Router accuracy
        lora_expert_accy = self._evaluate(lora_expert_pred, lora_expert_true)  # Classification accuracy of lora_expert directly selected by targets

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None
        # total_time = time.time() - start_time
        # self.test_time += round(total_time, 2)
        return cnn_accy, nme_accy, route_accy, lora_expert_accy

        # Calculate classification accuracy
    def _eval_classify_accuracy(self, loader):
        y_pred, y_true = [], []
        get_get_router_predicts_time = 0.0
        device = self._device
        # 1) Reset only once at the outermost layer
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

        # 2) Record full process baseline
        start_alloc = torch.cuda.memory_allocated(device)
        # Note: peak also starts accumulating here
        # torch.cuda.max_memory_allocated(device) is now equal to start_alloc

        # 3) Global variable used to record the maximum peak of the router part
        router_peak_delta = 0.0
        for _, (_, inputs, targets) in enumerate(loader):
            inputs, targets = inputs.to(self._device), targets.to(self._device)
            with torch.no_grad():
                torch.cuda.synchronize(device)
                mem_before_router = torch.cuda.memory_allocated(device)
                start_time = time.time()
                router_predicts = self.get_router_predicts(inputs, targets)
                get_get_router_predicts_time += round(time.time() - start_time, 2)
                torch.cuda.synchronize(device)
                # Read the peak from the beginning to this moment
                peak_after_router = torch.cuda.max_memory_allocated(device)
                # Calculate the increment of this peak for the router part
                delta_router = (peak_after_router - mem_before_router) / 1024 ** 2
                # Keep the largest one
                if delta_router > router_peak_delta:
                    router_peak_delta = delta_router
                all_features = torch.zeros(len(inputs), self._cur_task + 1, self._network.backbone.out_dim,
                                           device=self._device)
                for t_id in range(self._cur_task + 1):
                    t_features = self._network.backbone(inputs, adapter_id=t_id, train=False)["features"]
                    all_features[:, t_id, :] = t_features
                final_features = all_features[torch.arange(len(all_features)), router_predicts]
                # outputs = self._network.backbone(final_features, fc_only=True)["logits"][:, :self._total_classes]
                outputs = self._network.fc(final_features)["logits"][:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
        # # 4) After the whole process is finished, read the total peak once
        # torch.cuda.synchronize(device)
        # total_peak = torch.cuda.max_memory_allocated(device)
        # total_peak_delta = (total_peak - start_alloc) / 1024 ** 2
        # print('Inference GPU memory usage: for task {} : {} MB'.format(self._cur_task, round(total_peak_delta, 4)))
        # print('Router GPU memory usage: for task {} : {} MB'.format(self._cur_task, round(router_peak_delta, 4)))
        # print('Router time: for task {} : {}'.format(self._cur_task, get_get_router_predicts_time))
        return np.concatenate(y_pred), np.concatenate(y_true)  # [N, topk]

    def _eval_router_accuracy(self, loader):
        self.router_model.eval()
        route_pred, route_true = [], []  # Router accuracy
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            router_predicts = self.get_router_predicts(inputs, targets)
            # targets mapped to lora_expert
            experts_by_targets = torch.tensor([self.cls2task[v] for v in targets.cpu().numpy()],
                                              device=self._device)
            route_pred.append(router_predicts.cpu().numpy())
            route_true.append(experts_by_targets.cpu().numpy())
        from collections import Counter
        # Use Counter to count
        count = Counter(np.concatenate(route_pred))
        # Output each number and its occurrences
        print('Inc ' + str(self._cur_task) + ' Expert Distribution:')
        for num, freq in count.items():
            print('Expert ', num, ': ', freq)
        return np.concatenate(route_pred), np.concatenate(route_true)

    # Calculate the classification accuracy of lora_expert
    def _eval_lora_expert_accuracy_by_targets(self, loader):
        self._network.eval()
        self.router_model.eval()
        lora_expert_y_pred, lora_expert_y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            router_by_targets = torch.tensor([self.cls2task[v] for v in targets.cpu().numpy()], device=self._device)
            # router_by_targets = torch.tensor(0).expand(targets.shape).to(self._device)
            with torch.no_grad():
                all_features = torch.zeros(len(inputs), self._cur_task + 1, self._network.backbone.out_dim,
                                           device=self._device)
                for t_id in range(self._cur_task + 1):
                    t_features = self._network.backbone(inputs, adapter_id=t_id, train=False)["features"]
                    all_features[:, t_id, :] = t_features
                final_features = all_features[torch.arange(len(all_features)), router_by_targets]  # shape=[1, 768]
                # final_features = self._network.backbone(inputs, adapter_id=0, train=False)["features"]
                # outputs = self._network.backbone(final_features, fc_only=True)["logits"][:, :self._total_classes]
                outputs = self._network.fc(final_features)["logits"][:, :self._total_classes]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[
                1
            ]  # [bs, topk]
            lora_expert_y_pred.append(predicts.cpu().numpy())
            lora_expert_y_true.append(targets.cpu().numpy())
        return np.concatenate(lora_expert_y_pred), np.concatenate(lora_expert_y_true)

    def get_router_predicts(self, inputs, targets):
        router_model_update = self.args['router_train_method']
        if router_model_update == 'prototype_ensemble':
            router_predicts = self.get_router_predicts_by_prototype_ensemble(inputs)
        elif router_model_update == 'prototype':
            router_predicts = self.get_router_predicts_by_prototype(inputs)
        elif router_model_update == 'knn':
            router_predicts = self.get_router_predicts_by_knn(inputs)
        elif router_model_update == 'e_router':
            router_predicts = self.get_router_predicts_by_erouter(inputs)
        else:
            pass
        return router_predicts

    @torch.no_grad()
    def get_router_predicts_by_prototype_ensemble(self, inputs):
        if self._cur_task == 0:
            lora_expert_id = torch.zeros(inputs.size(0), dtype=torch.long).to(self._device)
        else:
            # Calculate cosine similarity between batch_embedding and class features
            frozen_batch_embedding = self.router_model(inputs)  # Dimension is (batch_size, 768)
            lora_batch_embedding = self._network.backbone.forward_features_router(inputs.to(self._device))["features"]
            # lora_batch_embedding = self._network(inputs, adapter_id=0, train=False)["features"]
            batch_embedding = torch.cat((frozen_batch_embedding.detach(), lora_batch_embedding.detach()), dim=1)

            cos_similarity = nn.functional.cosine_similarity(batch_embedding.detach().unsqueeze(1),
                                                             self.router_cls_mean.detach().unsqueeze(0), dim=2)
            router_predicts = torch.topk(cos_similarity, k=1, dim=1, largest=True, sorted=True)[1].squeeze(
                1)  # [bs, topk]
            # router_predicts mapped to lora_expert_id
            lora_expert_id = torch.tensor([self.cls2task[v] for v in router_predicts.cpu().numpy()],
                                          device=self._device)
        return lora_expert_id

    @torch.no_grad()
    def get_router_predicts_by_prototype(self, inputs):
        # Calculate cosine similarity between batch_embedding and class features
        batch_embedding = self.router_model(inputs)  # Dimension is (batch_size, 768)
        cos_similarity = nn.functional.cosine_similarity(batch_embedding.unsqueeze(1),
                                                         self.router_cls_mean.unsqueeze(0), dim=2)
        router_predicts = torch.topk(cos_similarity, k=1, dim=1, largest=True, sorted=True)[1].squeeze(1)  # [bs, topk]
        # router_predicts mapped to lora_expert_id
        lora_expert_id = torch.tensor([self.cls2task[v] for v in router_predicts.cpu().numpy()], device=self._device)
        return lora_expert_id

    @torch.no_grad()
    def get_router_predicts_by_erouter(self, inputs):
        if self._cur_task == 0:
            lora_expert_id = torch.zeros(inputs.size(0), dtype=torch.long).to(self._device)
        else:
            lora_batch_embedding = self._network.backbone.forward_features_router(inputs.to(self._device))["features"]

            cos_similarity = nn.functional.cosine_similarity(lora_batch_embedding.detach().unsqueeze(1),
                                                             self.router_cls_mean.detach().unsqueeze(0), dim=2)
            router_predicts = torch.topk(cos_similarity, k=1, dim=1, largest=True, sorted=True)[1].squeeze(
                1)  # [bs, topk]
            # router_predicts mapped to lora_expert_id
            lora_expert_id = torch.tensor([self.cls2task[v] for v in router_predicts.cpu().numpy()],
                                          device=self._device)
        return lora_expert_id

    @torch.no_grad()
    def get_router_predicts_by_knn(self, inputs):
        if self._cur_task == 0:
            lora_expert_id = torch.zeros(inputs.size(0), dtype=torch.long).to(self._device)
        else:
            batch_embedding = self.router_model(inputs)
            # Calculate cosine similarity between batch_embedding and class features
            cos_similarity = nn.functional.cosine_similarity(batch_embedding.detach().unsqueeze(1),
                                                             self.router_cls_mean.detach().unsqueeze(0), dim=2)
            # Process the true class corresponding to topk_class
            feature_predicts = torch.topk(cos_similarity, k=knn_topk, dim=1, largest=True, sorted=True)[1].squeeze(
                1)  # [bs, topk]
            # feature mapped to cls
            cls_predicts = torch.tensor(
                [self.knn_feature2cls[v] for v in feature_predicts.cpu().numpy().flatten()],
                device=self._device
            ).reshape(feature_predicts.shape)
            mode_cls_predicts = torch.mode(cls_predicts, dim=1).values
            # cls_predicts mapped to lora_expert_id
            lora_expert_id = torch.tensor([self.cls2task[v] for v in mode_cls_predicts.cpu().numpy()],
                                          device=self._device)

        return lora_expert_id

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []

        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device))
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device))
                    )

                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    @torch.no_grad()
    def _extract_vectors_by_sample_mean(self, loader):
        """
        Extract feature vectors used to construct class centers, support 'vit_mean' and 'vit_lora_mean' strategies.
        """
        sample_mean = self.args["sample_mean"]
        self._network.eval()
        self.router_model.eval()

        vectors, targets = [], []

        for _, _inputs, _targets in loader:
            inputs = _inputs.to(self._device)
            _targets_np = _targets.numpy()

            if sample_mean == 'vit_mean':
                # Custom DDP supports transparent call, no need for .module
                extect_vectors = self.router_model(inputs)
            else:
                cur_expert = torch.tensor(self._cur_task).expand(_targets_np.shape).to(self._device)
                extect_vectors = self._network.extract_vector(inputs, cur_expert)[0]

            _vectors = tensor2numpy(extect_vectors)
            vectors.append(_vectors)
            targets.append(_targets_np)

        return np.concatenate(vectors), np.concatenate(targets)
    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers = 16
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers = 16
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []  # [n, feature_dim]
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))
                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            # uniques = np.unique(selected_exemplars, axis=0)
            # print('Unique elements: {}'.format(len(uniques)))
            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset, batch_size=batch_size, shuffle=False, num_workers = 16
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        # Calculate the means of old classes with newly trained network
        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(class_data, class_targets)
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers = 16
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        # Construct exemplars for new classes and calculate the means
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset, batch_size=batch_size, shuffle=False, num_workers = 16
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            # Select
            selected_exemplars = []
            exemplar_vectors = []
            for k in range(1, m + 1):
                S = np.sum(
                    exemplar_vectors, axis=0
                )  # [feature_dim] sum of selected exemplars vectors
                mu_p = (vectors + S) / k  # [n, feature_dim] sum to all vectors
                i = np.argmin(np.sqrt(np.sum((class_mean - mu_p) ** 2, axis=1)))

                selected_exemplars.append(
                    np.array(data[i])
                )  # New object to avoid passing by inference
                exemplar_vectors.append(
                    np.array(vectors[i])
                )  # New object to avoid passing by inference

                vectors = np.delete(
                    vectors, i, axis=0
                )  # Remove it to avoid duplicative selection
                data = np.delete(
                    data, i, axis=0
                )  # Remove it to avoid duplicative selection

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            # Exemplar mean
            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset, batch_size=batch_size, shuffle=False, num_workers = 16
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            _class_means[class_idx, :] = mean

        self._class_means = _class_means

