import torch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture
import numpy as np

class LADA(nn.Module):
    def __init__(self, image_encoder, beta=1.0):
        super().__init__()
        self.image_encoder = image_encoder
        self.beta = beta
        self.register_buffer('prev_lada_features', torch.empty((self.image_encoder.proj.shape[1], 0)))
        self.register_buffer('joint_classifier', torch.empty((0, 0)))

    @torch.no_grad()
    def build_lada(self, data_loader, device, k=16):
        all_features = []
        all_labels = []

        for batch in tqdm(data_loader, desc=f'Building LADA Stage-1', ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(device)
            label = label.to(device)
            
            image_features = self.image_encoder(image)
            
            all_features.append(image_features.cpu())
            all_labels.append(label.cpu())

        all_features = torch.cat(all_features, dim=0)  # shape: (N, D)
        all_labels = torch.cat(all_labels, dim=0)      # shape: (N,)

        normalized_features = F.normalize(all_features, dim=-1)
        features_np = normalized_features.numpy()
        labels_np = all_labels.numpy()

        unique_labels = np.unique(labels_np)

        selected_features = []
        selected_labels = []

        for lbl in tqdm(unique_labels, desc=f'Building LADA Stage-2', ascii=True):
            lbl_indices = np.where(labels_np == lbl)[0]
            lbl_features = features_np[lbl_indices]  # (M, D)
            
            actual_k = min(k, len(lbl_features))
            
            kmeans = KMeans(n_clusters=actual_k, n_init=10, random_state=42).fit(lbl_features)
            cluster_centers = kmeans.cluster_centers_  # (k, D)

            selected_features.append(cluster_centers)
            selected_labels.append(np.full((actual_k,), lbl, dtype=labels_np.dtype))

        selected_features = np.concatenate(selected_features, axis=0)  # (K_total, D)
        selected_labels = np.concatenate(selected_labels, axis=0)      # (K_total,)

        selected_features = torch.from_numpy(selected_features).to(device)
        selected_labels = torch.from_numpy(selected_labels).to(device)

        lada_features = selected_features.t()  # (D, K_total)
        self.curr_lada_features = nn.Parameter(lada_features)

        # Update joint classifier
        selected_labels_one_hot = F.one_hot(selected_labels).float()  # shape: (K_total, num_classes)
        self.curr_classifier = selected_labels_one_hot

        N1, D1 = self.joint_classifier.shape
        N2, D2 = self.curr_classifier.shape
        joint_classifier = torch.zeros((N1 + N2, D1 + D2), dtype=self.curr_classifier.dtype).to(device)
        joint_classifier[:N1, :D1] = self.joint_classifier
        joint_classifier[N1:, D1:] = self.curr_classifier
        self.joint_classifier = joint_classifier

    def forward(self, image_features, lada_features, classifier):
        if lada_features is None:
            lada_features = torch.cat((self.prev_lada_features, self.curr_lada_features), dim=1)
        affinity = image_features @ lada_features
        lada_logits = ((-1) * (self.beta - self.beta * affinity)).exp() @ classifier

        return lada_logits

    @torch.no_grad()
    def update_lada_features(self):
        self.prev_lada_features = torch.cat((self.prev_lada_features, self.curr_lada_features), dim=1)


class DPT(nn.Module):
    def __init__(self, image_encoder, text_encoder):
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder

        self.register_buffer('image_prototypes', torch.empty((0, self.image_encoder.proj.shape[1])))
        self.register_buffer('image_prototypes_covs', torch.empty(0))
        self.register_buffer('image_prototypes_weights', torch.empty(0))
        self.register_buffer('text_prototypes', torch.empty((0, self.text_encoder.out_dim)))

    @torch.no_grad()
    def update_text_prototypes(self, prompts, text_tuner):
        text_features = self.text_encoder(prompts, text_tuner)
        self.text_prototypes = torch.cat((self.text_prototypes.detach(), text_features), dim=0)

    @torch.no_grad()
    def update_image_prototypes(self, data_loader, k, device):
        all_features = []
        all_labels = []

        for batch in tqdm(data_loader, desc=f'Updating image prototypes', ascii=True):
            image = batch[0]
            label = batch[1]
            image = image.to(device)
            label = label.to(device)
            
            image_features = self.image_encoder(image)
            
            all_features.append(image_features)
            all_labels.append(label)
        
        all_features = torch.cat(all_features, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        all_means = []
        all_covariances = []
        all_weights = []

        unique_labels = torch.unique(all_labels)
        sorted_labels, _ = torch.sort(unique_labels)
        for label in sorted_labels:
            label_indices = (all_labels == label)
            label_features = all_features[label_indices]

            gmm = GaussianMixture(n_components=k, covariance_type='spherical', random_state=42)
            label_features_numpy = label_features.cpu().numpy()
            gmm.fit(label_features_numpy)
            means = torch.tensor(gmm.means_, dtype=label_features.dtype).to(device)
            covariances = torch.tensor(gmm.covariances_, dtype=label_features.dtype).to(device)
            weights = torch.tensor(gmm.weights_, dtype=label_features.dtype).to(device)

            all_means.append(means)
            all_covariances.append(covariances)
            all_weights.append(weights)

        image_prototypes = torch.cat(all_means, dim=0)
        self.image_prototypes = torch.cat((self.image_prototypes.detach(), image_prototypes), dim=0)
        image_prototypes_covs = torch.cat(all_covariances, dim=0)
        self.image_prototypes_covs = torch.cat((self.image_prototypes_covs.detach(), image_prototypes_covs), dim=0)
        image_prototypes_ws = torch.cat(all_weights, dim=0)
        self.image_prototypes_weights = torch.cat((self.image_prototypes_weights.detach(), image_prototypes_ws), dim=0)
