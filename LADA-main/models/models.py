import torch
import torch.nn as nn
from torch.nn import functional as F

from clip.model import CLIP

from .lada import DPT, LADA
from .peft_text import Peft_Text, Text_Tuner


class ZeroShotCLIP(nn.Module):
    def __init__(self, clip_model:CLIP):
        super().__init__()
        self.text_encoder = Peft_Text(clip_model)
        self.image_encoder = clip_model.visual
        self.logit_scale = clip_model.logit_scale.exp()
        self.dtype = clip_model.dtype

        self.text_features = None

    def encode_text(self, text):
        text_features = self.text_encoder(text)
        return text_features

    def encode_image(self, image):
        return self.image_encoder(image.to(self.dtype))
    
    @torch.no_grad()
    def init_text_features(self, classnames, templates, tokenize, device):
        if not isinstance(templates, list):
            templates = [templates]
        zeroshot_weights = []
        for classname in classnames:
            classname = classname.replace('_', ' ')
            texts = [template.format(classname) for template in templates]
            texts = tokenize(texts).to(device)
            class_embeddings = self.encode_text(texts)
            class_embeddings /= class_embeddings.norm(dim=-1, keepdim=True)
            class_embedding = class_embeddings.mean(dim=0)
            class_embedding /= class_embedding.norm()
            zeroshot_weights.append(class_embedding)

        zeroshot_weights = torch.stack(zeroshot_weights, dim=1).to(device)
        self.text_features = zeroshot_weights.t()

    def forward(self, image):
        image_features = self.encode_image(image)
        image_features = F.normalize(image_features, dim=-1)
        logit = self.logit_scale * image_features @ self.text_features.t()
        return logit


class PeftModelFromCLIP(nn.Module):
    def __init__(self, cfg, clip_model:CLIP, num_classes, alpha=1.0, beta=1.0):
        super().__init__()

        self.image_encoder = clip_model.visual
        self.text_encoder = Peft_Text(clip_model)
        self.text_tuner = Text_Tuner(cfg, clip_model, num_classes)
        self.dpt = DPT(self.image_encoder, self.text_encoder)
        self.lada = LADA(self.image_encoder, beta=beta)
        self.logit_scale = clip_model.logit_scale.exp().detach()
        self.alpha = alpha
    
    @torch.no_grad()
    def init_prompts(self, classnames, templates, tokenize, device):
        template = templates[0]
        prompts = [template.format(classname.replace('_', ' ')) for classname in classnames]
        # print(f"Prompts: {prompts}")
        prompts = torch.cat([tokenize(p) for p in prompts])
        self.prompts = prompts.to(device)

    def forward(self, image, text_tuner=None, text_features=None, image_prototypes=None, text_prototypes=None, lada_features=None, classifier=None):
        image_features = self.image_encoder(image)
        if image_prototypes is not None:
            image_features = torch.cat([image_prototypes.detach(), image_features], dim=0)
        if text_features is None:
            text_features = self.text_encoder(self.prompts, text_tuner)
        if text_prototypes is not None:
            text_features = torch.cat((text_prototypes.detach(), text_features), dim=0)
        image_features = F.normalize(image_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        text_logits = self.logit_scale * image_features @ text_features.t()

        lada_logits = self.lada(image_features, lada_features=lada_features, classifier=classifier)
        lada_classes = lada_logits.shape[1]
        total_classes = text_logits.shape[1]
        if total_classes == lada_classes:
            return text_logits + self.alpha * lada_logits
        else:
            # only process when evaluation
            max_indices = text_logits.argmax(dim=1)
            mask = max_indices < lada_classes

            padding = torch.full((lada_logits.shape[0], total_classes - lada_classes), -1e6, dtype=lada_logits.dtype, device=lada_logits.device)
            # padding = torch.zeros((lada_logits.shape[0], total_classes - lada_classes), device=lada_logits.device, dtype=lada_logits.dtype)
            lada_logits_padded = torch.cat([lada_logits, padding], dim=1)

            mask = mask.unsqueeze(1).float()
            output = text_logits + mask * self.alpha * lada_logits_padded

            return output
