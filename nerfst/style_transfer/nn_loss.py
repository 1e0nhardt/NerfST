import torch
from torchvision import models, transforms
import torch.nn.functional as F

from nerfst.style_transfer.arf_util import *


class VGG(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.vgg = models.vgg16(pretrained=True).eval()
        self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_feats(self, x, layers=[], supress_assert=True):
        # Layer indexes:
        # Conv1_*: 1,3
        # Conv2_*: 6,8
        # Conv3_*: 11, 13, 15
        # Conv4_*: 18, 20, 22
        # Conv5_*: 25, 27, 29

        if not supress_assert:
            assert x.min() >= 0.0 and x.max() <= 1.0, "input is expected to be an image scaled between 0 and 1"

        x = self.normalize(x)
        final_ix = max(layers)
        outputs = []

        for ix, layer in enumerate(self.vgg.features):
            x = layer(x)
            if ix in layers:
                outputs.append(x)

            if ix == final_ix:
                break

        return outputs


class NNLoss(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        self.vgg = VGG().to(device)

    def forward(
        self,
        outputs,
        styles,
        blocks=[
            2,
        ],
        loss_names=["nn_loss"],  # can also include 'gram_loss', 'content_loss'
        contents=None,
    ):
        blocks.sort()
        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]

        x_feats_all = self.vgg.get_feats(outputs, all_layers)
        with torch.no_grad():
            s_feats_all = self.vgg.get_feats(styles, all_layers)
            if "content_loss" in loss_names:
                content_feats_all = self.vgg.get_feats(contents, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a

        nn_loss = 0.0
        gram_loss = 0.0
        content_loss = 0.0
        for block in blocks:
            layers = block_indexes[block]
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

            if "nn_loss" in loss_names:
                target_feats = feat_replace(x_feats, s_feats)
                nn_loss += cos_loss(x_feats, target_feats)

            if "gram_loss" in loss_names:
                gram_loss += torch.mean((gram_matrix(x_feats) - gram_matrix(s_feats)) ** 2)

            if "content_loss" in loss_names:
                content_feats = torch.cat([content_feats_all[ix_map[ix]] for ix in layers], 1)
                content_loss += torch.mean((content_feats - x_feats) ** 2)

        return nn_loss, gram_loss, content_loss

    def get_style_nn(
        self,
        outputs,
        styles,
        blocks=[
            2,
        ],
    ):
        blocks.sort()
        block_indexes = [[1, 3], [6, 8], [11, 13, 15], [18, 20, 22], [25, 27, 29]]

        all_layers = []
        for block in blocks:
            all_layers += block_indexes[block]

        x_feats_all = self.vgg.get_feats(outputs, all_layers)
        with torch.no_grad():
            s_feats_all = self.vgg.get_feats(styles, all_layers)

        ix_map = {}
        for a, b in enumerate(all_layers):
            ix_map[b] = a

        trgt_feats = []
        for block in blocks:
            layers = block_indexes[block]
            x_feats = torch.cat([x_feats_all[ix_map[ix]] for ix in layers], 1)
            s_feats = torch.cat([s_feats_all[ix_map[ix]] for ix in layers], 1)

            _, _, h, w = s_feats.size()
            styles_resample = F.interpolate(styles, (h, w), mode="bilinear")
            # ic(x_feats.shape, s_feats.shape, styles_resample.shape)

            feats = guided_feat_replace(x_feats, s_feats, styles_resample)
            trgt_feats.append(feats)

        return trgt_feats