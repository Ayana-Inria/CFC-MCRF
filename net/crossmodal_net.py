"""Cross-modal attention network"""

from types import SimpleNamespace


def get_default_config():
    cfg = SimpleNamespace()
    cfg.KV_size = 96
    cfg.expand_ratio = 4
    cfg.patch_sizes = [16, 8]
    cfg.transformer = {
        "num_heads": 4,
        "attention_dropout_rate": 0.0,
        "dropout_rate": 0.0,
        "embeddings_dropout_rate": 0.0,
    }
    return cfg


config = get_default_config()

import copy
import math
import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair


class Channel_Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, patchsize, img_size, in_channels):
        super().__init__()
        img_size = _pair(img_size)
        patch_size = _pair(patchsize)
        n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
        
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=in_channels,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, in_channels))
        self.dropout = Dropout(config.transformer["embeddings_dropout_rate"])

    def forward(self, x):
        if x is None:
            return None
        x = self.patch_embeddings(x)  # (B, hidden. n_patches^(1/2), n_patches^(1/2))
        x = x.flatten(2)
        x = x.transpose(-1, -2)  # (B, n_patches, hidden)
        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings


class Reconstruct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, scale_factor):
        super(Reconstruct, self).__init__()
        if kernel_size == 3:
            padding = 1
        else:
            padding = 0
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding)
        self.norm = nn.BatchNorm2d(out_channels)
        self.activation = nn.ReLU(inplace=True)
        self.scale_factor = scale_factor

    def forward(self, x):
        if x is None:
            return None

        B, n_patch, hidden = x.size()  # reshape from (B, n_patch, hidden) to (B, h, w, hidden)
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = x.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = nn.Upsample(scale_factor=self.scale_factor)(x)

        out = self.conv(x)
        out = self.norm(out)
        out = self.activation(out)
        return out


class Attention_org(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_org, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        for _ in range(config.transformer["num_heads"]):
            query1 = nn.Linear(channel_num[0] // 4, channel_num[0] // 4, bias=False)
            query2 = nn.Linear(channel_num[1] // 4, channel_num[1] // 4, bias=False)
            key = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            value = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))
        
        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, emb1, emb2, emb_all):
        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_K_list = []
        multi_head_V_list = []
        
        if emb1 is not None:
            Q0, Q1, Q2, Q3 = emb1.split(emb1.shape[2] // 4, dim=2)
            multi_head_Q1_list.append(self.query1[0](Q0))
            multi_head_Q1_list.append(self.query1[1](Q1))
            multi_head_Q1_list.append(self.query1[2](Q2))
            multi_head_Q1_list.append(self.query1[3](Q3))
        
        if emb2 is not None:
            Q0, Q1, Q2, Q3 = emb2.split(emb2.shape[2] // 4, dim=2)
            multi_head_Q2_list.append(self.query2[0](Q0))
            multi_head_Q2_list.append(self.query2[1](Q1))
            multi_head_Q2_list.append(self.query2[2](Q2))
            multi_head_Q2_list.append(self.query2[3](Q3))
        
        # Q0 = torch.cat([emb_all[:, :, 0:16], emb_all[:, :, 64:96], emb_all[:, :, 192:256], emb_all[:, :, 448:576]],
        #                dim=2)
        # Q1 = torch.cat([emb_all[:, :, 16:32], emb_all[:, :, 96:128], emb_all[:, :, 256:320], emb_all[:, :, 576:704]],
        #                dim=2)
        # Q2 = torch.cat([emb_all[:, :, 32:48], emb_all[:, :, 128:160], emb_all[:, :, 320:384], emb_all[:, :, 704:832]],
        #                dim=2)
        # Q3 = torch.cat([emb_all[:, :, 48:64], emb_all[:, :, 160:192], emb_all[:, :, 384:448], emb_all[:, :, 832:960]],
        #                dim=2)

        Q0 = torch.cat([emb_all[:, :, 0:6], emb_all[:, :, 6:12], emb_all[:, :, 12:18], emb_all[:, :, 18:24]], dim=2)
        Q1 = torch.cat([emb_all[:, :, 24:30], emb_all[:, :, 30:36], emb_all[:, :, 36:42], emb_all[:, :, 42:48]], dim=2)
        Q2 = torch.cat([emb_all[:, :, 48:54], emb_all[:, :, 54:60], emb_all[:, :, 60:66], emb_all[:, :, 66:72]], dim=2)
        Q3 = torch.cat([emb_all[:, :, 72:78], emb_all[:, :, 78:84], emb_all[:, :, 84:90], emb_all[:, :, 90:96]], dim=2)

        multi_head_K_list.append(self.key[0](Q0))
        multi_head_K_list.append(self.key[0](Q1))
        multi_head_K_list.append(self.key[0](Q2))
        multi_head_K_list.append(self.key[0](Q3))
        # Q0 = torch.cat([emb_all[:, :, 0:16], emb_all[:, :, 64:96], emb_all[:, :, 192:256], emb_all[:, :, 448:576]],
        #                dim=2)
        # Q1 = torch.cat([emb_all[:, :, 16:32], emb_all[:, :, 96:128], emb_all[:, :, 256:320], emb_all[:, :, 576:704]],
        #                dim=2)
        # Q2 = torch.cat([emb_all[:, :, 32:48], emb_all[:, :, 128:160], emb_all[:, :, 320:384], emb_all[:, :, 704:832]],
        #                dim=2)
        # Q3 = torch.cat([emb_all[:, :, 48:64], emb_all[:, :, 160:192], emb_all[:, :, 384:448], emb_all[:, :, 832:960]],
        #                dim=2)
        Q0 = torch.cat([emb_all[:, :, 0:6], emb_all[:, :, 6:12], emb_all[:, :, 12:18], emb_all[:, :, 18:24]], dim=2)
        Q1 = torch.cat([emb_all[:, :, 24:30], emb_all[:, :, 30:36], emb_all[:, :, 36:42], emb_all[:, :, 42:48]], dim=2)
        Q2 = torch.cat([emb_all[:, :, 48:54], emb_all[:, :, 54:60], emb_all[:, :, 60:66], emb_all[:, :, 66:72]], dim=2)
        Q3 = torch.cat([emb_all[:, :, 72:78], emb_all[:, :, 78:84], emb_all[:, :, 84:90], emb_all[:, :, 90:96]], dim=2)

        multi_head_V_list.append(self.value[0](Q0))
        multi_head_V_list.append(self.value[0](Q1))
        multi_head_V_list.append(self.value[0](Q2))
        multi_head_V_list.append(self.value[0](Q3))

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1) if emb1 is not None else None
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1) if emb2 is not None else None
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2) if emb1 is not None else None
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2) if emb2 is not None else None

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_K) if emb1 is not None else None
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_K) if emb2 is not None else None

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size) if emb1 is not None else None
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size) if emb2 is not None else None
        
        attention_probs1 = self.softmax(self.psi(attention_scores1)) if emb1 is not None else None
        attention_probs2 = self.softmax(self.psi(attention_scores2)) if emb2 is not None else None
        # print(attention_probs4.size())

        if self.vis:
            weights = []
            weights.append(attention_probs1.mean(1))
            weights.append(attention_probs2.mean(1))
        else:
            weights = None

        attention_probs1 = self.attn_dropout(attention_probs1) if emb1 is not None else None
        attention_probs2 = self.attn_dropout(attention_probs2) if emb2 is not None else None

        multi_head_V = multi_head_V.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V) if emb1 is not None else None
        context_layer2 = torch.matmul(attention_probs2, multi_head_V) if emb2 is not None else None

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous() if emb1 is not None else None
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous() if emb2 is not None else None
        context_layer1 = context_layer1.view(context_layer1.shape[0], context_layer1.shape[1],
                                             context_layer1.shape[2] * 4)
        context_layer2 = context_layer2.view(context_layer2.shape[0], context_layer2.shape[1],
                                             context_layer2.shape[2] * 4)

        O1 = self.out1(context_layer1) if emb1 is not None else None
        O2 = self.out2(context_layer2) if emb2 is not None else None
        O1 = self.proj_dropout(O1) if emb1 is not None else None
        O2 = self.proj_dropout(O2) if emb2 is not None else None
        return O1, O2, weights


class Attention_org_cross(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Attention_org_cross, self).__init__()
        self.vis = vis
        self.KV_size = config.KV_size
        self.channel_num = channel_num
        self.num_attention_heads = config.transformer["num_heads"]

        self.query1 = nn.ModuleList()
        self.query2 = nn.ModuleList()
        self.key = nn.ModuleList()
        self.value = nn.ModuleList()

        self.queryd1 = nn.ModuleList()
        self.queryd2 = nn.ModuleList()
        self.keyd = nn.ModuleList()
        self.valued = nn.ModuleList()

        for _ in range(config.transformer["num_heads"]):
            query1 = nn.Linear(channel_num[0] // 4, channel_num[0] // 4, bias=False)
            query2 = nn.Linear(channel_num[1] // 4, channel_num[1] // 4, bias=False)
            key = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            value = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            self.query1.append(copy.deepcopy(query1))
            self.query2.append(copy.deepcopy(query2))
            self.key.append(copy.deepcopy(key))
            self.value.append(copy.deepcopy(value))

            queryd1 = nn.Linear(channel_num[0] // 4, channel_num[0] // 4, bias=False)
            queryd2 = nn.Linear(channel_num[1] // 4, channel_num[1] // 4, bias=False)
            keyd = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            valued = nn.Linear(self.KV_size // 4, self.KV_size // 4, bias=False)
            self.queryd1.append(copy.deepcopy(queryd1))
            self.queryd2.append(copy.deepcopy(queryd2))
            self.keyd.append(copy.deepcopy(keyd))
            self.valued.append(copy.deepcopy(valued))

        self.psi = nn.InstanceNorm2d(self.num_attention_heads)
        self.psid = nn.InstanceNorm2d(self.num_attention_heads)
        self.softmax = Softmax(dim=3)
        self.out1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.out2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.outd1 = nn.Linear(channel_num[0], channel_num[0], bias=False)
        self.outd2 = nn.Linear(channel_num[1], channel_num[1], bias=False)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

    def forward(self, emb1, emb2, emb_all, embd1, embd2, emb_alld):
        multi_head_Q1_list = []
        multi_head_Q2_list = []
        multi_head_K_list = []
        multi_head_V_list = []

        multi_head_Qd1_list = []
        multi_head_Qd2_list = []
        multi_head_Kd_list = []
        multi_head_Vd_list = []

        if emb1 is not None:
            Q0, Q1, Q2, Q3 = emb1.split(emb1.shape[2] // 4, dim=2)
            multi_head_Q1_list.append(self.query1[0](Q0))
            multi_head_Q1_list.append(self.query1[1](Q1))
            multi_head_Q1_list.append(self.query1[2](Q2))
            multi_head_Q1_list.append(self.query1[3](Q3))
        if emb2 is not None:
            Q0, Q1, Q2, Q3 = emb2.split(emb2.shape[2] // 4, dim=2)
            multi_head_Q2_list.append(self.query2[0](Q0))
            multi_head_Q2_list.append(self.query2[1](Q1))
            multi_head_Q2_list.append(self.query2[2](Q2))
            multi_head_Q2_list.append(self.query2[3](Q3))
        # Q0, Q1, Q2, Q3 = emb_all.split(emb_all.shape[2] // 4, dim=2)
        # Q0 = torch.cat([emb_all[:, :, 0:16], emb_all[:, :, 64:96], emb_all[:, :, 192:256], emb_all[:, :, 448:576]],
        #                dim=2)
        # Q1 = torch.cat([emb_all[:, :, 16:32], emb_all[:, :, 96:128], emb_all[:, :, 256:320], emb_all[:, :, 576:704]],
        #                dim=2)
        # Q2 = torch.cat([emb_all[:, :, 32:48], emb_all[:, :, 128:160], emb_all[:, :, 320:384], emb_all[:, :, 704:832]],
        #                dim=2)
        # Q3 = torch.cat([emb_all[:, :, 48:64], emb_all[:, :, 160:192], emb_all[:, :, 384:448], emb_all[:, :, 832:960]],
        #                dim=2)
        Q0 = torch.cat([emb_all[:, :, 0:6], emb_all[:, :, 6:12], emb_all[:, :, 12:18], emb_all[:, :, 18:24]], dim=2)
        Q1 = torch.cat([emb_all[:, :, 24:30], emb_all[:, :, 30:36], emb_all[:, :, 36:42], emb_all[:, :, 42:48]], dim=2)
        Q2 = torch.cat([emb_all[:, :, 48:54], emb_all[:, :, 54:60], emb_all[:, :, 60:66], emb_all[:, :, 66:72]], dim=2)
        Q3 = torch.cat([emb_all[:, :, 72:78], emb_all[:, :, 78:84], emb_all[:, :, 84:90], emb_all[:, :, 90:96]], dim=2)
               
        multi_head_K_list.append(self.key[0](Q0))
        multi_head_K_list.append(self.key[0](Q1))
        multi_head_K_list.append(self.key[0](Q2))
        multi_head_K_list.append(self.key[0](Q3))
        # Q0, Q1, Q2, Q3 = emb_all.split(emb_all.shape[2] // 4, dim=2)
        # Q0 = torch.cat([emb_all[:, :, 0:16], emb_all[:, :, 64:96], emb_all[:, :, 192:256], emb_all[:, :, 448:576]],
        #                dim=2)
        # Q1 = torch.cat([emb_all[:, :, 16:32], emb_all[:, :, 96:128], emb_all[:, :, 256:320], emb_all[:, :, 576:704]],
        #                dim=2)
        # Q2 = torch.cat([emb_all[:, :, 32:48], emb_all[:, :, 128:160], emb_all[:, :, 320:384], emb_all[:, :, 704:832]],
        #                dim=2)
        # Q3 = torch.cat([emb_all[:, :, 48:64], emb_all[:, :, 160:192], emb_all[:, :, 384:448], emb_all[:, :, 832:960]],
        #                dim=2)
        Q0 = torch.cat([emb_all[:, :, 0:6], emb_all[:, :, 6:12], emb_all[:, :, 12:18], emb_all[:, :, 18:24]], dim=2)
        Q1 = torch.cat([emb_all[:, :, 24:30], emb_all[:, :, 30:36], emb_all[:, :, 36:42], emb_all[:, :, 42:48]], dim=2)
        Q2 = torch.cat([emb_all[:, :, 48:54], emb_all[:, :, 54:60], emb_all[:, :, 60:66], emb_all[:, :, 66:72]], dim=2)
        Q3 = torch.cat([emb_all[:, :, 72:78], emb_all[:, :, 78:84], emb_all[:, :, 84:90], emb_all[:, :, 90:96]], dim=2)


        multi_head_V_list.append(self.value[0](Q0))
        multi_head_V_list.append(self.value[0](Q1))
        multi_head_V_list.append(self.value[0](Q2))
        multi_head_V_list.append(self.value[0](Q3))

        if embd1 is not None:
            Q0, Q1, Q2, Q3 = embd1.split(embd1.shape[2] // 4, dim=2)
            multi_head_Qd1_list.append(self.queryd1[0](Q0))
            multi_head_Qd1_list.append(self.queryd1[1](Q1))
            multi_head_Qd1_list.append(self.queryd1[2](Q2))
            multi_head_Qd1_list.append(self.queryd1[3](Q3))
        if embd2 is not None:
            Q0, Q1, Q2, Q3 = embd2.split(embd2.shape[2] // 4, dim=2)
            multi_head_Qd2_list.append(self.queryd2[0](Q0))
            multi_head_Qd2_list.append(self.queryd2[1](Q1))
            multi_head_Qd2_list.append(self.queryd2[2](Q2))
            multi_head_Qd2_list.append(self.queryd2[3](Q3))
        # Q0 = torch.cat([emb_alld[:, :, 0:16], emb_alld[:, :, 64:96], emb_alld[:, :, 192:256], emb_alld[:, :, 448:576]],
        #                dim=2)
        # Q1 = torch.cat(
        #     [emb_alld[:, :, 16:32], emb_alld[:, :, 96:128], emb_alld[:, :, 256:320], emb_alld[:, :, 576:704]], dim=2)
        # Q2 = torch.cat(
        #     [emb_alld[:, :, 32:48], emb_alld[:, :, 128:160], emb_alld[:, :, 320:384], emb_alld[:, :, 704:832]], dim=2)
        # Q3 = torch.cat(
        #     [emb_alld[:, :, 48:64], emb_alld[:, :, 160:192], emb_alld[:, :, 384:448], emb_alld[:, :, 832:960]], dim=2)

        Q0 = torch.cat([emb_all[:, :, 0:6], emb_all[:, :, 6:12], emb_all[:, :, 12:18], emb_all[:, :, 18:24]], dim=2)
        Q1 = torch.cat([emb_all[:, :, 24:30], emb_all[:, :, 30:36], emb_all[:, :, 36:42], emb_all[:, :, 42:48]], dim=2)
        Q2 = torch.cat([emb_all[:, :, 48:54], emb_all[:, :, 54:60], emb_all[:, :, 60:66], emb_all[:, :, 66:72]], dim=2)
        Q3 = torch.cat([emb_all[:, :, 72:78], emb_all[:, :, 78:84], emb_all[:, :, 84:90], emb_all[:, :, 90:96]], dim=2)
        
        
        multi_head_Kd_list.append(self.keyd[0](Q0))
        multi_head_Kd_list.append(self.keyd[0](Q1))
        multi_head_Kd_list.append(self.keyd[0](Q2))
        multi_head_Kd_list.append(self.keyd[0](Q3))
        # Q0 = torch.cat([emb_alld[:, :, 0:16], emb_alld[:, :, 64:96], emb_alld[:, :, 192:256], emb_alld[:, :, 448:576]],
        #                dim=2)
        # Q1 = torch.cat(
        #     [emb_alld[:, :, 16:32], emb_alld[:, :, 96:128], emb_alld[:, :, 256:320], emb_alld[:, :, 576:704]], dim=2)
        # Q2 = torch.cat(
        #     [emb_alld[:, :, 32:48], emb_alld[:, :, 128:160], emb_alld[:, :, 320:384], emb_alld[:, :, 704:832]], dim=2)
        # Q3 = torch.cat(
        #     [emb_alld[:, :, 48:64], emb_alld[:, :, 160:192], emb_alld[:, :, 384:448], emb_alld[:, :, 832:960]], dim=2)
        Q0 = torch.cat([emb_all[:, :, 0:6], emb_all[:, :, 6:12], emb_all[:, :, 12:18], emb_all[:, :, 18:24]], dim=2)
        Q1 = torch.cat([emb_all[:, :, 24:30], emb_all[:, :, 30:36], emb_all[:, :, 36:42], emb_all[:, :, 42:48]], dim=2)
        Q2 = torch.cat([emb_all[:, :, 48:54], emb_all[:, :, 54:60], emb_all[:, :, 60:66], emb_all[:, :, 66:72]], dim=2)
        Q3 = torch.cat([emb_all[:, :, 72:78], emb_all[:, :, 78:84], emb_all[:, :, 84:90], emb_all[:, :, 90:96]], dim=2)
        
        
        multi_head_Vd_list.append(self.valued[0](Q0))
        multi_head_Vd_list.append(self.valued[0](Q1))
        multi_head_Vd_list.append(self.valued[0](Q2))
        multi_head_Vd_list.append(self.valued[0](Q3))

        multi_head_Q1 = torch.stack(multi_head_Q1_list, dim=1)
        multi_head_Q2 = torch.stack(multi_head_Q2_list, dim=1)
        multi_head_K = torch.stack(multi_head_K_list, dim=1)
        multi_head_V = torch.stack(multi_head_V_list, dim=1)

        multi_head_Qd1 = torch.stack(multi_head_Qd1_list, dim=1)
        multi_head_Qd2 = torch.stack(multi_head_Qd2_list, dim=1)
        multi_head_Kd = torch.stack(multi_head_Kd_list, dim=1)
        multi_head_Vd = torch.stack(multi_head_Vd_list, dim=1)

        multi_head_Q1 = multi_head_Q1.transpose(-1, -2)
        multi_head_Q2 = multi_head_Q2.transpose(-1, -2)

        multi_head_Qd1 = multi_head_Qd1.transpose(-1, -2)
        multi_head_Qd2 = multi_head_Qd2.transpose(-1, -2)

        attention_scores1 = torch.matmul(multi_head_Q1, multi_head_Kd)
        attention_scores2 = torch.matmul(multi_head_Q2, multi_head_Kd)

        attention_scoresd1 = torch.matmul(multi_head_Qd1, multi_head_K)
        attention_scoresd2 = torch.matmul(multi_head_Qd2, multi_head_K)

        attention_scores1 = attention_scores1 / math.sqrt(self.KV_size)
        attention_scores2 = attention_scores2 / math.sqrt(self.KV_size)

        attention_scoresd1 = attention_scoresd1 / math.sqrt(self.KV_size)
        attention_scoresd2 = attention_scoresd2 / math.sqrt(self.KV_size)

        attention_probs1 = self.softmax(self.psi(attention_scores1))
        attention_probs2 = self.softmax(self.psi(attention_scores2))

        attention_probsd1 = self.softmax(self.psid(attention_scoresd1))
        attention_probsd2 = self.softmax(self.psid(attention_scoresd2))

        if self.vis:
            weights = []
            weights.append(attention_probs1.mean(1))
            weights.append(attention_probs2.mean(1))
        else:
            weights = None

        attention_probs1 = self.attn_dropout(attention_probs1)
        attention_probs2 = self.attn_dropout(attention_probs2)

        attention_probsd1 = self.attn_dropout(attention_probsd1)
        attention_probsd2 = self.attn_dropout(attention_probsd2)

        multi_head_V = multi_head_V.transpose(-1, -2)
        multi_head_Vd = multi_head_Vd.transpose(-1, -2)
        context_layer1 = torch.matmul(attention_probs1, multi_head_V)
        context_layer2 = torch.matmul(attention_probs2, multi_head_V)

        context_layerd1 = torch.matmul(attention_probsd1, multi_head_Vd)
        context_layerd2 = torch.matmul(attention_probsd2, multi_head_Vd)

        context_layer1 = context_layer1.permute(0, 3, 2, 1).contiguous()
        context_layer2 = context_layer2.permute(0, 3, 2, 1).contiguous()

        context_layerd1 = context_layerd1.permute(0, 3, 2, 1).contiguous()
        context_layerd2 = context_layerd2.permute(0, 3, 2, 1).contiguous()

        context_layer1 = context_layer1.view(context_layer1.shape[0], context_layer1.shape[1],
                                             context_layer1.shape[2] * 4)
        context_layer2 = context_layer2.view(context_layer2.shape[0], context_layer2.shape[1],
                                             context_layer2.shape[2] * 4)
        context_layerd1 = context_layerd1.view(context_layerd1.shape[0], context_layerd1.shape[1],
                                               context_layerd1.shape[2] * 4)
        context_layerd2 = context_layerd2.view(context_layerd2.shape[0], context_layerd2.shape[1],
                                               context_layerd2.shape[2] * 4)

        O1 = self.out1(context_layer1)
        O2 = self.out2(context_layer2)
        Od1 = self.outd1(context_layerd1)
        Od2 = self.outd2(context_layerd2)
        O1 = self.proj_dropout(O1)
        O2 = self.proj_dropout(O2)
        Od1 = self.proj_dropout(Od1)
        Od2 = self.proj_dropout(Od2)
        return O1, O2, Od1, Od2, weights

class Mlp(nn.Module):
    def __init__(self, config, in_channel, mlp_channel):
        super(Mlp, self).__init__()
        self.fc1 = nn.Linear(in_channel, mlp_channel)
        self.fc2 = nn.Linear(mlp_channel, in_channel)
        self.act_fn = nn.GELU()
        self.dropout = Dropout(config.transformer["dropout_rate"])
        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Block_ViT(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.attn_norm = LayerNorm(config.KV_size, eps=1e-6)
        self.channel_attn = Attention_org(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.ffn1 = Mlp(config, channel_num[0], channel_num[0] * expand_ratio)
        self.ffn2 = Mlp(config, channel_num[1], channel_num[1] * expand_ratio)

    def forward(self, emb1, emb2):
        embcat = []
        org1 = emb1
        org2 = emb2
        for i in range(2):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

        emb_all = torch.cat(embcat, dim=2)
        cx1 = self.attn_norm1(emb1) if emb1 is not None else None
        cx2 = self.attn_norm2(emb2) if emb2 is not None else None
        emb_all = self.attn_norm(emb_all)
        cx1, cx2, weights = self.channel_attn(cx1, cx2, emb_all)
        cx1 = org1 + cx1 if emb1 is not None else None
        cx2 = org2 + cx2 if emb2 is not None else None


        org1 = cx1
        org2 = cx2
        x1 = self.ffn_norm1(cx1) if emb1 is not None else None
        x2 = self.ffn_norm2(cx2) if emb2 is not None else None
        x1 = self.ffn1(x1) if emb1 is not None else None
        x2 = self.ffn2(x2) if emb2 is not None else None
        x1 = x1 + org1 if emb1 is not None else None
        x2 = x2 + org2 if emb2 is not None else None

        return x1, x2, weights


class Encoder(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, weights = layer_block(emb1, emb2)
            if self.vis:
                attn_weights.append(weights)
        emb1 = self.encoder_norm1(emb1) if emb1 is not None else None
        emb2 = self.encoder_norm2(emb2) if emb2 is not None else None
        return emb1, emb2, attn_weights


class ChannelTransformer(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128], patchSize=[32, 16]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2,
                                               in_channels=channel_num[1])

        self.encoder = Encoder(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,
                                         scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,
                                         scale_factor=(self.patchSize_2, self.patchSize_2))


    def forward(self, en1, en2):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)

        encoded1, encoded2, attn_weights = self.encoder(emb1, emb2)  # (B, n_patch, hidden)
        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None

        x1 = x1 + en1 if en1 is not None else None
        x2 = x2 + en2 if en2 is not None else None

        return x1, x2, attn_weights


class Block_ViT_cross(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Block_ViT_cross, self).__init__()
        expand_ratio = config.expand_ratio
        self.attn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_norm2 = LayerNorm(channel_num[1], eps=1e-6)

        self.attn_normd1 = LayerNorm(channel_num[0], eps=1e-6)
        self.attn_normd2 = LayerNorm(channel_num[1], eps=1e-6)

        self.attn_normx = LayerNorm(config.KV_size, eps=1e-6)
        self.attn_normy = LayerNorm(config.KV_size, eps=1e-6)
        self.channel_attn = Attention_org_cross(config, vis, channel_num)

        self.ffn_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.ffn_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        self.ffn_normd1 = LayerNorm(channel_num[0], eps=1e-6)
        self.ffn_normd2 = LayerNorm(channel_num[1], eps=1e-6)

        self.ffn1 = Mlp(config, channel_num[0], channel_num[0] * expand_ratio)
        self.ffn2 = Mlp(config, channel_num[1], channel_num[1] * expand_ratio)
        self.ffnd1 = Mlp(config, channel_num[0], channel_num[0] * expand_ratio)
        self.ffnd2 = Mlp(config, channel_num[1], channel_num[1] * expand_ratio)

    def forward(self, emb1, emb2, embd1, embd2):
        embcat = []
        embcatd = []
        org1 = emb1
        org2 = emb2
        orgd1 = embd1
        orgd2 = embd2
        
        for i in range(2):
            var_name = "emb" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcat.append(tmp_var)

            var_name = "embd" + str(i + 1)
            tmp_var = locals()[var_name]
            if tmp_var is not None:
                embcatd.append(tmp_var)

        emb_all = torch.cat(embcat, dim=2)
        emb_alld = torch.cat(embcatd, dim=2)
        cx1 = self.attn_norm1(emb1)
        cx2 = self.attn_norm2(emb2)

        cy1 = self.attn_normd1(embd1)
        cy2 = self.attn_normd2(embd2)

        emb_all = self.attn_normx(emb_all)
        emb_alld = self.attn_normy(emb_alld)
        cx1, cx2, cy1, cy2, weights = self.channel_attn(cx1, cx2, emb_all, cy1, cy2, emb_alld)
        cx1 = org1 + cx1
        cx2 = org2 + cx2

        cy1 = orgd1 + cy1
        cy2 = orgd2 + cy2

        org1 = cx1
        org2 = cx2

        orgd1 = cy1
        orgd2 = cy2
        x1 = self.ffn_norm1(cx1)
        x2 = self.ffn_norm2(cx2)

        y1 = self.ffn_normd1(cy1)
        y2 = self.ffn_normd2(cy2)
        x1 = self.ffn1(x1)
        x2 = self.ffn2(x2)

        y1 = self.ffnd1(y1)
        y2 = self.ffnd2(y2)
        x1 = x1 + org1
        x2 = x2 + org2

        y1 = y1 + orgd1
        y2 = y2 + orgd2

        return x1, x2, y1, y2, weights


class Encoder_cross(nn.Module):
    def __init__(self, config, vis, channel_num):
        super(Encoder_cross, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm1 = LayerNorm(channel_num[0], eps=1e-6)
        self.encoder_norm2 = LayerNorm(channel_num[1], eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block_ViT_cross(config, vis, channel_num)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, emb1, emb2, embd1, embd2):
        attn_weights = []
        for layer_block in self.layer:
            emb1, emb2, embd1, embd2, weights = layer_block(emb1, emb2, embd1, embd2)
            if self.vis:
                attn_weights.append(weights)

        emb1 = self.encoder_norm1(emb1)
        emb2 = self.encoder_norm2(emb2)
        embd1 = self.encoder_norm1(embd1)
        embd2 = self.encoder_norm2(embd2)
        return emb1, emb2, embd1, embd2, attn_weights


class ChannelTransformer_cross(nn.Module):
    def __init__(self, config, vis, img_size, channel_num=[64, 128], patchSize=[32, 16]):
        super().__init__()

        self.patchSize_1 = patchSize[0]
        self.patchSize_2 = patchSize[1]
        self.embeddings_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddings_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2,
                                               in_channels=channel_num[1])

        self.embeddingsd_1 = Channel_Embeddings(config, self.patchSize_1, img_size=img_size, in_channels=channel_num[0])
        self.embeddingsd_2 = Channel_Embeddings(config, self.patchSize_2, img_size=img_size // 2,
                                                in_channels=channel_num[1])

        self.encoder = Encoder_cross(config, vis, channel_num)

        self.reconstruct_1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,
                                         scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,
                                         scale_factor=(self.patchSize_2, self.patchSize_2))
        self.reconstruct_d1 = Reconstruct(channel_num[0], channel_num[0], kernel_size=1,
                                          scale_factor=(self.patchSize_1, self.patchSize_1))
        self.reconstruct_d2 = Reconstruct(channel_num[1], channel_num[1], kernel_size=1,
                                          scale_factor=(self.patchSize_2, self.patchSize_2))

    def forward(self, en1, en2, end1, end2):
        emb1 = self.embeddings_1(en1)
        emb2 = self.embeddings_2(en2)

        embd1 = self.embeddingsd_1(end1)
        embd2 = self.embeddingsd_2(end2)

        encoded1, encoded2, encodedd1, encodedd2, attn_weights = self.encoder(
            emb1, emb2, embd1,
            embd2)  # (B, n_patch, hidden)
        x1 = self.reconstruct_1(encoded1) if en1 is not None else None
        x2 = self.reconstruct_2(encoded2) if en2 is not None else None
        y1 = self.reconstruct_d1(encodedd1) if en1 is not None else None
        y2 = self.reconstruct_d2(encodedd2) if en2 is not None else None

        x1 = x1 + en1 if en1 is not None else None
        x2 = x2 + en2 if en2 is not None else None
        y1 = y1 + end1 if end1 is not None else None
        y2 = y2 + end2 if end2 is not None else None

        return x1, x2, y1, y2, attn_weights

class CMFNet(nn.Module):
    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Linear):
            torch.nn.init.kaiming_normal(m.weight.data)

    def attention(self, num_channels):
        pool_attention = nn.AdaptiveAvgPool2d(1)
        conv_attention = nn.Conv2d(num_channels, num_channels, kernel_size=1)
        activate = nn.Sigmoid()

        return nn.Sequential(pool_attention, conv_attention, activate)

    def __init__(self, in_channels=233, out_channels=6):
        super(CMFNet, self).__init__()
        self.pool = nn.MaxPool2d(2, return_indices=True)
        self.unpool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)#nn.MaxUnpool2d(2)
        self.pool3 = nn.MaxPool2d(3, return_indices=True)
        self.unpool3 = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True)#nn.MaxUnpool2d(3)
        self.poolHSI = nn.MaxPool3d((3, 1, 1))
        self.up6 = nn.Upsample(scale_factor=6, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=3, mode='bilinear', align_corners=True)

        ##### PAN ENCODER ####
        self.conv1_1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv1_1_bn = nn.BatchNorm2d(32)
        self.conv1_2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv1_2_bn = nn.BatchNorm2d(32)

        self.conv2_1 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_1_bn = nn.BatchNorm2d(64)
        self.conv2_2 = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_2_bn = nn.BatchNorm2d(64)

        self.conv5_1 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5_1_bn = nn.BatchNorm2d(128)
        self.conv5_2 = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5_2_bn = nn.BatchNorm2d(128)

        ##### HSI ENCODER ####
        self.conv1_1_d = nn.Conv2d(in_channels, 32, kernel_size=3, padding=1)
        self.conv1_1_d_bn = nn.BatchNorm2d(32)
        self.conv1_2_d = nn.Conv2d(32, 32, 3, padding=1)
        self.conv1_2_d_bn = nn.BatchNorm2d(32)

        self.conv2_1_d = nn.Conv2d(32, 64, 3, padding=1)
        self.conv2_1_d_bn = nn.BatchNorm2d(64)
        self.conv2_2_d = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_2_d_bn = nn.BatchNorm2d(64)

        self.conv5_1_d = nn.Conv2d(64, 128, 3, padding=1)
        self.conv5_1_d_bn = nn.BatchNorm2d(128)
        self.conv5_2_d = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5_2_d_bn = nn.BatchNorm2d(128)


        ##### FUSION MODULE ####
        self.attention_5 = self.attention(128)
        self.attention_5_d = self.attention(128)

        ##### SKIP MODULE: UCTransNet ####
        vis = True
        config_vit = config
        self.mtc = ChannelTransformer_cross(config_vit, vis, 384,
                                      channel_num=[32, 64],
                                      patchSize=config_vit.patch_sizes)
        self.mtc1 = ChannelTransformer(config_vit, vis, 384,
                                      channel_num=[32, 64],
                                      patchSize=config_vit.patch_sizes)
        #### DECODER  ####
        self.conv5_2_D = nn.Conv2d(128, 128, 3, padding=1)
        self.conv5_2_D_bn = nn.BatchNorm2d(128)
        self.conv5_1_D = nn.Conv2d(128, 64, 3, padding=1)
        self.conv5_1_D_bn = nn.BatchNorm2d(64)

        self.conv2_2_D = nn.Conv2d(64, 64, 3, padding=1)
        self.conv2_2_D_bn = nn.BatchNorm2d(64)
        self.conv2_1_D = nn.Conv2d(64, 32, 3, padding=1)
        self.conv2_1_D_bn = nn.BatchNorm2d(32)

        self.conv1_2_D = nn.Conv2d(32, 32, 3, padding=1)
        self.conv1_2_D_bn = nn.BatchNorm2d(32)
        self.conv1_1_D = nn.Conv2d(32, out_channels, 3, padding=1)

        self.apply(self.weight_init)

    def forward(self, x, y):

        activations = {}
        ########  HSI ENCODER  ########
        # Encoder block 1
        y1 = self.conv1_1_d_bn(F.relu(self.conv1_1_d(y)))
        #y1 = self.conv1_2_d_bn(F.relu(self.conv1_2_d(y)))
        y1u = self.up6(y1)
        y = y1#self.poolHSI(y1)

        # Encoder block 2
        y2 = self.conv2_1_d_bn(F.relu(self.conv2_1_d(y)))
        #y2 = self.conv2_2_d_bn(F.relu(self.conv2_2_d(y)))
        y2u = self.up3(y2)
        y = y2#self.poolHSI(y2) 
        activations['hsi'] = y2[:,::8,:,:]

        ########  PAN ENCODER  ########
        # Encoder block 1
        x = self.conv1_1_bn(F.relu(self.conv1_1(x)))
        #x = self.conv1_2_bn(F.relu(self.conv1_2(x)))
        x1 = x
        x, mask1 = self.pool(x1)

        # Encoder block 2
        x = self.conv2_1_bn(F.relu(self.conv2_1(x)))
        #x = self.conv2_2_bn(F.relu(self.conv2_2(x)))
        x2 = x
        x, mask2 = self.pool3(x2)

        #### Serial mode: x1-x4 from SE fusion models
        xtf1, xtf2, ytf1, ytf2, att_weights = self.mtc(x1, x2, y1u, y2u)
        xtf1, xtf2, att_weights = self.mtc1(xtf1, xtf2)
        #x, mask4 = self.pool(x2)
        #y, mask4_d = self.pool(y2)

        # Encoder block y5
        y5 = self.conv5_1_d_bn(F.relu(self.conv5_1_d(y)))
        #y5 = self.conv5_2_d_bn(F.relu(self.conv5_2_d(y)))

        # Encoder block x5
        x = self.conv5_1_bn(F.relu(self.conv5_1(x)))
        #x = self.conv5_2_bn(F.relu(self.conv5_2(x)))
        x_attention = self.attention_5(x)
        y_attention = self.attention_5_d(y5)
        x = torch.mul(x, x_attention)
        y = torch.mul(y5, y_attention)
        x5 = x + y
        x, mask5 = self.pool(x5)

        ########  DECODER  ########
        # Decoder block 5
        x = self.unpool(x)
        x = x + x5
        #x = self.conv5_2_D_bn(F.relu(self.conv5_2_D(x)))
        x = self.conv5_1_D_bn(F.relu(self.conv5_1_D(x)))

        # Decoder block 2
        x = self.unpool3(x)
        x = x + x2 + xtf2
        #x = self.conv2_2_D_bn(F.relu(self.conv2_2_D(x)))
        x = self.conv2_1_D_bn(F.relu(self.conv2_1_D(x)))

        # Decoder block 1
        x = self.unpool(x)
        activations['pan'] = x[:,::4,:,:]
        x = x + x1 + xtf1
        x = self.conv1_2_D_bn(F.relu(self.conv1_2_D(x)))
        x = F.log_softmax(self.conv1_1_D(x), dim=1)

        ########  DECODER of Y only for HSI posteriors  ########
        #y_dec = self.conv5_2_D_bn(F.relu(self.conv5_2_D(y)))
        y_dec = self.conv5_1_D_bn(F.relu(self.conv5_1_D(y)))
        
        #y_dec = self.conv2_2_D_bn(F.relu(self.conv2_2_D(y_dec)))
        y_dec = self.conv2_1_D_bn(F.relu(self.conv2_1_D(y_dec)))
        
        y_dec = self.conv1_2_D_bn(F.relu(self.conv1_2_D(y_dec)))
        y_logits = F.log_softmax(self.conv1_1_D(y_dec), dim=1)
        
        return x, y_logits, activations