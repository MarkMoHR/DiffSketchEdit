from abc import ABC, abstractmethod
from typing import Optional, Union, Tuple, List, Dict

import torch
import torch.nn.functional as F

from .ptp_utils import (get_word_inds, get_time_words_attention_alpha)
from .seq_aligner import (get_replacement_mapper, get_refinement_mapper)


class AttentionControl(ABC):

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return 0

    @abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            h = attn.shape[0]
            attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {
            key: [item / self.cur_step for item in self.attention_store[key]]
            for key in self.attention_store
        }
        return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}


class LocalBlend:

    def __init__(self,
                 prompts: List[str],
                 words: [List[List[str]]],
                 tokenizer,
                 device,
                 threshold=.3,
                 max_num_words=77):
        self.max_num_words = max_num_words

        alpha_layers1 = torch.zeros(len(prompts), 1, 1, 1, 1, self.max_num_words)
        alpha_layers2 = torch.zeros(len(prompts), 1, 1, 1, 1, self.max_num_words)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            assert type(words_) is list and len(words_) == 2
            ind1 = get_word_inds(prompt, words_[0], tokenizer)
            alpha_layers1[i, :, :, :, :, ind1] = 1
            ind2 = get_word_inds(prompt, words_[1], tokenizer)
            alpha_layers2[i, :, :, :, :, ind2] = 1

        self.alpha_layers1 = alpha_layers1.to(device)
        self.alpha_layers2 = alpha_layers2.to(device)
        self.threshold = threshold

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers1.shape[0], -1, 1, 16, 16, self.max_num_words) for item in maps]
        maps = torch.cat(maps, dim=1)

        maps1 = (maps * self.alpha_layers1).sum(-1).mean(1)
        mask1 = F.max_pool2d(maps1, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask1 = F.interpolate(mask1, size=(x_t.shape[2:]))
        mask1 = mask1 / mask1.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask1 = mask1.gt(self.threshold)

        maps2 = (maps * self.alpha_layers2).sum(-1).mean(1)
        mask2 = F.max_pool2d(maps2, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask2 = F.interpolate(mask2, size=(x_t.shape[2:]))
        mask2 = mask2 / mask2.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask2 = mask2.gt(self.threshold)

        mask = (mask1 + mask2).float()

        prev_x_t = torch.cat([x_t[:1], x_t[:-1]], dim=0)
        x_t = (1 - mask) * prev_x_t + mask * x_t
        return x_t


class AttentionControlEdit(AttentionStore, ABC):

    def __init__(self,
                 prompts,
                 num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend],
                 tokenizer,
                 device):
        super(AttentionControlEdit, self).__init__()
        self.tokenizer = tokenizer
        self.device = device

        self.batch_size = len(prompts)
        self.cross_replace_alpha = get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                  self.tokenizer).to(self.device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend  # define outside

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.clone()
        else:
            return att_replace

    @abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        # FIXME not replace correctly
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[:-1], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                        1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn


class AttentionReplace(AttentionControlEdit):

    def __init__(self,
                 prompts,
                 num_steps: int,
                 cross_replace_steps: float,
                 self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,
                 tokenizer=None,
                 device=None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                                               local_blend, tokenizer, device)
        self.mapper = get_replacement_mapper(prompts, self.tokenizer).to(self.device)

    def replace_cross_attention(self, attn_base, att_replace):
        # attn_base/att_replace: (len(prompts)-1, 8, 4096, 77)
        # self.mapper: (len(prompts)-1, 77, 77)
        version = 'v2'

        if version == 'v1':
            return torch.einsum('bhpw,bwn->bhpn', attn_base, self.mapper)
        else:
            bsz = attn_base.size()[0]
            attn_base_replace = []
            for batch_i in range(bsz):
                if batch_i == 0:
                    attn_base_i = attn_base[batch_i]  # (8, 4096, 77)
                else:
                    attn_base_i = attn_base_replace[-1]
                mapper_i = self.mapper[batch_i:batch_i + 1, :, :]  # (1, 77, 77)
                attn_base_replace_i = torch.einsum('hpw,bwn->bhpn', attn_base_i, mapper_i)  # (1, 8, 4096, 77)
                attn_base_replace.append(attn_base_replace_i[0])
            attn_base_replace = torch.stack(attn_base_replace, dim=0)  # (len(prompts)-1, 8, 4096, 77)
            return attn_base_replace


class AttentionRefine(AttentionControlEdit):

    def __init__(self,
                 prompts,
                 num_steps: int,
                 cross_replace_steps: float,
                 self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None,
                 tokenizer=None,
                 device=None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                                              local_blend, tokenizer, device)
        self.mapper, alphas = get_refinement_mapper(prompts, self.tokenizer)
        self.mapper, alphas = self.mapper.to(self.device), alphas.to(self.device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])

    def replace_cross_attention(self, attn_base, att_replace):
        # attn_base/att_replace: (len(prompts)-1, 8, 4096, 77)
        version = 'v2'

        bsz = attn_base.size()[0]
        attn_base_replace = []
        for batch_i in range(bsz):
            if version == 'v1':
                attn_base_i = attn_base[batch_i]  # (8, 4096, 77)
            else:
                if batch_i == 0:
                    attn_base_i = attn_base[batch_i]
                else:
                    attn_base_i = attn_base_replace[-1]
            mapper_i = self.mapper[batch_i:batch_i + 1, :]  # (1, 77)
            attn_base_replace_i = attn_base_i[:, :, mapper_i].permute(2, 0, 1, 3)  # (1, 8, 4096, 77)
            attn_base_replace.append(attn_base_replace_i[0])
        attn_base_replace = torch.stack(attn_base_replace, dim=0)  # (len(prompts)-1, 8, 4096, 77)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace


class AttentionReweight(AttentionControlEdit):

    def __init__(self,
                 prompts,
                 num_steps: int,
                 cross_replace_steps: float,
                 self_replace_steps: float,
                 equalizer,
                 local_blend: Optional[LocalBlend] = None,
                 controller: Optional[AttentionControlEdit] = None,
                 tokenizer=None,
                 device=None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                                                local_blend, tokenizer, device)
        self.equalizer = equalizer.to(self.device)
        self.prev_controller = controller

    def replace_cross_attention(self, attn_base, att_replace):
        # attn_base/att_replace: (len(prompts)-1, 8, 4096, 77)
        # self.equalizer: (len(prompts)-1, 77)
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)

        version = 'v2'

        if version == 'v1':
            attn_replace = attn_base[:, :, :, :] * self.equalizer[:, None, None, :]
            return attn_replace
        else:
            bsz = attn_base.size()[0]
            attn_replace_rst_all = []
            for bi in range(bsz):
                if bi == 0:
                    attn_replace_rst = attn_base[bi, :, :, :] * self.equalizer[bi, None, None, :]
                else:
                    attn_replace_rst = attn_replace_rst_all[-1] * self.equalizer[bi, None, None, :]
                attn_replace_rst_all.append(attn_replace_rst)
            attn_replace_rst_all = torch.stack(attn_replace_rst_all, dim=0)
            return attn_replace_rst_all


def get_equalizer(tokenizer, texts: List[str],
                  word_select: List[str],
                  values: List[float]):
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for wi, word in enumerate(word_select):
        text = texts[wi]
        value = values[wi]
        inds = get_word_inds(text, word, tokenizer)
        equalizer[wi, inds] = value
    return equalizer
