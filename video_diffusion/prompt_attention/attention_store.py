# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import abc
import os
import copy
import torch
from video_diffusion.common.util import get_time_string
from typing import Optional, Union, Tuple, List, Callable, Dict
from video_diffusion.prompt_attention import ptp_utils
import numpy as np
import matplotlib.pyplot as plt
import datetime
from PIL import Image 
import os
from einops import rearrange

class AttentionControl(abc.ABC):
    
    def step_callback(self, x_t):
        self.cur_att_layer = 0
        self.cur_step += 1
        self.between_steps()
        return x_t
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        # return self.num_att_layers if config_dict['LOW_RESOURCE'] else 0
        return 0
    
    @abc.abstractmethod
    def forward (self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if self.LOW_RESOURCE:
                # For inversion without null text file 
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                # For classifier-free guidance scale!=1
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)#ddiminversion的时候没有用到
        self.cur_att_layer += 1
        return attn
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, 
                 ):
        self.LOW_RESOURCE = False # assume the edit have cfg
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class AttentionStore(AttentionControl):
    def step_callback(self, x_t):
        x_t = super().step_callback(x_t)
        self.latents_store.append(x_t.cpu().detach())
        return x_t

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [],  "mid_self": [],  "up_self": [],
                "down_temp_cross": [], "mid_temp_cross": [], "up_temp_cross": [],
                "down_temp_self": [],  "mid_temp_self": [],  "up_temp_self": [],
                }

    @staticmethod
    def get_empty_cross_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                }


    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[-2] <= 2*(32 ** 2):  # avoid memory overhead
            if is_cross or self.save_self_attention:
                if attn.shape[-2] >= 32**2:
                    append_tensor = attn.cpu().detach()
                else:
                    append_tensor = attn.cpu().detach()
            
                self.step_store[key].append(copy.deepcopy(append_tensor))

        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    self.attention_store[key][i] += self.step_store[key][i]                
        
        if self.disk_store:
            path = self.store_dir + f'/{self.cur_step:03d}.pt'
            torch.save(copy.deepcopy(self.step_store), path)
            self.attention_store_all_step.append(path)
        else:
            self.attention_store_all_step.append(copy.deepcopy(self.step_store))
        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        "divide the attention map value in attention store by denoising steps"
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in self.attention_store}
        return average_attention


    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store_all_step = []
        self.attention_store = {}

    def __init__(self, save_self_attention:bool=True, disk_store=False):
        super(AttentionStore, self).__init__()
        self.disk_store = disk_store
        if self.disk_store:#不存
            time_string = get_time_string()
            path = f'./trash/attention_cache_{time_string}'
            os.makedirs(path, exist_ok=True)
            self.store_dir = path
        else:
            self.store_dir =None
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.save_self_attention = save_self_attention
        self.latents_store = []
        self.attention_store_all_step = []

    def visualization_camap(self,vil,dataset_name):
        steps=len(self.attention_store_all_step)
        for step in range(0,steps):
            attention_store=[]
            name=[]
            count=0
            for item in self.attention_store_all_step[step]["down_cross"]:
                attention_store.append(item)
                name_temp=f"down_cross_{count}"
                name.append(name_temp)
                print(name_temp)
                print(item.shape)
                count+=1
            count=0
            for item in self.attention_store_all_step[step]["up_cross"]:
                attention_store.append(item)
                name_temp=f"up_cross_{count}"
                name.append(name_temp)
                print(name_temp)
                print(item.shape)
                count+=1
            vil(attention_store,step,name,dataset_name)


class VisualizationMask:
    def __call__(self, attention_store,step,name,dataset_name):
        k = 1
        for i in range(0,len(attention_store)):
            item=attention_store[i]
            if len(item.shape) == 4: item = item[None, ...]
            ( p, c, heads, r, w)= item.shape #(1,8,20,256,77)
            res_h = int(np.sqrt(r))
            rearranged_item = rearrange(  item, "p c h (res_h res_w) w -> p h c res_h res_w w ", 
                            h=heads, res_h=res_h, res_w=res_h)
            maps = rearranged_item
            if maps.dim() == 5: self.alpha = self.alpha[:, None, ...]
            maps=maps.to("cuda")
            maps = (maps * self.alpha_layers).sum(-1).mean(1) #（maps*self.alpha_layers全是正数）
            folder_path=f'./camap/{dataset_name}'
            if not os.path.exists(folder_path):
                # 如果文件夹不存在，创建它
                os.makedirs(folder_path)
            if step%1==0:
                data_np=maps.squeeze(0).permute(1,0,2).detach().cpu().numpy()
                data_np=data_np.reshape(-1,res_h*8)
                fig, ax = plt.subplots(figsize=(10, 10*8))
                heatmap = ax.imshow(data_np, cmap='hot', interpolation='nearest')
                ax.axis('off') 
                filename = os.path.join(f'{folder_path}/maps_{name[i]}_step{step:02d}.png')
                plt.savefig(filename, bbox_inches='tight', pad_inches=0, dpi=300)
                plt.close(fig)               
       
    def __init__(self, prompts: List[str], words: [List[List[str]]], tokenizer,threshold=.3):
        MAX_NUM_WORDS=77
        alpha_layers = torch.zeros(1, 1, 1, 1,MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[:,:,:,:, ind] = 1
        self.alpha_layers = alpha_layers.to("cuda")
        self.threshold = threshold
        self.prompts=prompts
        self.tokenizer=tokenizer