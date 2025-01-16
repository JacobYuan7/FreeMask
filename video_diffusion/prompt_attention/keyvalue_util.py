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

from typing import Optional, Union, Tuple, List, Dict
import abc
import numpy as np
import copy
from einops import rearrange

import torch
import torch.nn.functional as F

import video_diffusion.prompt_attention.ptp_utils as ptp_utils
import video_diffusion.prompt_attention.seq_aligner as seq_aligner
from video_diffusion.prompt_attention.spatial_blend import SpatialBlender, NewSpatialBlender
from video_diffusion.prompt_attention.visualization import show_cross_attention, show_self_attention_comp
from video_diffusion.prompt_attention.keyvalue_store import KeyValueStore, KeyValueControl
from video_diffusion.prompt_attention.attention_store import AttentionStore, AttentionControl
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

        
        
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class KeyValueControlEdit(KeyValueStore, abc.ABC):
    """Decide self or cross-attention. Call the reweighting cross attention module

    Args:
        AttentionStore (_type_): ([1, 4, 8, 64, 64])
        abc (_type_): [8, 8, 1024, 77]
    """
    
    def step_callback(self):
        super().step_callback()
        
    def replace_keyvalue(self, key_base, key_replace, value_base, value_replace,reshaped_mask=None):
        if (key_replace.shape[0]) <= 32 ** 2:
            target_device = key_replace.device
            target_dtype  = key_replace.dtype
            key_base = key_base.to(target_device, dtype=target_dtype)
            value_base = value_base.to(target_device, dtype=target_dtype)
            if reshaped_mask is not None:
                reshaped_mask=reshaped_mask.permute(1,2,0,3).squeeze(0)
                #for shape
                return_key= (1-reshaped_mask)*key_replace + reshaped_mask*key_base
                return_value= (1-reshaped_mask)*value_replace + reshaped_mask*value_base
                return return_key,return_value
            return key_base, value_base
        else:
            return key_replace, value_replace

    def replace_selfkeyvalue(self, key_base, key_replace, value_base, value_replace,reshaped_mask=None):
        if (key_replace.shape[1]) <= 32 ** 2:
            target_device = key_replace.device
            target_dtype  = key_replace.dtype
            key_base = key_base.to(target_device, dtype=target_dtype)
            value_base = value_base.to(target_device, dtype=target_dtype)
            if reshaped_mask is not None:
                reshaped_mask=reshaped_mask.permute(1,0,2,3).squeeze(0)
                return_key= reshaped_mask*key_replace + (1-reshaped_mask)*key_base
                return_value= reshaped_mask*value_replace + (1-reshaped_mask)*value_base
                return return_key,return_value
            else:
                return key_base,value_base
        else:
            return key_replace, value_replace
    
    
    def update_attention_position_dict(self, current_attention_key):
        self.keyvalue_position_counter_dict[current_attention_key] +=1


    def forward(self, key,value, place_in_unet: str):
        super(KeyValueControlEdit, self).forward(key,value, place_in_unet)
        start=key.shape[0]//2
        if (key.shape[0]//2) <= 102 ** 2:
            position_key = f"{place_in_unet}_{'key'}"
            position_value = f"{place_in_unet}_{'value'}"
            current_pos = self.keyvalue_position_counter_dict[position_key]
            step_in_store = len(self.additional_keyvalue_store.keyvalue_store_all_step) - self.cur_step -1
            step_in_store_keyvalue_dict = self.additional_keyvalue_store.keyvalue_store_all_step[step_in_store]
            step_in_store_atten_dict =self.additional_attention_store.attention_store_all_step[10]
            key_base = step_in_store_keyvalue_dict[position_key][current_pos]        
            value_base = step_in_store_keyvalue_dict[position_value][current_pos]      
            self.update_attention_position_dict(position_key)
            self.update_attention_position_dict(position_value)

            if "temp" in place_in_unet:
                if key_base.shape[0]==1024:
                    attention_store= step_in_store_atten_dict["down_cross"][0:2]+step_in_store_atten_dict["up_cross"][3:6]
                elif key_base.shape[0]==256:
                    attention_store=step_in_store_atten_dict["down_cross"][2:4]+step_in_store_atten_dict["up_cross"][0:3]
                elif "mid" in place_in_unet:
                    attention_store=step_in_store_atten_dict["mid_cross"][0]
                if  self.num_temp_replace[0] <= self.cur_step < self.num_temp_replace[1]:
                    start=key.shape[0]//2
                    key_base,key_replace=key_base,key[start:]
                    value_base,value_replace=value_base,value[start:]
                        
                    if key_replace.shape[0] <= 32 ** 2:
                        if self.attention_blend: #when style editing, is none
                            h = int(np.sqrt(key_base.shape[0]))
                            w = h
                            mask = self.attention_blend(target_h = h, target_w =w, attention_store=attention_store, step_in_store=step_in_store,position=position_key)
                            reshaped_mask = rearrange(mask, "d c h w -> c d (h w)")[..., None]
                        else:
                            reshaped_mask=None
                    else: 
                        reshaped_mask = None
                    key[start:],value[start:]= self.replace_keyvalue(key_base, key_replace, value_base, value_replace, reshaped_mask)   
                                     
  
        
            elif "self" in place_in_unet:
                if key_base.shape[1]==1024:
                    attention_store= step_in_store_atten_dict["down_cross"][0:2]+step_in_store_atten_dict["up_cross"][3:6]
                elif key_base.shape[1]==256:
                    attention_store=step_in_store_atten_dict["down_cross"][2:4]+step_in_store_atten_dict["up_cross"][0:3]
                elif "mid" in place_in_unet:
                    attention_store=step_in_store_atten_dict["mid_cross"][0]
                if  self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]:
                    start=key.shape[0]//2
                    key_base,key_replace=key_base,key[start:]
                    value_base,value_replace=value_base,value[start:]
                        
                    if key_replace.shape[1] <= 32 ** 2:
                        h = int(np.sqrt(key_base.shape[1]))
                        w = h
                        if self.attention_blend:
                            mask = self.attention_blend(target_h = h, target_w =w, attention_store=attention_store, step_in_store=step_in_store)
                            # reshape from ([ 1, 2, 32, 32]) -> [2, 1, 1024, 1]
                            reshaped_mask = rearrange(mask, "d c h w -> c d (h w)")[..., None]
                            
                        else:
                            reshaped_mask=None
                    else: 
                        reshaped_mask = None
                    key[start:],value[start:]= self.replace_selfkeyvalue(key_base, key_replace, value_base, value_replace, reshaped_mask)
        return key,value
    
    def between_steps(self):

        super().between_steps()
        self.step_store = self.get_empty_store()
        
        self.keyvalue_position_counter_dict = {
            'down_temp_key': 0,
            'mid_temp_key': 0,
            'up_temp_key': 0,
            'down_temp_value': 0,
            'mid_temp_value': 0,
            'up_temp_value': 0,
            'down_self_key': 0,
            'mid_self_key': 0,
            'up_self_key': 0,
            'down_self_value': 0,
            'mid_self_value': 0,
            'up_self_value': 0,
        }        
        return 

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_temp_steps: Union[float, Tuple[float, float]],
                 self_steps: Union[float, Tuple[float, float]],
                 tokenizer=None, 
                 additional_attention_store: AttentionStore =None,
                 additional_keyvalue_store: KeyValueStore=None,
                 attention_blend: SpatialBlender= None,
                 ):
        super(KeyValueControlEdit, self).__init__()
        self.additional_attention_store = additional_attention_store
        self.additional_keyvalue_store=additional_keyvalue_store
        self.batch_size = len(prompts)
        self.attention_blend = attention_blend
        if self.additional_attention_store is not None:
            # the attention_store is provided outside, only pass in one promp
            self.batch_size = len(prompts) //2
            assert self.batch_size==1, 'Only support single video editing with additional attention_store'

        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps, tokenizer).to(device)
        if type(self_temp_steps) is float:
            self_temp_steps = 0, self_temp_steps
        self.num_temp_replace = int(num_steps * self_temp_steps[0]), int(num_steps * self_temp_steps[1])
        if type(self_steps) is float:
            self_steps = 0, self_steps
        self.num_self_replace = int(num_steps * self_steps[0]), int(num_steps * self_steps[1])
        # We need to know the current position in attention
        self.prev_attention_key_name = 0
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])
        self.keyvalue_position_counter_dict = {
            'down_temp_key': 0,
            'mid_temp_key': 0,
            'up_temp_key': 0,
            'down_temp_value': 0,
            'mid_temp_value': 0,
            'up_temp_value': 0,
            'down_self_key': 0,
            'mid_self_key': 0,
            'up_self_key': 0,
            'down_self_value': 0,
            'mid_self_value': 0,
            'up_self_value': 0,
        }








def make_controller(tokenizer, prompts: List[str], 
                    cross_replace_steps: Dict[str, float],
                    self_temp_steps: float=0.0, 
                    self_steps: float=0.0, 
                    additional_attention_store=None, 
                    additional_keyvalue_store=None, 
                    blend_th: float=(0.3, 0.3),
                    blend_words=None, 
                    NUM_DDIM_STEPS=None,
                    ) -> KeyValueControlEdit:
    if (blend_words is None) or (blend_words == 'None'):
        attention_blend =None
    else:
        attention_blend = NewSpatialBlender( prompts, blend_words, 
                                                start_blend = 0.0, end_blend=2,
                                                tokenizer=tokenizer, th=blend_th, NUM_DDIM_STEPS=NUM_DDIM_STEPS,
                        save_path=None,
                        prompt_choose='source')
        print(f'Blend self attention mask with threshold {blend_th}')



    controller = KeyValueControlEdit(prompts, NUM_DDIM_STEPS,
                                    cross_replace_steps=cross_replace_steps, 
                                    self_temp_steps=self_temp_steps,
                                    self_steps=self_steps,
                                    tokenizer=tokenizer,
                                    additional_attention_store=additional_attention_store,
                                    additional_keyvalue_store=additional_keyvalue_store,
                                    attention_blend=attention_blend,
                                    )
    
    return controller



