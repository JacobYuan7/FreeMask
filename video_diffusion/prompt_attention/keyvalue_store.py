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

class KeyValueControl(abc.ABC):
    
    def step_callback(self):
        self.cur_att_layer = 0
        self.cur_step += 1
        self.between_steps()
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        return 0
    
    @abc.abstractmethod
    def forward (self, key, value, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, key, value, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            key,value = self.forward(key, value, place_in_unet)
            self.cur_att_layer += 1

        return key,value
    
    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self, 
                 ):
        self.LOW_RESOURCE = False # assume the edit have cfg
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class KeyValueStore(KeyValueControl):
    def step_callback(self):
        super().step_callback()
    
    @staticmethod
    def get_empty_store():
        return {"down_temp_key": [], "mid_temp_key": [], "up_temp_key": [],
                "down_temp_value": [],  "mid_temp_value": [],  "up_temp_value": [],
                "down_self_key": [], "mid_self_key": [], "up_self_key": [],
                "down_self_value": [],  "mid_self_value": [],  "up_self_value": []}


    def forward(self, key, value,place_in_unet: str):
        position_key = f"{place_in_unet}_{'key'}"
        position_value = f"{place_in_unet}_{'value'}"
        append_key=key.cpu().detach()
        append_value=value.cpu().detach()
        self.step_store[position_key].append(copy.deepcopy(append_key))
        self.step_store[position_value].append(copy.deepcopy(append_value))
        return key,value

    def between_steps(self):
        if len(self.keyvalue_store) == 0:
            self.keyvalue_store = self.step_store
        else:
            for item in self.keyvalue_store:
                for i in range(len(self.keyvalue_store[item])):
                    self.keyvalue_store[item][i] += self.step_store[item][i]      

        self.keyvalue_store_all_step.append(copy.deepcopy(self.step_store))
        self.step_store = self.get_empty_store()



    def reset(self):
        super(KeyValueStore, self).reset()
        self.step_store = self.get_empty_store()
        self.keyvalue_store_all_step = []
        self.keyvalue_store = {}

    def __init__(self, save_keyvalue:bool=True):
        super(KeyValueStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.keyvalue_store = {}
        self.save_keyvalue = save_keyvalue
        self.keyvalue_store_all_step = []