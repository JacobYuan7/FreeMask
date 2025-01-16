"""
Code of attention storer AttentionStore, which is a base class for attention editor in attention_util.py

"""

import abc
import os
import copy
import torch
from video_diffusion.common.util import get_time_string

class LatentControl(abc.ABC):
    
    def step_callback(self):
        self.cur_step += 1
        self.between_steps()
    
    def between_steps(self):
        return
    
    @property
    def num_uncond_att_layers(self):
        """I guess the diffusion of google has some unconditional attention layer
        No unconditional attention layer in Stable diffusion

        Returns:
            _type_: _description_
        """
        # return self.num_att_layers if config_dict['LOW_RESOURCE'] else 0
        return 0
    
    @abc.abstractmethod
    def forward (self, latent):
        raise NotImplementedError

    def __call__(self, latent):
        latent = self.forward(latent)

        return latent
    
    def reset(self):
        self.cur_step = 0


    def __init__(self, 
                 ):
        self.LOW_RESOURCE = False # assume the edit have cfg
        self.cur_step = 0



class LatentStore(LatentControl):
    def step_callback(self):
        super().step_callback()
    
    @staticmethod
    def get_empty_store():
        return {"down_latent":[]}


    def forward(self, latent):
        self.step_store["down_latent"].append(copy.deepcopy(latent))
        return latent

    def between_steps(self):     

        self.latent_store_all_step.append(copy.deepcopy(self.step_store))
        self.step_store = self.get_empty_store()



    def reset(self):
        super(LatentStore, self).reset()
        self.step_store = self.get_empty_store()
        self.latent_store_all_step = []


    def __init__(self):
        super(LatentStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.latent_store_all_step = []