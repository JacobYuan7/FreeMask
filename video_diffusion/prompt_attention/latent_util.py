"""
Collect all function in prompt_attention folder.
Provide a API `make_controller' to return an initialized AttentionControlEdit class object in the main validation loop.
"""

from typing import Optional, Union, Tuple, List, Dict
import abc
import numpy as np
import copy
from einops import rearrange

import torch
import torch.nn.functional as F

import video_diffusion.prompt_attention.ptp_utils as ptp_utils
import video_diffusion.prompt_attention.seq_aligner as seq_aligner
from video_diffusion.prompt_attention.spatial_blend import SpatialBlender
from video_diffusion.prompt_attention.visualization import show_cross_attention, show_self_attention_comp
from video_diffusion.prompt_attention.latent_store import LatentStore, LatentControl
from video_diffusion.prompt_attention.model_register import register_model_control
# from video_diffusion.prompt_attention.attention_register_edit import register_attention_control_edit
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


        
        
class EmptyControl:
    
    
    def step_callback(self, x_t):
        return x_t
    
    def between_steps(self):
        return
    
    def __call__(self,latent):
        return latent


class LatentControlEdit(LatentStore, abc.ABC):
    """Decide self or cross-attention. Call the reweighting cross attention module

    Args:
        AttentionStore (_type_): ([1, 4, 8, 64, 64])
        abc (_type_): [8, 8, 1024, 77]
    """
    
    def step_callback(self):
        super().step_callback()




    def forward(self, x_t):
        super(LatentControlEdit, self).forward(x_t)
        x_t_device = x_t[0].device
        x_t_dtype = x_t[0].dtype
        self.latent_blend=True
        self.use_inversion_attention=True
        # print("self.cure_step: ",self.cur_step)
        # print("len(self.additional_latent_store.latent_store_all_step): ",len(self.additional_latent_store.latent_store_all_step))
        if self.latent_blend is True:
            if self.use_inversion_attention:
                step_in_store = len(self.additional_latent_store.latent_store_all_step) - self.cur_step-1
            else:
                step_in_store = self.cur_step
            
            inverted_latents =self.additional_latent_store.latent_store_all_step[step_in_store]["down_latent"][0]
            # print("x_t.shape: ",len(x_t),x_t[0].shape,len(inverted_latents),inverted_latents[0].shape)
            start=x_t[0].shape[0]//2
            # for i in range(6,8):        
            #     x_t[i][start:]=inverted_latents[i].to(x_t_device,dtype=x_t_dtype)
        return x_t

    
    def between_steps(self):

        super().between_steps()
        self.step_store = self.get_empty_store()
          
        return 
    def __init__(self, num_steps: int,
                 latent_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 latent_blend=True, 
                 additional_latent_store: LatentStore =None,
                 use_inversion_attention: bool=True,
                 ):
        super(LatentControlEdit, self).__init__()
        self.additional_latent_store = additional_latent_store#之前存的存到这里了
        if type(latent_replace_steps) is float:
            latent_replace_steps = 0, latent_replace_steps
        self.num_replace = int(num_steps * latent_replace_steps[0]), int(num_steps * latent_replace_steps[1])#(0,6)
        self.latent_blend = latent_blend
        self.use_inversion_attention = use_inversion_attention







def make_controller(latent_replace_steps: Dict[str, float], 
                    additional_latent_store=None, use_inversion_attention = True, 
                    NUM_DDIM_STEPS=None,
                    blend_latents = True,
                    ) -> LatentControlEdit:
    controller=LatentControlEdit(num_steps=NUM_DDIM_STEPS,latent_replace_steps=latent_replace_steps,
    latent_blend=blend_latents,additional_latent_store=additional_latent_store,use_inversion_attention=use_inversion_attention)
    return controller


