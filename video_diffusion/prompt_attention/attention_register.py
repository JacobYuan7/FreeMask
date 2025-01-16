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
from einops import rearrange
import torch
import torch.nn.functional as F
import gc

def register_attention_control(model, controller,controller1=None):
    "Connect a model with a controller"
    def attention_controlled_forward(self, place_in_unet, attention_type='cross'):
        to_out = self.to_out
        # print("self.to_out: ",to_out)
        if type(to_out) is torch.nn.modules.container.ModuleList:
            to_out = self.to_out[0]
        else:
            to_out = self.to_out

        def reshape_heads_to_batch_dim(tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
            return tensor
        
        def reshape_batch_dim_to_heads(tensor):
            batch_size, seq_len, dim = tensor.shape
            head_size = self.heads
            tensor = tensor.reshape(batch_size // head_size, head_size, seq_len, dim)
            tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size // head_size, seq_len, dim * head_size)
            return tensor
        
        def _attention( query, key, value, is_cross, attention_mask=None):
            if self.upcast_attention:
                query = query.float()
                key = key.float()

            attention_scores = torch.baddbmm(
                torch.empty(query.shape[0], query.shape[1], key.shape[1], dtype=query.dtype, device=query.device),
                query,
                key.transpose(-1, -2),
                beta=0,
                alpha=self.scale,
            )
            if attention_mask is not None:
                attention_scores = attention_scores + attention_mask

            if self.upcast_softmax:
                attention_scores = attention_scores.float()

            attention_probs = attention_scores.softmax(dim=-1)

            attention_probs = attention_probs.to(value.dtype)

            #controller
            attention_probs = controller(reshape_batch_dim_to_temporal_heads(attention_probs), 
                                         is_cross, place_in_unet)
            attention_probs = reshape_temporal_heads_to_batch_dim(attention_probs)

            hidden_states = torch.bmm(attention_probs, value)#矩阵乘法
            hidden_states = reshape_batch_dim_to_heads(hidden_states)

            return hidden_states


        def reshape_temporal_heads_to_batch_dim( tensor):
            head_size = self.heads
            tensor = rearrange(tensor, " b h s t -> (b h) s t ", h = head_size)
            return tensor

        def reshape_batch_dim_to_temporal_heads(tensor):
            head_size = self.heads
            tensor = rearrange(tensor, "(b h) s t -> b h s t", h = head_size)
            return tensor
        
        @torch.no_grad()
        def attention_forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            is_cross = encoder_hidden_states is not None
            if hidden_states.shape[1]<hidden_states.shape[0]:
                hidden_states=hidden_states.permute(1,0,2)
            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
                tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
                return tensor
            def reshape_batch_dim_to_temporal_heads(tensor):
                head_size = self.heads
                tensor = rearrange(tensor, "b s (t h) -> b h s t", h = head_size)
                return tensor

            residual = hidden_states
            flag=0

            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )            

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])


            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
                flag=1
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            encoder_hidden_states = encoder_hidden_states if encoder_hidden_states is not None else hidden_states
            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states) #(8,4096,320)
            

            key =reshape_heads_to_batch_dim(key)
            value =reshape_heads_to_batch_dim(value)
            query =reshape_heads_to_batch_dim(query)

            hidden_states = _attention(query, key, value, is_cross=is_cross, attention_mask=attention_mask)
       
            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            if flag == 1 :
                hidden_states = self.to_out[1](hidden_states)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor
            return hidden_states

        @torch.no_grad()        
        def temp_attention_forward(hidden_states, encoder_hidden_states=None, attention_mask=None):
            is_cross = encoder_hidden_states is not None

            def reshape_heads_to_batch_dim(tensor):
                batch_size, seq_len, dim = tensor.shape
                head_size = self.heads
                tensor = tensor.reshape(batch_size, seq_len, head_size, dim // head_size)
                tensor = tensor.permute(0, 2, 1, 3).reshape(batch_size * head_size, seq_len, dim // head_size)
                return tensor
            def reshape_batch_dim_to_temporal_heads(tensor):
                head_size = self.heads
                tensor = rearrange(tensor, "b s (t h) -> b h s t", h = head_size)
                return tensor

            residual = hidden_states
            flag=0
            if self.spatial_norm is not None:
                hidden_states = self.spatial_norm(hidden_states, temb)

            input_ndim = hidden_states.ndim

            if input_ndim == 4:
                batch_size, channel, height, width = hidden_states.shape
                hidden_states = hidden_states.view(batch_size, channel, height * width).transpose(1, 2)

            batch_size, sequence_length, _ = (
                hidden_states.shape if encoder_hidden_states is None else encoder_hidden_states.shape
            )

            if attention_mask is not None:
                attention_mask = self.prepare_attention_mask(attention_mask, sequence_length, batch_size)
                attention_mask = attention_mask.view(batch_size, attn.heads, -1, attention_mask.shape[-1])


            if self.group_norm is not None:
                hidden_states = self.group_norm(hidden_states.transpose(1, 2)).transpose(1, 2)

            query = self.to_q(hidden_states)

            if encoder_hidden_states is None:
                encoder_hidden_states = hidden_states
                flag=1
            elif self.norm_cross:
                encoder_hidden_states = self.norm_encoder_hidden_states(encoder_hidden_states)

            key = self.to_k(encoder_hidden_states)
            value = self.to_v(encoder_hidden_states)

            if not (controller1== None):
                key,query=controller1(key,query,place_in_unet)

            inner_dim = key.shape[-1]
            head_dim = inner_dim // self.heads
            query = query.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key = key.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            value = value.view(batch_size, -1, self.heads, head_dim).transpose(1, 2)
            key= key.reshape(batch_size*self.heads,-1,head_dim)
            value=value.reshape(batch_size*self.heads,-1,head_dim)
            query=query.reshape(batch_size*self.heads,-1,head_dim)

            hidden_states = _attention(query, key, value, is_cross=is_cross, attention_mask=attention_mask)
       
            # linear proj
            hidden_states = self.to_out[0](hidden_states)
            # dropout
            if flag == 1 :
                hidden_states = self.to_out[1](hidden_states)

            if self.residual_connection:
                hidden_states = hidden_states + residual

            hidden_states = hidden_states / self.rescale_output_factor

            return hidden_states



        if attention_type == "TempAttention":
            return temp_attention_forward
        elif attention_type == "Attention":
            return attention_forward

    class DummyController:

        def __call__(self, *args):
            return args[0]

        def __init__(self):
            self.num_att_layers = 0

    if controller is None:
        controller = DummyController()
    

    def register_recr(net_, count, place_in_unet):
        place0=place_in_unet
        if net_[1].__class__.__name__ in ['Attention','TempAttention']: 
            net_[1].forward = attention_controlled_forward(net_[1], place_in_unet, attention_type = net_[1].__class__.__name__)
            return count + 1
        elif hasattr(net_[1], 'children'):
            for net in net_[1].named_children():
                if net[0] =='temp_attentions':
                    place_in_unet=place0+"_temp"
                    # print("place_in_unet: ",place_in_unet)
                    count = register_recr(net, count, place_in_unet)
                else:
                    count = register_recr(net, count, place_in_unet)
        return count

    cross_att_count = 0
    sub_nets = model.unet.named_children()
    for net in sub_nets:
        if "down" in net[0]:
            cross_att_count += register_recr(net, 0, "down")
        elif "up" in net[0]:
            cross_att_count += register_recr(net, 0, "up")
        elif "mid" in net[0]:
            cross_att_count += register_recr(net, 0, "mid")
    print(f"Number of attention layer registered {cross_att_count}")
    controller.num_att_layers = cross_att_count