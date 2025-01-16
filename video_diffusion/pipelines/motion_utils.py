import gc
import random
from math import sqrt
import numpy as np
import torch
from typing import Union, List
from torchvision.io import write_video
import argparse
import copy
import os
from pathlib import Path
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from diffusers import TextToVideoSDPipeline, DDIMScheduler
from diffusers.pipelines.text_to_video_synthesis.pipeline_text_to_video_synth import tensor2vid
from einops import rearrange
from omegaconf import OmegaConf
from torch.cuda.amp import GradScaler
from torchvision.transforms import ToTensor
from tqdm import tqdm
from transformers import logging



def clean_memory():
    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()

def isinstance_str(x: object, cls_name: Union[str, List[str]]):
    """
    Checks whether x has any class *named* cls_name in its ancestry.
    Doesn't require access to the class's implementation.

    Useful for patching!
    """
    if type(cls_name) == str:
        for _cls in x.__class__.__mro__:
            if _cls.__name__ == cls_name:
                return True
    else:
        for _cls in x.__class__.__mro__:
            if _cls.__name__ in cls_name:
                return True
    return False

def register_time(pipeline, t):
    for _, module in pipeline.unet.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "t", t)

def register_batch(pipeline, b):
    for _, module in pipeline.unet.named_modules():
        if isinstance_str(module, ["ModuleWithGuidance", "ModuleWithConvGuidance"]):
            setattr(module, "b", b)

@torch.autocast(device_type="cuda", dtype=torch.float32)
def calculate_losses(orig_features, target_features, **args):
    orig = orig_features
    target = target_features

    orig = orig.detach()

    total_loss = 0
    losses = {}
    if args["features_loss_weight"] > 0:
        if args["global_averaging"]:
            orig = orig.mean(dim=(2, 3), keepdim=True)
            target = target.mean(dim=(2, 3), keepdim=True)

        features_loss = compute_feature_loss(orig, target)
        total_loss += args["features_loss_weight"] * features_loss
        losses["features_mse_loss"] = features_loss

    if args["features_diff_loss_weight"] > 0:
        features_diff_loss = 0
        # print("original.shape: ",orig.shape)
        # print("target.shape: ",target.shape)
        orig = orig.mean(dim=(-2, -1), keepdim=True)  # t d 1 1
        target = target.mean(dim=(-2,-1), keepdim=True)

        for i in range(len(orig)):
            orig_anchor = orig[i]
            target_anchor = target[i]
            orig_diffs = orig - orig_anchor  # t d 1 1
            target_diffs = target - target_anchor  # t d 1 1
            features_diff_loss += 1 - F.cosine_similarity(target_diffs, orig_diffs.detach(), dim=1).mean()
        features_diff_loss /= len(orig)

        total_loss += args["features_diff_loss_weight"] * features_diff_loss
        losses["features_diff_loss"] = features_diff_loss

    losses["total_loss"] = total_loss
    return losses

def register_guidance(pipeline,**kwargs):
    guidance_schedule = pipeline.guidance_schedule
    num_frames = kwargs["image"].shape[0]
    h = kwargs["image"].shape[2]
    w = kwargs["image"].shape[3]

    class ModuleWithConvGuidance(torch.nn.Module):
        def __init__(pipeline, module, guidance_schedule, num_frames, h, w, block_name, config, module_type):
            super().__init__()
            pipeline.module = module
            pipeline.guidance_schedule = guidance_schedule
            pipeline.num_frames = num_frames
            assert module_type in [
                "spatial_convolution",
            ]
            pipeline.module_type = module_type
            if pipeline.module_type == "spatial_convolution":
                pipeline.starting_shape = "(b t) d h w"
            pipeline.h = h
            pipeline.w = w
            pipeline.block_name = block_name
            pipeline.config = config
            pipeline.saved_features = None

        def forward(pipeline, input_tensor, temb):
            # print("pipeline.module: ",pipeline.module)
            hidden_states = input_tensor

            hidden_states = pipeline.module.norm1(hidden_states)
            hidden_states = pipeline.module.nonlinearity(hidden_states)

            if pipeline.module.upsample is not None:
                # upsample_nearest_nhwc fails with large batch sizes. see https://github.com/huggingface/diffusers/issues/984
                if hidden_states.shape[0] >= 64:
                    input_tensor = input_tensor.contiguous()
                    hidden_states = hidden_states.contiguous()
                input_tensor = pipeline.module.upsample(input_tensor)
                hidden_states = pipeline.upsample(hidden_states)
            elif pipeline.module.downsample is not None:
                input_tensor = pipeline.module.downsample(input_tensor)
                hidden_states = pipeline.module.downsample(hidden_states)

            hidden_states = pipeline.module.conv1(hidden_states)

            if temb is not None:
                temb = pipeline.module.time_emb_proj(pipeline.module.nonlinearity(temb))[:, :, None, None]

            if temb is not None and pipeline.module.time_embedding_norm == "default":
                hidden_states = hidden_states + temb

            hidden_states = pipeline.module.norm2(hidden_states)

            if temb is not None and pipeline.module.time_embedding_norm == "scale_shift":
                scale, shift = torch.chunk(temb, 2, dim=1)
                hidden_states = hidden_states * (1 + scale) + shift

            hidden_states = pipeline.module.nonlinearity(hidden_states)

            hidden_states = pipeline.module.dropout(hidden_states)
            hidden_states = pipeline.module.conv2(hidden_states)

            if pipeline.config["guidance_before_res"] and (pipeline.t.cpu() in pipeline.guidance_schedule):
                pipeline.saved_features = rearrange(
                    hidden_states, f"{pipeline.starting_shape} -> b t d h w", t=pipeline.num_frames
                )

            if pipeline.module.conv_shortcut is not None:
                input_tensor = pipeline.module.conv_shortcut(input_tensor)

            output_tensor = (input_tensor + hidden_states) / pipeline.module.output_scale_factor
            if not pipeline.config["guidance_before_res"] and (pipeline.t.cpu() in pipeline.guidance_schedule):
                pipeline.saved_features = rearrange(
                    output_tensor, f"{pipeline.starting_shape} -> b t d h w", t=pipeline.num_frames
                )
            return output_tensor

    class ModuleWithGuidance(torch.nn.Module):
        def __init__(pipeline, module, guidance_schedule, num_frames, h, w, block_name, config, module_type):
            super().__init__()
            pipeline.module = module
            pipeline.guidance_schedule = guidance_schedule
            pipeline.num_frames = num_frames
            assert module_type in [
                "temporal_attention",
                "spatial_attention",
                "temporal_convolution",
                "upsampler",]
            pipeline.module_type = module_type
            if pipeline.module_type == "temporal_attention":
                pipeline.starting_shape = "(b h w) t d"
            elif pipeline.module_type == "spatial_attention":
                pipeline.starting_shape = "(b t) (h w) d"
            elif pipeline.module_type == "temporal_convolution":
                pipeline.starting_shape = "(b t) d h w"
            elif pipeline.module_type == "upsampler":
                pipeline.starting_shape = "(b t) d h w"
            pipeline.h = h
            pipeline.w = w
            pipeline.block_name = block_name
            pipeline.config = config

        def forward(pipeline, x, *args, **kwargs):
            if not isinstance(args, tuple):
                args = (args,)
            out = pipeline.module(x, *args, **kwargs)
            t = pipeline.num_frames
            if pipeline.module_type == "temporal_attention":
                size = out.shape[0] // pipeline.b
            elif pipeline.module_type == "spatial_attention":
                size = out.shape[1]
            elif pipeline.module_type == "temporal_convolution":
                size = out.shape[2] * out.shape[3]
            elif pipeline.module_type == "upsampler":
                size = out.shape[2] * out.shape[3]

            if pipeline.t.cpu() in pipeline.guidance_schedule:
                h, w = int(sqrt(size * pipeline.h / pipeline.w)), int(sqrt(size * pipeline.h / pipeline.w) * pipeline.w / pipeline.h)
                # pipeline.saved_features = rearrange(
                #     out, f"{pipeline.starting_shape} -> b t d h w", t=pipeline.num_frames, h=h, w=w
                # )
                pipeline.saved_features=out #modified0307

            return out

    up_res_dict = kwargs["up_res_dict"]

    for res in up_res_dict:
        module = pipeline.unet.up_blocks[res]
        samplers = module.upsamplers
        if kwargs["use_upsampler_features"]:
            if samplers is not None:
                for i in range(len(samplers)):
                    submodule = samplers[i]
                    samplers[i] = ModuleWithGuidance(
                        submodule,
                        guidance_schedule,
                        num_frames,
                        h,
                        w,
                        block_name=f"decoder_res{res}_upsampler",
                        config=kwargs,
                        module_type="upsampler",
                    )
        for block in up_res_dict[res]:
            block_name = f"decoder_res{res}_block{block}"
            if kwargs["use_conv_features"]:
                block_name_conv = f"{block_name}_spatial_convolution"
                submodule = module.resnets[block] #CrossAttnUpBlock3D.ResnetBlock2D
                module.resnets[block] = ModuleWithConvGuidance(
                    submodule,
                    guidance_schedule,
                    num_frames,
                    h,
                    w,
                    block_name=block_name_conv,
                    config=kwargs,
                    module_type="spatial_convolution",
                )

            if kwargs["use_temp_conv_features"]:
                block_name_conv = f"{block_name}_temporal_convolution"
                submodule = module.temp_convs[block]
                module.temp_convs[block] = ModuleWithGuidance(
                    submodule,
                    guidance_schedule,
                    num_frames,
                    h,
                    w,
                    block_name=block_name_conv,
                    config=kwargs,
                    module_type="temporal_convolution",
                )

            if res == 0:  # UpBlock3D does not have attention
                if kwargs["use_spatial_attention_features"]:
                    block_name_spatial = f"{block_name}_spatial_attn1"
                    submodule = module.attentions[block].transformer_blocks[0]
                    assert isinstance_str(submodule, "BasicTransformerBlock")
                    submodule.attn1 = ModuleWithGuidance(
                        submodule.attn1,
                        guidance_schedule,
                        num_frames,
                        h,
                        w,
                        block_name=block_name_spatial,
                        config=kwargs,
                        module_type="spatial_attention",
                    )
                if kwargs["use_temporal_attention_features"]:
                    submodule = module.temp_attentions[block].transformer_blocks[0]
                    assert isinstance_str(submodule, "BasicTransformerBlock")
                    block_name_temp = f"{block_name}_temporal_attn1"
                    submodule.attn1 = ModuleWithGuidance(
                        submodule.attn1,
                        guidance_schedule,
                        num_frames,
                        h=h,
                        w=w,
                        block_name=block_name_temp,
                        config=kwargs,
                        module_type="temporal_attention",)
                    block_name_temp = f"{block_name}_temporal_attn2"
                    submodule.attn2 = ModuleWithGuidance(
                        submodule.attn2,
                        guidance_schedule,
                        num_frames,
                        h=h,
                        w=w,
                        block_name=block_name_temp,
                        config=kwargs,
                        module_type="temporal_attention",)


# def guidance_step(pipeline, x, i, t,text_embeddings_source,text_embeddings,**args):
#     register_batch(pipeline, 1)
#     module_names = ["ModuleWithConvGuidance", "ModuleWithGuidance"]
#     scaler = GradScaler()
#     change_mode(pipeline,train=True,**args)
#     optimized_x = x.clone().detach().requires_grad_(True)

#     if args["with_lr_decay"]:
#         lr = pipeline.scale_range[i]
#     else:
#         lr = pipeline.optim_lr

#     optimizer = torch.optim.Adam([optimized_x], lr=lr)#这里是有调整x的？？？？？

#     latents = x

#     with torch.no_grad():
#         # latent features
#         orig_features_pos = {}
#         with torch.autocast(device_type="cuda", dtype=torch.float16):
#             pipeline.unet(latents, t, encoder_hidden_states=text_embeddings_source, return_dict=False)[0]
#             # print("finished getting orig_features_pos")
#         for _, module in pipeline.unet.named_modules():
#             if isinstance_str(module, module_names):
#                 orig_features_pos[module.block_name] = module.saved_features[0].cpu()

#     for _ in tqdm(range(2)):
#         optimizer.zero_grad()
#         # target features
#         target_features_pos = {}
#         # print(torch.cuda.memory_summary())
#         optimized_x = optimized_x.float().to('cpu')
#         t = t.float().to('cpu')
#         text_embeddings = text_embeddings.float().to('cpu')
#         cpu_model = pipeline.unet.cpu().float()
#         cpu_model(optimized_x, t, encoder_hidden_states=text_embeddings.detach(), return_dict=False)[0]
#         # pipeline.unet(optimized_x, t, encoder_hidden_states=text_embeddings.detach(), return_dict=False)[0]
#         for _, module in pipeline.unet.named_modules():
#             if isinstance_str(module, module_names):
#                 print("module_name: ",module_names)
#                 target_features_pos[module.block_name] = module.saved_features[0]

#         losses_log = {}
#         total_loss = 0
#         # total_loss.to("cpu")

#         for (orig_name, orig_features), (target_name, target_features) in zip(
#             orig_features_pos.items(), target_features_pos.items()
#         ):
#             assert orig_name == target_name

#             losses = calculate_losses(orig_features.detach(), target_features, **args)
#             for key, value in losses.items():
#                 losses_log[f"Loss/{orig_name}/{key}/time_step{t.item()}"] = value.item()
#             total_loss += losses["total_loss"]

#         losses_log[f"Loss/total_loss/time_step{t.item()}"] = total_loss.item()
#         total_loss.backward()
#         # scaler.scale(total_loss).backward()
#         optimizer.step()
#         # scaler.step(optimizer)
#         # scaler.update()
#         del losses_log, total_loss, losses, target_features_pos
#         for _, module in pipeline.unet.named_modules():
#             if isinstance_str(module, module_names):
#                 module.saved_features = None

#     optimized_x = optimized_x.to(torch.float16)
#     t = t.to(torch.float16)
#     text_embeddings = text_embeddings.to(torch.float16)
#     cpu_model = pipeline.unet.to(torch.float16)
#     optimized_x=optimized_x.to("cuda")
#     t=t.to("cuda")
#     text_embeddings=text_embeddings.to("cuda")
#     cpu_model=cpu_model.to("cuda")

#     return optimized_x


def guidance_step(pipeline, x, i, t,text_embeddings_source,text_embeddings,**args):
    register_batch(pipeline, 1)
    module_names = ["ModuleWithConvGuidance", "ModuleWithGuidance"]
    scaler = GradScaler()
    change_mode(pipeline,train=True,**args)
    optimized_x = x.clone().detach().float().requires_grad_(True)

    if args["with_lr_decay"]:
        lr = pipeline.scale_range[i]
    else:
        lr = pipeline.optim_lr

    # print("x.type: ",optimized_x.dtype)
    optimizer = torch.optim.Adam([optimized_x], lr=lr)#这里是有调整x的？？？？？

    latents_path="/mnt/workspace/cailingling/code/diffusion-motion-transfer/data/dog_sit/ddim_latents"
    latents_t_path = os.path.join(latents_path, f"noisy_latents_{t}.pt")
    assert os.path.exists(latents_t_path), f"Missing latents at t {t} path {latents_t_path}"
    latents = torch.load(latents_t_path).float()
    latents=torch.cat([latents] * 2)
    # latents = x.float()#问题出在了这里 

    with torch.no_grad():
        # latent features
        orig_features_pos = {}
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            pipeline.unet(latents, t, encoder_hidden_states=text_embeddings_source, return_dict=False)[0]
            # print("finished getting orig_features_pos")
        for _, module in pipeline.unet.named_modules():
            if isinstance_str(module, module_names):
                orig_features_pos[module.block_name] = module.saved_features[0]

    for _ in tqdm(range(10)):
        optimizer.zero_grad()
        # target features
        target_features_pos = {}
        # print(torch.cuda.memory_summary())
        # optimized_x = optimized_x.float().to('cpu')
        # t = t.float().to('cpu')
        # text_embeddings = text_embeddings.float().to('cpu')
        cpu_model = pipeline.unet
        cpu_model(optimized_x, t, encoder_hidden_states=text_embeddings.detach(), return_dict=False)[0]
        # pipeline.unet(optimized_x, t, encoder_hidden_states=text_embeddings.detach(), return_dict=False)[0]
        for _, module in pipeline.unet.named_modules():
            if isinstance_str(module, module_names):
                # print("module_name: ",module_names)
                target_features_pos[module.block_name] = module.saved_features[0]

        losses_log = {}
        total_loss = 0
        # total_loss.to("cpu")

        for (orig_name, orig_features), (target_name, target_features) in zip(
            orig_features_pos.items(), target_features_pos.items()
        ):
            assert orig_name == target_name

            losses = calculate_losses(orig_features.detach(), target_features, **args)
            for key, value in losses.items():
                losses_log[f"Loss/{orig_name}/{key}/time_step{t.item()}"] = value.item()
            total_loss += losses["total_loss"]

        losses_log[f"Loss/total_loss/time_step{t.item()}"] = total_loss.item()
        # # total_loss.backward()
        # total_loss=total_loss.float()
        # optimized_x=optimized_x.float()
        total_loss=total_loss.float()
        # optimized_x=optimized_x.float()
        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # # 假设 total_loss 是你的损失值
        # scaler.scale(total_loss).backward()  # 缩放损失并进行反向传播
        # optimizer.step()  # 这一步会自动处理梯度的缩放和反缩放
        # # 更新缩放因子
        # scaler.update()
        # # 清除梯度
        # optimizer.zero_grad()

        del losses_log, total_loss, losses, target_features_pos
        for _, module in pipeline.unet.named_modules():
            if isinstance_str(module, module_names):
                module.saved_features = None

    # optimized_x = optimized_x.to(torch.float16)
    # t = t.to(torch.float16)
    # text_embeddings = text_embeddings.to(torch.float16)
    # cpu_model = pipeline.unet.to(torch.float16)
    # optimized_x=optimized_x.to("cuda")
    # t=t.to("cuda")
    # text_embeddings=text_embeddings.to("cuda")
    # cpu_model=cpu_model.to("cuda")
    optimized_x=optimized_x.to(torch.float16)
    return optimized_x
# def guidance_step(pipeline, x, i, t,text_embeddings_source,text_embeddings,**args):
#     register_batch(pipeline, 1)
#     module_names = ["ModuleWithConvGuidance", "ModuleWithGuidance"]
#     scaler = GradScaler()
#     change_mode(pipeline,train=True,**args)
#     optimized_x = x.clone().detach().requires_grad_(True)

#     if args["with_lr_decay"]:
#         lr = pipeline.scale_range[i]
#     else:
#         lr = pipeline.optim_lr

#     optimizer = torch.optim.Adam([optimized_x], lr=lr)#这里是有调整x的？？？？？

#     latents = x

#     with torch.no_grad():
#         # latent features
#         orig_features_pos = {}
#         with torch.autocast(device_type="cuda", dtype=torch.float16):
#             pipeline.unet(latents, t, encoder_hidden_states=text_embeddings_source, return_dict=False)[0]
#             print("finished getting orig_features_pos")
#         for _, module in pipeline.unet.named_modules():
#             if isinstance_str(module, module_names):
#                 orig_features_pos[module.block_name] = module.saved_features[0].cpu()

#     for _ in tqdm(range(2)):
#         optimizer.zero_grad()

#         # target features
#         target_features_pos = {}
#         # print(torch.cuda.memory_summary())
#         optimized_x = optimized_x.to('cuda:1')
#         t = t.to('cuda:1')
#         text_embeddings = text_embeddings.to('cuda:1')
#         cpu_model = pipeline.unet.to("cuda:1")
#         cpu_model(optimized_x, t, encoder_hidden_states=text_embeddings.detach(), return_dict=False)[0]
#         for _, module in pipeline.unet.named_modules():
#             if isinstance_str(module, module_names):
#                 target_features_pos[module.block_name] = module.saved_features[0]

#         losses_log = {}
#         total_loss = 0
#         # total_loss.to("cpu")

#         for (orig_name, orig_features), (target_name, target_features) in zip(
#             orig_features_pos.items(), target_features_pos.items()
#         ):
#             assert orig_name == target_name

#             losses = calculate_losses(orig_features.detach(), target_features, **args)
#             for key, value in losses.items():
#                 losses_log[f"Loss/{orig_name}/{key}/time_step{t.item()}"] = value.item()
#             total_loss += losses["total_loss"]

#         losses_log[f"Loss/total_loss/time_step{t.item()}"] = total_loss.item()
#         total_loss.backward()
#         # scaler.scale(total_loss).backward()
#         optimizer.step()
#         # scaler.step(optimizer)
#         # scaler.update()
#         del losses_log, total_loss, losses, target_features_pos
#         for _, module in pipeline.unet.named_modules():
#             if isinstance_str(module, module_names):
#                 module.saved_features = None
#     # optimized_x = optimized_x.to(torch.float16)
#     # t = t.to(torch.float16)
#     # text_embeddings = text_embeddings.to(torch.float16)
#     # cpu_model = pipeline.unet.to(torch.float16)
#     optimized_x=optimized_x.to("cuda:0")
#     t=t.to("cuda:0")
#     text_embeddings=text_embeddings.to("cuda:0")
#     cpu_model=cpu_model.to("cuda:0")
#     return optimized_x



#add motion transfer
def change_mode(pipeline, train=True,**args):
    if len(args["up_res_dict"]) != 0:
        index = max(args["up_res_dict"].keys())
        for i, block in enumerate(pipeline.unet.up_blocks):
            if i > index:
                if train:
                    pipeline.unet.up_blocks[i].original_forward = pipeline.unet.up_blocks[i].forward
                    pipeline.unet.up_blocks[i].forward = pipeline.my_pass
                else:
                    pipeline.unet.up_blocks[i].forward = pipeline.unet.up_blocks[i].original_forward

    if pipeline.unet.conv_norm_out:
        if train:
            pipeline.unet.conv_norm_out.original_forward = pipeline.unet.conv_norm_out.forward
            pipeline.unet.conv_norm_out.forward = pipeline.my_pass
        else:
            pipeline.unet.conv_norm_out.forward = pipeline.unet.conv_norm_out.original_forward

        if train:
            pipeline.unet.conv_act.original_forward = pipeline.unet.conv_act.forward
            pipeline.unet.conv_act.forward = pipeline.my_pass
        else:
            pipeline.unet.conv_act.forward = pipeline.unet.conv_act.original_forward

    if train:
        pipeline.unet.conv_out.original_forward = pipeline.unet.conv_out.forward
        pipeline.unet.conv_out.forward = pipeline.my_pass
    else:
        pipeline.unet.conv_out.forward = pipeline.unet.conv_out.original_forward

    def check_inputs(pipeline, prompt, height, width, callback_steps, strength=None):
        if not isinstance(prompt, str) and not isinstance(prompt, list):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        if strength is not None:
            if strength <= 0 or strength > 1:
                raise ValueError(f"The value of strength should in (0.0, 1.0] but is {strength}")

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(
                f"`height` and `width` have to be divisible by 8 but are {height} and {width}."
            )

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )