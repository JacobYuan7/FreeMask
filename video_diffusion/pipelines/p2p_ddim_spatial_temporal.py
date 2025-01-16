# code mostly taken from https://github.com/huggingface/diffusers

from typing import Callable, List, Optional, Union
import os, sys
import PIL
import torch
import numpy as np
from einops import rearrange
from tqdm import trange, tqdm
from PIL import Image
from transformers import CLIPTextModel, CLIPTokenizer

from diffusers.utils import deprecate, logging
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
from diffusers.models import AutoencoderKL,UNet3DConditionModel
from diffusers.schedulers import KarrasDiffusionSchedulers
from .zeroscope import VideoToVideoSDPipeline
from video_diffusion.prompt_attention import attention_util
from video_diffusion.prompt_attention import keyvalue_util
from video_diffusion.prompt_attention import latent_util
from video_diffusion.prompt_attention.visualization import show_cross_attention, show_self_attention_comp
from video_diffusion.pipelines import motion_utils
logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
import gc



class P2pDDIMSpatioTemporalPipeline(VideoToVideoSDPipeline):
    def __init__(
        self,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        unet: UNet3DConditionModel,
        scheduler: KarrasDiffusionSchedulers,
        disk_store: bool=False
        ):
        super().__init__(vae, text_encoder, tokenizer, unet, scheduler)
        self.store_controller = attention_util.AttentionStore(disk_store=disk_store)
        self.store_controller1 = keyvalue_util.KeyValueStore()
        # self.latent_store_controller=latent_util.LatentStore()
        self.empty_controller = attention_util.EmptyControl()
        
    r"""
    Pipeline for text-to-video generation using Spatio-Temporal Stable Diffusion.
    """

    def check_inputs(self, prompt, height, width, callback_steps, strength=None):
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
    
    @torch.no_grad()
    def prepare_latents_ddim_inverted(self, image, batch_size, num_images_per_prompt, 
                                        text_embeddings,
                                        store_attention=False, prompt=None,
                                        generator=None,
                                        LOW_RESOURCE = True,
                                        save_path = None,
                                        **args
                                      ):
        self.prepare_before_train_loop()
        # print(f"before register  store attention : Current memory cached: {torch.cuda.memory_cached() / (1024 ** 3)} GB")
        if store_attention:
            attention_util.register_attention_control(self, self.store_controller,self.store_controller1)
            # latent_util.register_model_control(self,self.latent_store_controller)
        # print(f" after register  store attention : Current memory cached: {torch.cuda.memory_cached() / (1024 ** 3)} GB")
        resource_default_value = self.store_controller.LOW_RESOURCE
        self.store_controller.LOW_RESOURCE = LOW_RESOURCE  # in inversion, no CFG, record all latents attention
        batch_size = batch_size * num_images_per_prompt
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if isinstance(generator, list):
            init_latents = [
                self.vae.encode(image[i : i + 1]).latent_dist.sample(generator[i]) for i in range(batch_size)
            ]
            init_latents = torch.cat(init_latents, dim=0)
        else:
            init_latents = self.vae.encode(image).latent_dist.sample(generator)

        init_latents = 0.18215 * init_latents

        if batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] == 0:
            # expand init_latents for batch_size
            deprecation_message = (
                f"You have passed {batch_size} text prompts (`prompt`), but only {init_latents.shape[0]} initial"
                " images (`image`). Initial images are now duplicating to match the number of text prompts. Note"
                " that this behavior is deprecated and will be removed in a version 1.0.0. Please make sure to update"
                " your script to pass as many initial images as text prompts to suppress this warning."
            )
            deprecate("len(prompt) != len(image)", "1.0.0", deprecation_message, standard_warn=False)
            additional_image_per_prompt = batch_size // init_latents.shape[0]
            init_latents = torch.cat([init_latents] * additional_image_per_prompt, dim=0)
        elif batch_size > init_latents.shape[0] and batch_size % init_latents.shape[0] != 0:
            raise ValueError(
                f"Cannot duplicate `image` of batch size {init_latents.shape[0]} to {batch_size} text prompts."
            )
        else:
            init_latents = torch.cat([init_latents], dim=0)

        init_latents_bcfhw = rearrange(init_latents, "(b f) c h w -> b c f h w", b=batch_size)
        ddim_latents_all_step = self.ddim_clean2noisy_loop(init_latents_bcfhw, text_embeddings, self.store_controller,self.store_controller1)# here register attention

        if args["cal_maps"]:
            ############ visualization ####################
            prompts =args["editing_prompts"]
            word=args["word"]
            dataset_name=args["dataname"]
            vil = attention_util.VisualizationMask(prompts, word, self.tokenizer,dataset_name)
            self.store_controller.visualization_camap(vil,dataset_name)
            print("finish visualization")
            ###############################################
      


        # print(f"after ddim_latents_all_step : Current memory cached: {torch.cuda.memory_cached() / (1024 ** 3)} GB")#这里陡增 
        if store_attention and (save_path is not None) :
            attention_output=None
            # Detach the controller for safety
            # print(f"before register  empty attention : Current memory cached: {torch.cuda.memory_cached() / (1024 ** 3)} GB")
            attention_util.register_attention_control(self, self.empty_controller)
            # latent_util.register_model_control(self, self.empty_controller)
            # print(f"after register empty attention : Current memory cached: {torch.cuda.memory_cached() / (1024 ** 3)} GB")
        self.store_controller.LOW_RESOURCE = resource_default_value
        
        return ddim_latents_all_step



    #add motion transfer
    @staticmethod
    def my_pass(*args, **kwargs):
        if len(args) == 0:
            return kwargs["hidden_states"]
        else:
            return args[0]
    
    # @torch.no_grad()
    # def ddim_clean2noisy_loop(self, latent, text_embeddings, controller:attention_util.AttentionControl=None):
    #     weight_dtype = latent.dtype
    #     uncond_embeddings, cond_embeddings = text_embeddings.chunk(2)
    #     all_latent = [latent]
    #     latent = latent.clone().detach()
    #     print('Invert clean image to noise latents by DDIM and Unet')
    #     for i in trange(len(self.scheduler.timesteps)):
    #         t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            
    #         # [1, 4, 8, 64, 64] ->  [1, 4, 8, 64, 64])
    #         noise_pred = self.unet(latent, t, encoder_hidden_states=cond_embeddings)["sample"]
            
    #         latent = self.next_clean2noise_step(noise_pred, t, latent)
    #         if controller is not None: controller.step_callback(latent)
    #         all_latent.append(latent.to(dtype=weight_dtype))
        
    #     return all_latent

    @torch.no_grad()
    def ddim_clean2noisy_loop(self, latent, text_embeddings, controller:attention_util.AttentionControl=None,controller1:keyvalue_util.KeyValueControl=None,controller2:latent_util.LatentControl=None):
        weight_dtype = latent.dtype
        uncond_embeddings, cond_embeddings = text_embeddings.chunk(2)
        all_latent = [latent.cpu()]
        latent = latent.clone().detach()
        print('Invert clean image to noise latents by DDIM and Unet')
        for i in trange(len(self.scheduler.timesteps)):
            t = self.scheduler.timesteps[len(self.scheduler.timesteps) - i - 1]
            
            # [1, 4, 8, 64, 64] ->  [1, 4, 8, 64, 64])
            noise_pred = self.unet(sample=latent, timestep=t, encoder_hidden_states=cond_embeddings)["sample"]
            
            latent = self.next_clean2noise_step(noise_pred, t, latent)
            if controller is not None: 
                controller.step_callback(latent)
            if controller1 is not None:
                controller1.step_callback()
            if controller2 is not None:
                controller2.step_callback()
            all_latent.append(latent.cpu().to(dtype=weight_dtype))

        #CUDA clean
        # print(f"before del latent : Current memory cached: {torch.cuda.memory_cached() / (1024 ** 3)} GB")
        del latent, noise_pred,t
        gc.collect()
        torch.cuda.empty_cache()
        # print(f"after del latent : Current memory cached: {torch.cuda.memory_cached() / (1024 ** 3)} GB")

        return all_latent
    
    def next_clean2noise_step(self, model_output: Union[torch.FloatTensor, np.ndarray], timestep: int, sample: Union[torch.FloatTensor, np.ndarray]):
        """
        Assume the eta in DDIM=0
        """
        timestep, next_timestep = min(timestep - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps, 999), timestep
        alpha_prod_t = self.scheduler.alphas_cumprod[timestep] if timestep >= 0 else self.scheduler.final_alpha_cumprod
        alpha_prod_t_next = self.scheduler.alphas_cumprod[next_timestep]
        beta_prod_t = 1 - alpha_prod_t
        next_original_sample = (sample - beta_prod_t ** 0.5 * model_output) / alpha_prod_t ** 0.5
        next_sample_direction = (1 - alpha_prod_t_next) ** 0.5 * model_output
        next_sample = alpha_prod_t_next ** 0.5 * next_original_sample + next_sample_direction
        return next_sample
    
    def get_timesteps(self, num_inference_steps, strength, device):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start
    
    def p2preplace_edit(self, **kwargs):
        # Edit controller during inference
        # The controller must know the source prompt for replace mapping
        
        len_source = {len(kwargs['source_prompt'].split(' '))}
        len_target = {len(kwargs['prompt'].split(' '))}
        equal_length = (len_source == len_target)
        # print(f" len_source: {len_source}, len_target: {len_target}, equal_length: {equal_length}")
        # blend_words=kwargs.get('blend_words', None)
        # if blend_words:
        key_value_replace=kwargs.get("key_value_replace",False)

        edit_controller = attention_util.make_controller(
                            self.tokenizer, 
                            [ kwargs['source_prompt'], kwargs['prompt']],
                            NUM_DDIM_STEPS = kwargs['num_inference_steps'],
                            is_replace_controller=kwargs.get('is_replace_controller', True) and equal_length,
                            cross_replace_steps=kwargs['cross_replace_steps'], 
                            self_replace_steps=kwargs['self_replace_steps'], 
                            temp_replace_steps=kwargs["temp_replace_steps"],
                            blend_words=kwargs.get('blend_words', None),
                            equilizer_params=kwargs.get('eq_params', None),
                            additional_attention_store=self.store_controller,
                            use_inversion_attention = kwargs['use_inversion_attention'],
                            blend_th = kwargs.get('blend_th', (0.3, 0.3)),
                            blend_self_attention = kwargs.get('blend_self_attention', None),
                            blend_latents=kwargs.get('blend_latents', None),
                            save_path=kwargs.get('save_path', None),
                            save_self_attention = kwargs.get('save_self_attention', True),
                            disk_store = kwargs.get('disk_store', False),
                            key_value_replace=key_value_replace
                            )
        if key_value_replace:
            edit_controller1 = keyvalue_util.make_controller(
                                tokenizer=self.tokenizer, 
                                prompts=[ kwargs['source_prompt'], kwargs['prompt']],
                                NUM_DDIM_STEPS = kwargs['num_inference_steps'],
                                cross_replace_steps=kwargs['cross_replace_steps'], 
                                self_temp_steps=kwargs["temp_replace_steps"],
                                self_steps=kwargs["self_replace_steps"],
                                blend_words=kwargs.get('blend_words', None),
                                additional_attention_store=self.store_controller,
                                additional_keyvalue_store=self.store_controller1,
                                blend_th = kwargs.get('blend_th', (0.3, 0.3))
                                )
        else:
            edit_controller1=None
        attention_util.register_attention_control(self, edit_controller,edit_controller1)

        # edit_controller2 = latent_util.make_controller(
        #                     NUM_DDIM_STEPS = kwargs['num_inference_steps'],
        #                     latent_replace_steps=[0,0.6], 
        #                     additional_latent_store=self.latent_store_controller
        #                     )
        # latent_util.register_model_control(self, controller=edit_controller2)
        
        ##########################################add motion guidance 1########################################################
        # if kwargs["motion_guidance"]==True:
        #     motion_utils.clean_memory()
        #     motion_utils.clean_memory()
        #     self.guidance_schedule=self.scheduler.timesteps[0:-40*self.scheduler.order]
        #     self.latents=kwargs["latents"]
        #     # Optimization params:
        #     self.optim_lr = kwargs["optim_lr"]
        #     self.optimization_steps = kwargs["optimization_steps"]
        #     self.scale_range = np.linspace(kwargs["scale_range"][0], kwargs["scale_range"][1], len(self.guidance_schedule))
        #     print(" p2p_config_now[register_flag",kwargs["register_flag"])
        #     if kwargs["register_flag"]==0:
        #         motion_utils.register_guidance(self,**kwargs)
        ##########################################add motion guidance 1########################################################

        # In ddim inferece, no need source prompt
        # print(f"before sdimage output : Current memory cached: {torch.cuda.memory_cached() / (1024 ** 3)} GB")
        sdimage_output = self.sd_ddim_pipeline(
            controller = edit_controller, 
            controller1 = edit_controller1,
            # controller2 = edit_controller2,
            # target_prompt = kwargs['prompts'][1],
            **kwargs)
        if hasattr(edit_controller.latent_blend, 'mask_list'):
            mask_list = edit_controller.latent_blend.mask_list
        else:
            mask_list = None
        if len(edit_controller.attention_store.keys()) > 0:
            # attention_output = attention_util.show_cross_attention(self.tokenizer, kwargs['prompt'], 
            #                                                    edit_controller, 16, ["up", "down"])
            attention_output=None #modified0218
            
            
        else:
            attention_output = None
        dict_output = {
                "sdimage_output" : sdimage_output,
                "attention_output" : attention_output,
                "mask_list" : mask_list,
            }
        attention_util.register_attention_control(self, self.empty_controller)
        # latent_util.register_model_control(self, self.empty_controller)
        return dict_output

    
    
    
   # @torch.no_grad()
    def __call__(self, **kwargs):
        edit_type = kwargs['edit_type']
        assert edit_type in ['save', 'swap', None]
        # print("edit_type: ",edit_type)
        if edit_type is None:
            return self.sd_ddim_pipeline(controller = None, **kwargs)

        if edit_type == 'save':
            del self.store_controller
            self.store_controller = attention_util.AttentionStore()
            attention_util.register_attention_control(self, self.store_controller)
            sdimage_output = self.sd_ddim_pipeline(controller = self.store_controller,**kwargs)
            
            mask_list = None
            attention_output = attention_util.show_cross_attention(self.tokenizer, kwargs['prompt'], self.store_controller, 16, ["up", "down"])
            # attention_util.show_self_attention_comp(self.store_controller, 16, ["up", "down"])

            dict_output = {
                "sdimage_output" : sdimage_output,
                "attention_output"   : attention_output,
                'mask_list':  mask_list
            }

            # Detach the controller for safety
            attention_util.register_attention_control(self, self.empty_controller)
            # latent_util.latent_attention_control(self, self.empty_controller)
            return dict_output
        
        if edit_type == 'swap':
            # 同上，检查缓存内存的使用情况
            # print(f"in swap : Current memory cached: {torch.cuda.memory_cached() / (1024 ** 3)} GB")
            # print(f" in swap: Max memory cached so far: {torch.cuda.max_memory_cached() / (1024 ** 3)} GB")
            return self.p2preplace_edit(**kwargs)

    
    
    def sd_ddim_pipeline(
        self,
        prompt: Union[str, List[str]],
        image: Union[torch.FloatTensor, PIL.Image.Image] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        strength: float = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
        controller: attention_util.AttentionControl = None,
        controller1: keyvalue_util.KeyValueControl = None,
        controller2: latent_util.LatentControl = None,
        **args
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            image (`torch.FloatTensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch, that will be used as the starting point for the
                process. Only used in DDIM or strength<1.0
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            strength (`float`, *optional*, defaults to 1.0):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.            
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        with torch.no_grad():
            # 0. Default height and width to unet
            height = height or self.unet.config.sample_size * self.vae_scale_factor
            width = width or self.unet.config.sample_size * self.vae_scale_factor

            # 1. Check inputs. Raise error if not correct
            self.check_inputs(prompt, height, width, callback_steps, strength)

            # 2. Define call parameters
            batch_size = 1 if isinstance(prompt, str) else len(prompt)
            device = self._execution_device
            # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
            # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
            # corresponds to doing no classifier free guidance.
            do_classifier_free_guidance = guidance_scale > 1.0

            # 3. Encode input prompt
            text_embeddings = self._encode_prompt(
                prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
            )#（2，77，1024）
            # print("text_embeddings.shape: ",text_embeddings.shape)#(2,77,768)
            # ####################### blip #########################################
            # c=torch.load("/mnt/workspace/cailingling/code/DEADiff/c.pt")
            # unconditional_conditioning=torch.load("/mnt/workspace/cailingling/code/DEADiff/uc.pt")
            # # print("len(c): ",len(c))
            # # print("len(unconditional_conditioning): ",len(unconditional_conditioning))
            # # c[0][1]=text_embeddings[1].unsqueeze(0)
            # # c[1][1]=text_embeddings[1].unsqueeze(0)
            # unconditional_conditioning=[[unconditional_conditioning[0],unconditional_conditioning[1]], [unconditional_conditioning[0],unconditional_conditioning[1]]]
            #unconditional_conditioning=[[unconditional_conditioning[0],text_embeddings[0].unsqueeze(0)], [unconditional_conditioning[0],text_embeddings[0].unsqueeze(0)]]
            # if isinstance(c, dict):
            #     assert isinstance(unconditional_conditioning, dict)
            #     c_in = dict()
            #     for k in c:
            #         if isinstance(c[k], list):
            #             c_in[k] = []
            #             for i in range(len(c[k])):
            #                 if isinstance(c[k][i], list):
            #                     c_in_temp = []
            #                     for j in range(len(c[k][i])):
            #                         c_in_temp.append(torch.cat([unconditional_conditioning[k][i][j], c[k][i][j]]))
            #                     c_in[k].append(c_in_temp)
            #                 else:
            #                     c_in[k] = [torch.cat([
            #                         unconditional_conditioning[k][i],
            #                         c[k][i]]) for i in range(len(c[k]))]
            #         else:
            #             c_in[k] = torch.cat([
            #                     unconditional_conditioning[k],
            #                     c[k]])
            # elif isinstance(c, list):
            #     c_in = list()
            #     assert isinstance(unconditional_conditioning, list)
            #     for i in range(len(c)):
            #         if isinstance(c[i], list):
            #             c_in_temp = []
            #             for j in range(len(c[i])):
            #                 if c[i][j] is None:
            #                     c_in_temp.append(None)
            #                 else:
            #                     c_in_temp.append(torch.cat([unconditional_conditioning[i][j], c[i][j]]))
            #             c_in.append(c_in_temp)
            #         else:
            #             c_in.append(torch.cat([unconditional_conditioning[i], c[i]]))
            # else:
            #     c_in = torch.cat([unconditional_conditioning, c])
            
            # def move_tensors_to_cuda(tensors):
            #     if isinstance(tensors, list):
            #         return [move_tensors_to_cuda(tensor) for tensor in tensors]
            #     elif isinstance(tensors, torch.Tensor):
            #         return tensors.to("cuda")
            #     else:
            #         raise TypeError(f"Unsupported type: {type(tensors)}")
            # text_embeddings=move_tensors_to_cuda(c_in)
            # print("load blip text_embeddings")
            #add motion guidance
            # source_prompt=args["source_prompt"]
            # text_embeddings_source = self._encode_prompt(source_prompt, device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt)

            # 4. Prepare timesteps
            self.scheduler.set_timesteps(num_inference_steps, device=device)
            timesteps = self.scheduler.timesteps
            
            if latents is None:
                ddim_latents_all_step = self.prepare_latents_ddim_inverted(
                    image, batch_size, num_images_per_prompt, 
                    text_embeddings,
                    store_attention=False, # avoid recording attention in first inversion
                    generator = generator,
                )
                latents = ddim_latents_all_step[-1]
                latents=latents.to("cuda")
            else:
                ddim_latents_all_step=None

            latents_dtype = latents.dtype

            # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
            extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

            # 7. Denoising loop
            num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i, t in enumerate(tqdm(timesteps)):
                # expand the latents if we are doing classifier free guidance
                latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

                latent_model_input=latent_model_input.cuda()
                # text_embeddings=text_embeddings.cuda()
                # self.unet=self.unet.cuda()

                with torch.no_grad():
                    # print(f"before noise pred : Current memory cached: {torch.cuda.memory_cached() / (1024 ** 3)} GB")
                    # predict the noise residual
                    noise_pred = self.unet(
                        latent_model_input, t, encoder_hidden_states=text_embeddings
                    ).sample.to(dtype=latents_dtype)

                    # perform guidance
                    if do_classifier_free_guidance:
                        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
                        noise_pred = noise_pred_uncond + guidance_scale * (
                            noise_pred_text - noise_pred_uncond
                        )

                    # compute the previous noisy sample x_t -> x_t-1
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs).prev_sample
                    
                    # Edit the latents using attention map
                    if controller is not None: 
                        dtype = latents.dtype
                        latents_new = controller.step_callback(latents)
                        if controller1 is not None:
                            controller1.step_callback()
                        if controller2 is not None:
                            controller2.step_callback()
                        latents = latents_new.to(dtype)
                    # call the callback, if provided
                    if i == len(timesteps) - 1 or (
                        (i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0
                    ):
                        progress_bar.update()
                        if callback is not None and i % callback_steps == 0:
                            callback(i, t, latents)
                    torch.cuda.empty_cache()

        with torch.no_grad():
            # 8. Post-processing
            image = self.decode_latents(latents)

            # 9. Run safety checker
            has_nsfw_concept = None


            # 10. Convert to PIL
            if output_type == "pil":
                # image = self.numpy_to_pil(image)
                images=self.tensor_to_numpy_edit(image)
                pil_images = []
                is_video = (len(images.shape)==5)
                if is_video:
                    for sequence in images:
                        # 判断 x 的类型
                        if isinstance(sequence, torch.Tensor): # 如果 x 是 tensor 类型
                            sequence = sequence.cpu().numpy() # 将 x 转换成 numpy 类型
                        elif isinstance(sequence, np.ndarray): # 如果 x 是 numpy 类型
                            pass # 不需要改变类型
                        else: # 如果 x 不是 tensor 或 numpy 类型
                            print("输入序列的类型不合法") # 打印错误信息
                        pil_images.append(self.numpy_to_pil_edit(sequence))
                else:
                    pil_images.append(self.numpy_to_pil_edit(images))
                image=pil_images

            if not return_dict:
                return (image, has_nsfw_concept)
            torch.cuda.empty_cache()
            return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
    
    @staticmethod
    def tensor_to_numpy_edit(image, b=1):
        image = (image / 2 + 0.5).clamp(0, 1)
        # we always cast to float32 as this does not cause significant overhead and is compatible with bfloa16

        image = image.cpu().float().numpy()
        image = rearrange(image, "b c f h w -> b f h w c", b=b)
        return image

    @staticmethod
    def numpy_to_pil_edit(images):
        """
        Convert a numpy image or a batch of images to a PIL image.
        """
        if images.ndim == 3:
            images = images[None, ...]
        images = (images * 255).round().astype("uint8")
        if images.shape[-1] == 1:
            # special case for grayscale (single channel) images
            pil_images = [Image.fromarray(image.squeeze(), mode="L") for image in images]
        else:
            for image in images:
                pil_images = [Image.fromarray(image) for image in images]

        return pil_images

    def print_pipeline(self, logger):
        print('Overview function of pipeline: ')
        print(self.__class__)

        print(self)
        
        expected_modules, optional_parameters = self._get_signature_keys(self)        
        components_details = {
            k: getattr(self, k) for k in self.config.keys() if not k.startswith("_") and k not in optional_parameters
        }
        import json
        logger.info(str(components_details))
        # logger.info(str(json.dumps(components_details, indent = 4)))
        # print(str(components_details))
        # print(self._optional_components)
        
        print(f"python version {sys.version}")
        print(f"torch version {torch.__version__}")
        print(f"validate gpu status:")
        print( torch.tensor(1.0).cuda()*2)
        os.system("nvcc --version")

        import diffusers
        print(diffusers.__version__)
        print(diffusers.__file__)

        try:
            import bitsandbytes
            print(bitsandbytes.__file__)
        except:
            print("fail to import bitsandbytes")
        # os.system("accelerate env")
        # os.system("python -m xformers.info")
