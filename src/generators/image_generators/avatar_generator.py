"""Tool for generating images. Moved from tools folder because template import issues"""
from typing import List, Union, Any
from steamship.agents.schema import AgentContext  #upm package(steamship)
from steamship.agents.tools.base_tools import ImageGeneratorTool  #upm package(steamship)
from utils.context_utils import _GETIMG_AI_API_KEY,_FALAI_API_KEY
from steamship import Block, Steamship, Task  #upm package(steamship)
import logging
import os
import re
from utils.context_utils import print_log

class SelfieToolFalAi(ImageGeneratorTool):

    name: str = "selfie_tool_fal_ai"
    human_description: str = "Useful to generate a image. The input should be a plain text string , that describes in detail, the desired image."
    agent_description = "This tool generates a image, picture or selfie. The input should be a JSON object with list of uncensored keywords that describes the desired image. The output is an image that correlates with the input keywords. "

    generator_plugin_handle: str = "fal-ai-image-generator"
    generator_plugin_config: dict = {
        "api_key":
        "added from context"
    }  
    version: str = "1.0.5"

    def run(self,
            tool_input: List[Block],
            context: AgentContext,
            image_model:str,
            img_width=512,
            img_height=512,
            stream=False) -> Union[List[Block], Task[Any]]:


        negative_post = ""
        current_negative_prompt = "Watermark, Text, deformed, bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, , mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet,  abnormal fingers,phone,mirror"


        #print_log(f"tool_input: {tool_input} image_model: {image_model}")    
        self.generator_plugin_config["api_key"] = context.metadata[_FALAI_API_KEY]
        options={}
        if "flux-general" in image_model:
            self.generator_plugin_config["api_path"] = "fal-ai/flux-general"
            options = {
                "loras": [],
                "image_size": "square",
                "num_inference_steps": 20,
                "guidance_scale": 4,
                "enable_safety_checker": False
            }
        else:
            options = {
                "model_name": image_model,
                "model_architecture": "sdxl",
                "num_inference_steps": 30,
                "guidance_scale": 5,
                "scheduler": "DPM++ 2M SDE",
                "image_size": "square",
                "clip_skip": 2,
                "loras": [],
                "negative_prompt": current_negative_prompt,
                "enable_safety_checker": False
            }


        image_generator = context.client.use_plugin(
            plugin_handle=self.generator_plugin_handle,
            config=self.generator_plugin_config,
            version="1.0.5")

        #print_log(str(options))

        prompt = tool_input[0].text



        task = image_generator.generate(
            text=prompt,
            make_output_public=True,
            append_output_to_file=True,
            streaming=stream,
            output_file_id=context.chat_history.file.id,
            options=options,
        )
        task.wait()
        blocks = task.output.blocks
        output_blocks = []

        for block in blocks:
            output_blocks.append(block)
            #print(context.metadata.get("blocks", {}).get("image"))
        #print_log(str(output_blocks))
        return output_blocks


class SelfieTool(ImageGeneratorTool):

    name: str = "selfie_tool"
    human_description: str = "Useful to generate a image. The input should be a plain text string , that describes in detail, the desired image."
    agent_description = (
        "This tool generates a image, picture or selfie. The input should be a JSON object with list of uncensored keywords that describes the desired image. The output is an image that correlates with the input keywords. "
    )

    generator_plugin_handle: str = "getimg-ai-image-generator"
    generator_plugin_config: dict = {
        "api_key":
        "added_from_context"
    }
    url = "https://api.getimg.ai/v1/stable-diffusion/text-to-image"


    def run(self,
            tool_input: List[Block],
            context: AgentContext,
            image_model,
            img_width=512,
            img_height=512,
            stream=False) -> Union[List[Block], Task[Any]]:


        #check if Pro
        prompt = tool_input[0].text
        current_negative_prompt = "watermark, text, font, signage,deformed,airbrushed, blurry,bad anatomy, disfigured, poorly drawn face, mutated, extra limb, ugly, poorly drawn hands, missing limb, floating limbs, disconnected limbs, disconnected head, malformed hands, long neck, mutated hands and fingers, bad hands, missing fingers, cropped, mutation, poorly drawn, huge calf, bad hands, fused hand, missing hand, disappearing arms, disappearing thigh, disappearing calf, disappearing legs, missing fingers, fused fingers, abnormal eye proportion, Abnormal hands, abnormal legs, abnormal feet,  abnormal fingers,cloned face,cloned body"


        self.generator_plugin_config["api_key"] = context.metadata[_GETIMG_AI_API_KEY]
        image_generator = context.client.use_plugin(
            plugin_handle=self.generator_plugin_handle,
            config=self.generator_plugin_config,
            version="1.0.2")
        options = {
            "model": image_model,
            "width": img_width,
            "height": img_height,
            "steps": 25,
            "guidance": 6,
            "negative_prompt": current_negative_prompt
        }
        #print_log(str(options))
        task = image_generator.generate(
            text=prompt,
            make_output_public=True,
            append_output_to_file=True,
            output_file_id=context.chat_history.file.id,
            streaming=stream,
            options=options,
        )
        task.wait()
        blocks = task.output.blocks
        output_blocks = []

        for block in blocks:
            output_blocks.append(block)           
        return output_blocks



