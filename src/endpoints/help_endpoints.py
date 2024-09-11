import json
from typing import List,Optional

from steamship import Steamship, SteamshipError,Block
from steamship.agents.service.agent_service import AgentService
from steamship.invocable import post
from steamship.invocable.package_mixin import PackageMixin
from generators.image_generators.avatar_generator import SelfieTool,SelfieToolFalAi

from utils.context_utils import print_log
from utils.generation_utils import generate_action_choices


class HelpMixin(PackageMixin):
    """Provides endpoints for providing help to users."""

    agent_service: AgentService
    client: Steamship

    def __init__(self, client: Steamship, agent_service: AgentService):
        self.client = client
        self.agent_service = agent_service

    @post("/generate_action_choices")
    def generate_action_choices(self, **kwargs) -> List[str]:
        """Generate (synchronously) a JSON List of multiple choice options for user actions in a quest."""
        try:
            context = self.agent_service.build_default_context()
            choices_json_block = generate_action_choices(context=context)
            cleaned_block_text = choices_json_block.text.split("\n\n")[0]
            choices = json.loads(choices_json_block.text)
            choices_data = choices.get("choices", [])
            print(choices_data)
            return choices_data

        except BaseException as e:
            raise SteamshipError(
                "Could not generate next action choices. Please try again.", error=e
            )
            
    @post("/generate_avatar", public=True)
    def generate_avatar(self,
                        prompt: Optional[str] = None,
                        image_model: Optional[str] = None,):
        """Run an agent with the provided text as the input."""
        
        #print_log(f"generate_avatar prompt={prompt} image_model={image_model}")
        with self.agent_service.build_default_context() as context:
   
            get_img_ai_models = [
                "realistic-vision-v3", "dark-sushi-mix-v2-25",
                "absolute-reality-v1-8-1", "van-gogh-diffusion",
                "neverending-dream", "mo-di-diffusion", "synthwave-punk-v2",
                "dream-shaper-v8", "arcane-diffusion"
            ]

            
            if image_model is not None:
                if image_model in get_img_ai_models:
                    #print_log(f"generate_avatar image_model={image_model}")
                    selfie_tool = SelfieTool()
                    selfie_response = selfie_tool.run([Block(text=prompt)],
                                                      context=context,
                                                      image_model=image_model,
                                                        img_height=512,
                                                        img_width=512,
                                                        stream=False
                                                     )
                    return selfie_response


                if "flux-general" in image_model or "civitai.com" in image_model:
                    #print_log(f"generate_avatar Fal.ai FLUX image_model={image_model}")
                    selfie_tool = SelfieToolFalAi()
                    selfie_response = selfie_tool.run([Block(text=prompt)],
                                                      context=context,
                                                      image_model=image_model,
                                                      img_height=512,
                                                      img_width=512,
                                                      stream=False)
                    return selfie_response
                    
                    
            print_log("Error parsing image model")
            return "Error parsing image model"