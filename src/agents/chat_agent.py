import json
import logging
from datetime import datetime, timezone
from enum import Enum
from random import randint, random
import textwrap
from typing import Dict, List

from pydantic.utils import Representation
from steamship import SteamshipError, Tag
from steamship.agents.logging import AgentLogging
from steamship.agents.schema import Action, AgentContext
from steamship.agents.schema.action import FinishAction

from generators.generator_context_utils import (
    get_music_generator,
    get_quest_background_image_generator,
    get_chat_image_generator
)
from schema.game_state import GameState
from schema.quest import Quest, QuestChallenge, QuestDescription
from schema.server_settings import Difficulty
from tools.end_quest_tool import EndQuestTool
from utils.context_utils import (
    FinishActionException,
    await_ask,
    get_current_quest,
    get_game_state,
    get_server_settings,
    save_game_state,
)
from utils.generation_utils import (
    await_streamed_block,
    generate_is_solution_attempt,
    generate_likelihood_estimation,
    generate_quest_arc,
    send_story_generation,
)
from utils.interruptible_python_agent import InterruptiblePythonAgent
from utils.moderation_utils import mark_block_as_excluded
from utils.tags import InstructionsTag, QuestIdTag, QuestTag, TagKindExtensions



class ChatAgent(InterruptiblePythonAgent):
    """
    The Chat agent goes on a chat!

    HOW THIS AGENT IS ACTIVATED
    ===========================

    The game log defers to this agent when `game_state.current_quest` is not None.

    The `game_state.current_quest` argument matches `game_state.quests[].name` and is used to provide the
    Quest object to this agent at construction time so that it has a handle on where to load/store state.

    WHAT CAUSES THAT ACTIVATION TO HAPPEN
    =====================================

    The `use_settings.current_quest` string is set to not-None when the following things happen:

    - POST /start_quest (See the quest_mixin)
    - maybe later: The Camp Agent runs the Start Quest Tool

    It can be slotted into as a state machine sub-agent by the overall agent.
    """
    
    def run(self, context: AgentContext) -> Action:  # noqa: C901
        """
        It could go in a tool, but that doesn't feel necessary... there are some other spots where tools feel very
        well fit, but this might be better left open-ended, so we can stop/start things as we like.
        """
        
        # Load the main things we're working with. These can modified and the save_game_state called at any time
        game_state = get_game_state(context)
        player = game_state.player
        quest = get_current_quest(context)
        server_settings = get_server_settings(context)


        logging.debug(
            "Running Chat Agent",
            extra={
                AgentLogging.IS_MESSAGE: True,
                AgentLogging.MESSAGE_TYPE: AgentLogging.AGENT,
                AgentLogging.MESSAGE_AUTHOR: AgentLogging.TOOL,
                AgentLogging.AGENT_NAME: self.__class__.__name__,
            },
        )
        
        if not game_state.chat_intro_complete:
            #user_prompt = await_ask(
                #f"What do you say next?",
                #context,
                #key_suffix=
                #f"user input {quest.name}"

            #)
            game_state.chat_intro_complete = True
            user_prompt =""
            if context.chat_history and context.chat_history.last_user_message:
                if context.chat_history.last_user_message.text:
                    user_prompt = context.chat_history.last_user_message.text
                
            save_game_state(game_state, context)
            
            
            block = send_story_generation(
            prompt=user_prompt,
            quest_name=quest.name,
            context=context,
            )
            await_streamed_block(block, context)
        

        if game_state.chat_mode and game_state.chat_intro_complete:      
            
            user_prompt = await_ask(
                f"What do you say next?",
                context,
                key_suffix=
                f"user input {quest.name}"

            )
            save_game_state(game_state, context)
            additional_info = ""
            if server_settings.enable_images_in_chat:                
                #TODO ask llm if we want to create image based on input
                response_plan = self.generate_plan(game_state, context, quest,user_prompt=user_prompt)
                logging.warning(response_plan)
                if not "NoImage" in response_plan:
                    #TODO: add image generation
                
                    if image_gen := get_chat_image_generator(context):
                        logging.warning("Generating chat image")
                        task=image_gen.request_chat_image_generation(
                        description=response_plan, context=context
                    )
                    additional_info=". Note: You agreed to send the image already and have the image ready, so respond accordingly"
                
            
            response_block = self.respond_to_user(game_state,context,quest,user_prompt=user_prompt,additional_context=additional_info)
            #logging.warning(response_block.text)
            
            
            user_prompt = await_ask(
                f"What do you say next?",
                context,
                key_suffix=
                f"user input {quest.name}"

            )
        
        blocks = []
        return FinishAction(output=blocks)


#       *** END RUN FUNCTION ***    

    
    def tags(self, part: QuestTag, quest: "Quest") -> List[Tag]:  # noqa: F821
        return [
            Tag(kind=TagKindExtensions.QUEST, name=part),
            QuestIdTag(quest.name)
        ]

    def respond_to_user(
        self,
        game_state: GameState,
        context: AgentContext,
        quest: Quest,
        user_prompt: str = None,
        additional_context: str = "",
    ):
        prompt = f"{user_prompt}{additional_context}"
        solution_block = send_story_generation(
            prompt=prompt,            
            quest_name=quest.name,
            context=context,
            additional_context=additional_context, 
        )
        return await_streamed_block(solution_block, context)


    def generate_plan(self, game_state: GameState, context: AgentContext,
    quest: Quest,user_prompt:str):

        prompt = textwrap.dedent(f"""\
        ### Instruction
        Switch to function mode. 
        If the user is asking for an image/selfie/visual of {game_state.player.name},
        generate an matching image description keywords in square brackets, otherwise leave empty square brackets.
        Respond only with given format, nothing else is necessary. This response is not included in chat log.
        Respond using this format:
        `[NoImage]` or `[Image: insert detailed image description keywords here for the image]`
        
        ### Input:
        Last input from user to {game_state.player.name} was: "{user_prompt}"
        
        ### Response:
        [NoImage]/[Image]: """)

        is_solution_attempt_response = generate_is_solution_attempt(
    prompt=prompt,
    quest_name=quest.name,
    context=context,
    )
        logging.warning("problem prompt: " +prompt)
        logging.warning(f"Plan response: {is_solution_attempt_response.text}")
        return is_solution_attempt_response.text.strip()

