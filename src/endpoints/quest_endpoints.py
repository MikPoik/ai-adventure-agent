from typing import List,Optional
import textwrap
from steamship import Block, File, Steamship, SteamshipError, Tag
from steamship.agents.schema import AgentContext
from steamship.agents.service.agent_service import AgentService
from steamship.data.tags.tag_constants import RoleTag
from steamship.invocable import post
from steamship.invocable.package_mixin import PackageMixin
from steamship.agents.schema.message_selectors import MessageWindowMessageSelector
from tools.end_quest_tool import EndQuestTool
from tools.start_quest_tool import StartQuestTool
from utils.context_utils import (
    append_chat_intro_messages,
    get_audio_narration_generator,
    get_game_state,
    get_server_settings,
    save_game_state,
)
from utils.error_utils import record_and_throw_unrecoverable_error
from utils.generation_utils import generate_quest_arc
from utils.tags import QuestIdTag, SceneTag, TagKindExtensions,CharacterTag,StoryContextTag,InstructionsTag,QuestTag
from utils.tags import QuestIdTag
import logging

class QuestMixin(PackageMixin):
    """Provides endpoints for Game State."""

    client: Steamship
    agent_service: AgentService

    def __init__(self, client: Steamship, agent_service: AgentService):
        self.client = client
        self.agent_service = agent_service
        
        @post("append_history")
        def append_history(self,
                           prompt: Optional[str] = None,
                           context_id: Optional[str] = None):
            """Append Bolna phonecall messages to the chat history. Need to add tags to the blocks."""
            if prompt:
                try:
                    # Parse the JSON string to extract messages
                    messages = loads(prompt)
                    context = self.agent_service.build_default_context()

                    # Loop through each message and add it to chat history
                    for message in messages:
                        if message['role'] == 'assistant':
                            context.chat_history.append_assistant_message(
                                text=message['content'],
                            tags=[QuestIdTag(QuestTag.CHAT_QUEST)])
                        elif message['role'] == 'user':
                            context.chat_history.append_user_message(
                                text=message['content'],
                            tags=[QuestIdTag(QuestTag.CHAT_QUEST)])
                except Exception as e:
                    logging.warning(
                        "Failed to parse prompt or append to chat history: " +
                        str(e))
            return "OK"

    
    @post("delete_messages")
    def delete_messages(self, context_id="", companionId=""):
        """TODO: check if needs modification"""
        #check history length, catch errors.
        try:
            context = self.agent_service.build_default_context()
            last_user_message = context.chat_history.last_user_message
            last_agent_message = context.chat_history.last_agent_message
            selector = MessageWindowMessageSelector(k=1)
            user_message_count = 0
            assistant_message_count = 0
            for msg in context.chat_history.messages:
                if msg.chat_role == RoleTag.USER:
                    user_message_count += 1
                elif msg.chat_role == RoleTag.ASSISTANT:
                    assistant_message_count += 1

            #Only delete messages after seed
            if context.chat_history and assistant_message_count > 1 and user_message_count > 1:
                context.chat_history.delete_messages(selector)
                return "MESSAGES_DELETED"
            else:
                return "NO_MESSAGES_TO_DELETE"

        except Exception as e:
            logging.warning(str(e))
            return "ERROR_DELETING_MESSAGES"
            
    @post("/clear_history")
    def clear_history(self):
        """Clears the agent's chat history."""
        context = self.agent_service.build_default_context()
        game_state = get_game_state(context)
        if not game_state:
            logging.error("Game state is None, cannot proceed with clearing history.")
            return "GAME_STATE_NOT_FOUND"

        context.chat_history.clear()
        #Re-seed onboarding message
        onboarding_message = game_state.onboarding_message.format(
            player_name=game_state.player.name,
            player_description=game_state.player.description,
            player_appearance=game_state.player.appearance,
            player_personality=game_state.player.personality,
            player_background=game_state.player.background,
            tags=game_state.tags,) 
        
        context.chat_history.append_system_message(
            text=onboarding_message,
            tags=[
                Tag(
                    kind=TagKindExtensions.INSTRUCTIONS,
                    name=InstructionsTag.ONBOARDING,
                ),
                Tag(kind=TagKindExtensions.CHARACTER, name=CharacterTag.NAME),
                Tag(kind=TagKindExtensions.CHARACTER, name=CharacterTag.BACKGROUND),
                Tag(kind=TagKindExtensions.CHARACTER, name=CharacterTag.MOTIVATION),
                Tag(
                    kind=TagKindExtensions.CHARACTER, name=CharacterTag.DESCRIPTION
                ),
                Tag(
                    kind=TagKindExtensions.STORY_CONTEXT,
                    name=StoryContextTag.BACKGROUND,
                ),
                Tag(
                    kind=TagKindExtensions.STORY_CONTEXT, name=StoryContextTag.TONE
                ),
                Tag(
                    kind=TagKindExtensions.STORY_CONTEXT, name=StoryContextTag.VOICE
                ),
            ],
        )
        append_chat_intro_messages(context)
        
        if game_state.player.seed_message:
            #logging.warning(f"Appending seed message: {game_state.player.seed_message}")
            context.chat_history.append_assistant_message(
                text=game_state.player.seed_message,
                tags=[                       
                    Tag(
                        kind=TagKindExtensions.CHARACTER,
                        name=CharacterTag.SEED,
                    ),
                    Tag(kind=TagKindExtensions.CHARACTER,
                        name=CharacterTag.INTRODUCTION),
                    Tag(
                        kind=TagKindExtensions.CHARACTER,
                        name=CharacterTag.INTRODUCTION_PROMPT,
                    ),
                    Tag(
                        kind=TagKindExtensions.INSTRUCTIONS,
                        name=InstructionsTag.QUEST,
                    ),
                    QuestIdTag(QuestTag.CHAT_QUEST)
                    ],
            )
        return "OK"
        
    @post("/generate_quest_arc")
    def generate_quest_arc(self) -> List[dict]:
        context = self.agent_service.build_default_context()
        try:
            game_state = get_game_state(context)
            quest_arc = generate_quest_arc(player=game_state.player, context=context)
            game_state.quest_arc = quest_arc
            save_game_state(game_state, context)
            return [quest_description.dict() for quest_description in quest_arc]
        except BaseException as e:
            record_and_throw_unrecoverable_error(e, context)

    @post("/start_quest")
    def start_quest(self, **kwargs) -> dict:
        """Starts a quest."""
        context = self.agent_service.build_default_context()
        try:
            game_state = get_game_state(context)
            quest_tool = StartQuestTool()
            quest = quest_tool.start_quest(game_state, context)
            return quest.dict()
        except BaseException as e:
            record_and_throw_unrecoverable_error(e, context)

    @post("/end_quest")
    def end_quest(self, **kwargs) -> str:
        """Starts a quest."""
        context = self.agent_service.build_default_context()
        try:
            game_state = get_game_state(context)
            quest_tool = EndQuestTool()
            return quest_tool.end_quest(game_state, context)
        except BaseException as e:
            record_and_throw_unrecoverable_error(e, context)

    @post("/get_quest")
    def get_quest(self, quest_id: str, **kwargs) -> List[dict]:
        """Gets the blocks for an existing quest."""
        context = self.agent_service.build_default_context()
        try:
            blocks = []

            def matches_quest(_block: Block, _quest_id: str) -> bool:
                for tag in _block.tags or []:
                    if QuestIdTag.matches(tag, _quest_id):
                        return True
                return False

            if (
                context.chat_history
                and context.chat_history.file
                and context.chat_history.file.blocks
            ):
                for block in context.chat_history.file.blocks:
                    if matches_quest(block, quest_id):
                        blocks.append(block)

            return [block.dict(by_alias=True) for block in blocks]
        except BaseException as e:
            record_and_throw_unrecoverable_error(e, context)

    @post("/narrate_block")
    def narrate_block(self, block_id: str, **kwargs) -> dict:
        """Returns a streaming narration for a block."""
        block = Block.get(self.client, _id=block_id)
        context = self.agent_service.build_default_context()
        new_block = QuestMixin._narrate_block(block, context)
        return {"url": new_block.to_public_url()}

    @staticmethod
    def _narrate_block(block: Block, context: AgentContext) -> Block:

        # Only narrate if it's actually
        if not block.is_text():
            raise SteamshipError(
                message=f"Block {block.id} is not a text block. Unable to narrate."
            )

        chat_role = block.chat_role
        if chat_role not in [RoleTag.ASSISTANT, RoleTag.USER]:
            raise SteamshipError(
                message=f"Block {block.id} did not have the chat role of assistant or user. Unable to narrate."
            )

        narration_model = get_audio_narration_generator(context)
        file = File.create(context.client, blocks=[])
        generation = narration_model.generate(
            text=block.text,
            make_output_public=True,
            append_output_to_file=True,
            output_file_id=file.id,
            streaming=True,
            tags=[Tag(kind=TagKindExtensions.SCENE, name=SceneTag.NARRATION)],
        )

        return generation.wait().blocks[0]
