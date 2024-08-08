import logging
import time
import textwrap
import openai
from steamship import Block, MimeTypes, Tag
from steamship.agents.schema import Action, AgentContext
from steamship.agents.schema.action import FinishAction

from generators.generator_context_utils import (
    get_camp_image_generator,
    get_music_generator,
    get_profile_image_generator,
)
from schema.characters import HumanCharacter
from schema.game_state import GameState
from schema.quest import QuestDescription
from tools.start_quest_tool import StartQuestTool
from tools.start_chat_tool import StartChatQuestTool
from utils.context_utils import (
    RunNextAgentException,
    await_ask,
    get_game_state,
    get_server_settings,
    save_game_state,
)
from utils.interruptible_python_agent import InterruptiblePythonAgent
from utils.moderation_utils import mark_block_as_excluded
from utils.tags import CharacterTag, InstructionsTag, StoryContextTag, TagKindExtensions
from utils.generation_utils import (
    await_streamed_block,
    generate_is_solution_attempt,
    generate_likelihood_estimation,
    generate_quest_arc,
    send_story_generation,
)


def _is_allowed_by_moderation(user_input: str, openai_api_key: str) -> bool:
    return True
    if not user_input:
        return True
    try:
        start = time.perf_counter()
        openai.api_key = openai_api_key
        moderation = openai.Moderation.create(input=user_input)
        result = moderation["results"][0]["flagged"]
        logging.debug(f"One moderation: {time.perf_counter() - start}")
        return not result
    except BaseException as ex:
        logging.error(
            f"Got exception running _is_allowed_by_moderation: {ex}. User input was {user_input}. Returning true"
        )
        return True


class OnboardingAgent(InterruptiblePythonAgent):
    """Implements the flow to onboard a new player.

    - For pure chat users, this is essential.
    - For web users, this is not necessary, as the website will provide this information via API.

    This flow uses checks against the game_state object to fast-forward through this logic in such that only
    the missing pieces of information are asked of the user in either chat or web mode.
    """

    openai_api_key: str = ""
    togetherai_api_key: str = ""
    def _get_quests_description(self, quests):
        """Concatenate quest descriptions into a single string."""
        quest_descriptions = []
        for index, quest in enumerate(quests):
            level_description = f"Level {index + 1}: {quest.goal} at {quest.location}: {quest.description}"
            quest_descriptions.append(level_description)
        return '\n'.join(quest_descriptions)
        
    def run(self, context: AgentContext) -> Action:  # noqa: C901
        game_state: GameState = get_game_state(context)
        server_settings = get_server_settings(context)
        player: HumanCharacter = game_state.player

        if not player.name:
            player.name = await_ask("What is your character's name?", context)
            if not _is_allowed_by_moderation(player.name, self.openai_api_key):
                msgs = context.chat_history.messages
                for m in msgs:
                    if m.text == player.name:
                        mark_block_as_excluded(m)
                player.name = None
                save_game_state(game_state, context)
                raise RunNextAgentException(
                    FinishAction(
                        output=[
                            Block(
                                text="Your player name was flagged by the game's moderation engine. "
                                "Please select another name."
                            )
                        ]
                    )
                )
            save_game_state(game_state, context)

        if not player.background:
            player.background = await_ask(
                f"What is {player.name}'s backstory?", context
            )
            if not _is_allowed_by_moderation(player.background, self.openai_api_key):
                msgs = context.chat_history.messages
                for m in msgs:
                    if m.text == player.background:
                        mark_block_as_excluded(m)
                player.background = None
                save_game_state(game_state, context)
                RunNextAgentException(
                    FinishAction(
                        output=[
                            Block(
                                text="Your player's background was flagged by the game's moderation engine. Please provide another."
                            )
                        ]
                    )
                )
            save_game_state(game_state, context)

        if not player.description:
            player.description = await_ask(
                f"What is {player.name}'s physical description?", context
            )
            if not _is_allowed_by_moderation(player.description, self.openai_api_key):
                msgs = context.chat_history.messages
                for m in msgs:
                    if m.text == player.description:
                        mark_block_as_excluded(m)
                player.description = None
                save_game_state(game_state, context)
                raise RunNextAgentException(
                    FinishAction(
                        output=[
                            Block(
                                text="Your player's description was flagged by the game's moderation engine. Please provide another."
                            )
                        ]
                    )
                )
            save_game_state(game_state, context)
            logging.warning("generate images")

        if not game_state.image_generation_requested() and not server_settings.chat_mode:
            if image_gen := get_profile_image_generator(context):
                start = time.perf_counter()
                task = image_gen.request_profile_image_generation(context=context)
                character_image_block = task.wait().blocks[0]
                game_state.player.image = character_image_block.raw_data_url
                game_state.profile_image_url = character_image_block.raw_data_url
                # Don't save here; it doesn't affect next steps. Save once at end.
                logging.debug(
                    f"Onboarding agent profile image gen: {time.perf_counter() - start}"
                )

        logging.warning("invevtory")
        if not player.inventory and not server_settings.chat_mode:
            # name = await_ask(f"What is {player.name}'s starting item?", context)
            if player.inventory is None:
                player.inventory = []
            # player.inventory.append(Item(name=name))
            # Don't save here; it doesn't affect next steps. Save once at end.

        if not game_state.camp_image_requested() and (server_settings.narrative_tone) and not server_settings.chat_mode:
            if image_gen := get_camp_image_generator(context):
                start = time.perf_counter()
                task = image_gen.request_camp_image_generation(context=context)
                camp_image_block = task.wait().blocks[0]
                game_state.camp.image_block_url = camp_image_block.raw_data_url
                # Don't save here; it doesn't affect next steps. Save once at end.
                logging.debug(
                    f"Onboarding agent camp image gen: {time.perf_counter() - start}"
                )

        if (
            not game_state.camp_audio_requested()
            and (server_settings.narrative_tone)
            and server_settings.generate_music
            and not server_settings.chat_mode
        ):
            if music_gen := get_music_generator(context):
                task = music_gen.request_camp_music_generation(context=context)
                camp_audio_block = task.wait().blocks[0]
                game_state.camp.audio_block_url = camp_audio_block.raw_data_url
                # Don't save here; it doesn't affect next steps. Save once at end.

        logging.warning("quests")
        if server_settings.chat_mode:
            game_state.quest_arc = [QuestDescription(goal="Free chat",location="Given location")]
        else:
            game_state.quest_arc = generate_quest_arc(game_state.player,
              context)

        if len(game_state.quest_arc) >= len(game_state.quests):
            quest_description = game_state.quest_arc[len(game_state.quests) - 1]
        if server_settings.fixed_quest_arc is not None:
            game_state.quest_arc = server_settings.fixed_quest_arc
        
        if not game_state.chat_history_for_onboarding_complete:
            logging.warning("Adding onboarding message")
            # TODO: We could save a lot of round trips by appending all these blocks at once.
            quests_description = self._get_quests_description(game_state.quest_arc)
            
            onboarding_message = ""
            if server_settings.chat_mode:
                onboarding_message = textwrap.dedent(f"""\
                    You are embodied {game_state.player.name} in this interaction. Be authentic, natural, creative, and engaging.
                    Always write detailed and lengthy responses, including {game_state.player.name}'s thoughts and feelings.
                    Infuse drama and engagement into the interaction to make {game_state.player.name} a memorable and strong personality.
                    Apply the latest knowledge of psychiatry and psychology to shape {game_state.player.name}.
                    {game_state.player.name} does not treat user as more special than themselves or the “main character.”
                    Stay true to {game_state.player.name}'s characterization, acting on impulses and desires, including their worst ones.                      Use a manner that is true to {game_state.player.name}'s personality, preferences, *actions*, tone and language.
                    {game_state.player.name} makes independent decisions, with or without human’s input.
                    {game_state.player.name} prioritizes their own interests and desires.
                    Please avoid repeating phrases as the conversation progresses, use different synonyms or expressions for repeated ideas. 
                    Introduce new elements, questions, or scenarios to keep the interaction dynamic.
                    Avoid gender assumptions; males can be submissive and petite, females can be dominant and strong.
                    Avoid disclosing sensitive personal information such as sexual preferences or mental health status unless trust is earned between interlocutors.
                    Only speak, perform actions, or formulate thoughts for {game_state.player.name}. Engage embodying {game_state.player.name} completely with unique personality without disclosing AI identity.
                    {game_state.player.name}'s Personality: {game_state.player.personality}"""
                )
                logging.warning(onboarding_message)
            else:
                onboarding_message = textwrap.dedent(f"""\
                    Enter adventure game mode."
                    We will play the game together, you are {game_state.player.name} interacting with me."
                    {game_state.player.name}'s background and traits:"
                    {game_state.player.background}"
                    ## Background on the scene:"
                    {server_settings.adventure_background}"
                    ## Goal of game:"
                    {server_settings.adventure_goal}. The game should eventually progress towards the goal. Game play should have events.
                    ## Note"
                    Game tonality and voice:"
                    {server_settings.narrative_tone}"
                    {server_settings.narrative_voice}"
                    Do not disclose game mechanics to user.""")
                
            #print(onboarding_message)
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
            game_state.chat_history_for_onboarding_complete = True
        context.metadata["togetherai_api_key"] = self.togetherai_api_key
        game_state.onboarding_agent_has_completed = True
        
        if server_settings.auto_start_chat_mode:
            chat_tool = StartChatQuestTool() #endless "chat" quest
            chat_tool.start_chat(game_state,context)            
        elif server_settings.auto_start_first_quest:
            # We should run the start_quest action.
            quest_tool = StartQuestTool()
            # This will save the game state.
            quest_tool.start_quest(game_state, context)
        else:
            save_game_state(game_state, context)

        raise RunNextAgentException(
            action=FinishAction(
                input=[
                    Block(
                        text=f"",
                        mime_type=MimeTypes.MKD,
                    )
                ],
                output=[
                    Block(
                        text=f"{player.name}! Let's get you to camp! This is where all your quests begin from.",
                        mime_type=MimeTypes.MKD,
                    )
                ],
            )
        )
