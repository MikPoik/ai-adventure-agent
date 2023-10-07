import uuid
from typing import Any, List, Optional, Union

from steamship import Block, Task
from steamship.agents.schema import AgentContext, ChatHistory, Tool

from schema.game_state import GameState
from schema.quest import Quest
from utils.context_utils import get_game_state, save_game_state


class StartQuestTool(Tool):
    """Starts a quest.

    This Tool is meant to TRANSITION from the CAMP AGENT to the QUEST AGENT.

    It can either be called by:
     - The CAMP AGENT (when in full-chat mode) -- see camp_agent.py
     - The WEB APP (when in web-mode, via api) -- see quest_mixin.py

    This Tool does the following things:
    - Creates a new QUEST and CHAT HISTORY
    - Sets the GAME STATE to that QUEST
    - Seeds the CHAT HISTORY with system messages related to the overall game GAME STATE.

    That's it. All other interactions, which may involve asking the user questions or dynamically generating assets,
    are handled by the QUEST AGENT. This includes naming the quest, picking a location, etc etc.

    That keeps tools simple -- as objects whose purpose is to transition -- and leaves the Agents as objects that
    have more complex logic with conditionals / user-interrupt behavior / etc.
    """

    def __init__(self, **kwargs):
        kwargs["name"] = "StartQuestTool"
        kwargs[
            "agent_description"
        ] = "Use when the user wants to go on a quest. The input is the kind of quest, if provided. The output is the Quest Name"
        kwargs[
            "human_description"
        ] = "Tool to initiate a quest. Modifies the global state such that the next time the agent is contacted, it will be on a quets."
        # It always returns.. OK! Let's go!
        kwargs["is_final"] = True
        super().__init__(**kwargs)

    def start_quest(
        self,
        game_state: GameState,
        context: AgentContext,
        purpose: Optional[str] = None,
    ) -> Quest:
        quest = Quest(chat_file_id=f"quest-{uuid.uuid4()}")
        player = game_state.player

        chat_history = ChatHistory.get_or_create(
            context.client, {"id": quest.chat_file_id}
        )
        chat_history.append_system_message(
            f"We are writing a story about the adventure of a character named {player.name}."
        )
        chat_history.append_system_message(
            f"{player.name} has the following background: {player.background}"
        )

        # Add in information about pinventory
        items = []
        for item in player.inventory:
            items.append(item.name)
        if len(items) > 0:
            item_list = ",".join(items)
            chat_history.append_system_message(
                f"{player.name} has the following things in their inventory: {item_list}"
            )

        chat_history.append_system_message(
            f"{player.name}'s motivation is to {player.motivation}"
        )
        chat_history.append_system_message(
            f"The tone of this story is {game_state.tone}"
        )

        # Add in information about prior quests
        prepared_mission_summaries = []
        for prior_quest in game_state.quests:
            prepared_mission_summaries.append(prior_quest.text_summary)
        if len(prepared_mission_summaries) > 0:
            chat_history.append_system_message(
                f"{player.name} has already been on previous missions: \n {prepared_mission_summaries}"
            )

        # Now save the chat history file
        quest.chat_file_id = chat_history.file.id

        if not game_state.quests:
            game_state.quests = []
        game_state.quests.append(quest)

        game_state.current_quest = quest.name

        # This saves it in a way that is both persistent (KV Store) and updates the context
        save_game_state(game_state, context)

        return quest

    def run(
        self, tool_input: List[Block], context: AgentContext
    ) -> Union[List[Block], Task[Any]]:
        purpose = None
        game_state = get_game_state(context)

        if tool_input:
            purpose = tool_input[0].text

        self.start_quest(game_state, context, purpose)
        return [Block(text="Starting quest...")]
