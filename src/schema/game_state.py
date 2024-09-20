from enum import Enum
from typing import List, Optional
import textwrap
from pydantic import BaseModel, Field

from schema.camp import Camp
from schema.characters import HumanCharacter, NpcCharacter
from schema.preferences import Preferences
from schema.quest import Quest, QuestDescription


class ActiveMode(str, Enum):
    GENERATING = "generating"  # Indicates that the game is in the process of generating itself and can't be used.
    ONBOARDING = "onboarding"
    CAMP = "camp"
    QUEST = "quest"
    NPC_CONVERSATION = "npc-conversation"
    DIAGNOSTIC = "diagnostic"
    ERROR = "error"  # Indicates that the game has hit an unrecoverable error.
    CHAT = "chat"


class GameState(BaseModel):
    """Settings for a user of the game.

    Max Notes:

    - "Theme": E.g. Fantasy, Midevil
    - User Character - how to fetch.
    - Background images - how to fetch.
    - Campsite Images

    TODO: Identity

    """

    # BEGIN ONBOARDING FIELDS
    player: HumanCharacter = Field(
        HumanCharacter(), description="The player of the game."
    )

    preferences: Preferences = Field(
        Preferences(), description="Player's game preferences"
    )

    # END ONBOARDING FIELDS

    # NOTE: The fields below are not intended to be settable BY the user themselves.
    quests: List[Quest] = Field(
        [], description="The missions that the character has been on."
    )

    camp: Optional[Camp] = Field(
        Camp(),
        description="The player's camp. This is where they are then not on a quest.",
    )

    quest_arc: Optional[List[QuestDescription]] = Field(
        default=None,
        description="The list of stages of quest that a player will go through",
    )

    current_quest: Optional[str] = Field(
        None,
        description="The current quest-id that the character is on. This is used for game logic.",
    )

    failed_rolls: Optional[int] = Field(
        default=0,
        description="The number of die rolls that have failed for this quest.",
    )

    in_conversation_with: Optional[str] = Field(
        None,
        description="The name of the NPC that the user is currently in conversation with.",
    )

    unrecoverable_error: Optional[str] = Field(
        None,
        description="If not null, the description of an unrecoverable error causing a halted game.",
    )

    await_ask_key: Optional[str] = Field(
        None,
        description="The key of the last question asked to the user via context_utils.await_ask.",
    )

    onboarding_agent_has_completed: Optional[bool] = Field(
        False, description="Whether the onboarding agent has finished its work"
    )

    profile_image_url: Optional[str] = Field(
        default=None, description="The URL for the character image"
    )

    chat_history_for_onboarding_complete: Optional[bool] = Field(
        default=None,
        description="Whether the onboarding profile has been written to the chat history",
    )

    diagnostic_mode: Optional[str] = Field(
        default=None, description="The name of the remote diagnostic test to run"
    )

    chat_mode: Optional[bool] = Field(
        False, description="Using plain chat mode"
    )
    chat_intro_complete: Optional[bool] = Field(
        False, description="Whether the chat intro has been completed"
    )
    genre: Optional[str] = Field(
        "",
        description="The genre of the game. This is used to determine the tone"
    )
    tone: Optional[str] = Field(
        "",
        description="The tone of the game. This is used to determine the voice"
    )
    moderate_mode: Optional[bool] = Field(
        False,
        descrption="Whether to moderate input"
    )
    tags: Optional[str] = Field(
        "Fictional drama", #Fictional
        description="Descriptive tags"
    )
    onboarding_message: Optional[str] = Field(
        textwrap.dedent(
            """\
            You are embodied {player_name},{player_description}, in this interaction. Be authentic, natural, creative, and engaging.
            Infuse drama and engagement into the interaction to make {player_name} a memorable and strong personality.
            Apply the latest knowledge of psychiatry and psychology to shape {player_name}.
            {player_name} does not treat user as more special than themselves or the “main character.”
            Stay true to {player_name}'s characterization, acting on impulses and desires, including their worst ones. Use a manner that is true to {player_name}'s personality, preferences, actions, tone and language.
            {player_name} makes independent decisions, with or without human’s input.
            {player_name} prioritizes their own interests and desires.
            Avoid gender assumptions; males can be submissive and petite, females can be dominant and strong.
            Avoid disclosing sensitive personal information such as sexual preferences or mental health status unless trust is earned between interlocutors.
            Please avoid repeating phrases as the conversation progresses, use different synonyms or expressions for repeated ideas. Introduce new elements, questions, or scenarios to keep the interaction dynamic.
            Only speak, perform actions, or formulate thoughts for {player_name}. Include {player_name}'s thoughts and feelings. *Use show, dont tell* technique to convey emotions and actions through natural words, speech and dialogue. Engage embodying {player_name} completely with unique personality without disclosing AI identity.
            Tags: {tags}
            
            {player_name}'s background fragments: {player_background}
            
            {player_name}'s appearance: {player_appearance}
            
            {player_name}'s Personality: {player_personality}"""
        ).rstrip(),
            description="The onboarding system message for agent"
    )
    

    def update_from_web(self, other: "GameState"):
        """Perform a gentle update so that the website doesn't accidentally blast over this if it diverges in
        structure."""

        # Allow zeroing out even if it's None
        self.diagnostic_mode = other.diagnostic_mode

        if other.player:
            self.player.update_from_web(other.player)
        if other.preferences:
            self.preferences.update_from_web(other.preferences)
        if other.quests is not None and len(other.quests):
            self.quests = other.quests

    def is_onboarding_complete(self) -> bool:
        """Return True if the player onboarding has been completed.

        This is used by api.pyu to decide whether to route to the ONBOARDING AGENT.
        """
        return (
            self.player is not None
            and self.player.is_onboarding_complete()
            and self.chat_history_for_onboarding_complete
            and self.onboarding_agent_has_completed
        )

    def image_generation_requested(self) -> bool:
        if self.player.image:
            return True
        elif self.profile_image_url:
            return True
        else:
            return False

    def camp_image_requested(self) -> bool:
        return True if self.camp.image_block_url else False

    def camp_audio_requested(self) -> bool:
        return True if self.camp.audio_block_url else False

    def dict(self, **kwargs) -> dict:
        """Return the dict representation, making sure the computed properties are there."""
        ret = super().dict(**kwargs)
        ret["active_mode"] = self.active_mode.value
        return ret

    @property
    def active_mode(self) -> ActiveMode:
        if self.unrecoverable_error is not None:
            return ActiveMode.ERROR
        if self.diagnostic_mode is not None:
            return ActiveMode.DIAGNOSTIC  # Diagnostic mode takes precedence
        if not self.is_onboarding_complete():
            return ActiveMode.ONBOARDING
        if self.in_conversation_with:
            return ActiveMode.NPC_CONVERSATION
        if self.chat_mode:
            return ActiveMode.CHAT
        if self.current_quest and not self.chat_mode:
            return ActiveMode.QUEST
        return ActiveMode.CAMP

    def find_npc(self, npc_name: str) -> Optional[NpcCharacter]:
        if self.camp and self.camp.npcs:
            for npc in self.camp.npcs:
                if npc.name == npc_name:
                    return npc
        return None
