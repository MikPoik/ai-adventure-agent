from steamship import Block
from steamship.agents.schema import Action, Agent, AgentContext, FinishAction
import logging
from tools.start_quest_tool import StartQuestTool
from tools.start_conversation_tool import StartConversationTool
from tools.start_chat_quest_tool import StartChatQuestTool


class CampAgent(Agent):
    """The Camp Agent is currently just
    the default agent for CLI debugging.  It is not used in the web game.
    """

    def __init__(self, **kwargs):
        super().__init__(
            tools=[StartQuestTool(),StartConversationTool(),StartChatQuestTool()],  # , StartConversationTool()],
            **kwargs,
        )

    def next_action(self, context: AgentContext) -> Action:
        """Only one action possible - start quest (for CLI usage).  Else return dummy string."""

        last_message = ""
        if last_message_block := context.chat_history.last_user_message:
            if last_message_block.text is not None:
                last_message = last_message_block.text

        if "quest" in last_message.lower():
            return Action(tool=self.tools[0].name, output=[], input=[])
        elif "npc "in last_message.lower():
            return Action(tool=self.tools[1].name, output=[], input=[Block(text="The Merchant")])
        elif "chat" in last_message.lower():
            return Action(tool=self.tools[2].name, output=[], input=[])
        else:
            return FinishAction(
                output=[Block(text=f"Camp agent received text: {last_message}")]
            )
