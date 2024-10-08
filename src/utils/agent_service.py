import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from steamship import Block, File, SteamshipError, Task
from steamship.agents.llms.openai import ChatOpenAI
from steamship.agents.logging import AgentLogging, StreamingOpts
from steamship.agents.schema import Action, Agent, FinishAction
from steamship.agents.schema.context import AgentContext, EmitFunc, Metadata
from steamship.agents.utils import with_llm
from steamship.data import TagKind
from steamship.data.tags.tag_constants import ChatTag
from steamship.invocable import PackageService, post
from steamship.invocable.invocable_response import StreamingResponse

from utils.context_utils import (
    RunNextAgentException,
    emit,
    get_game_state,
    get_server_settings,
    with_game_state,
    with_server_settings,
)
from utils.error_utils import record_and_throw_unrecoverable_error
from utils.tags import QuestIdTag
from utils.moderation_utils import mark_block_as_excluded


def build_context_appending_emit_func(
    context: AgentContext, make_blocks_public: Optional[bool] = False
) -> EmitFunc:
    """Build an emit function that will append output blocks directly to ChatHistory, via AgentContext.

    NOTE: Messages will be tagged as ASSISTANT messages, as this assumes that agent output should be considered
    an assistant response to a USER.

    EXTENSION NOTE:
    - Some blocks are generated and then emitted when Generation is complete.
    - Some blocks are streamed directly into the ChatHistory (for the web user) but we still want to emit them for users
    of non-streaming clients: ship run local, AgentREPL, Telegram, etc.

    As a hack, this project adopts the following convention:

    - If a Block has the tag `kind=chat, name=streamed-to-chat-history` then it doesn't call emit here, since that
      would be redundant.
    """

    def chat_history_append_func(blocks: List[Block], metadata: Metadata):
        for block in blocks:
            # Check if this block was already streamed to ChatHistory
            already_streamed_to_chat_history = False
            for tag in block.tags or []:
                if (
                    tag.kind == TagKind.CHAT.value
                    and tag.name == "streamed-to-chat-history"
                ):
                    already_streamed_to_chat_history = True
                    break

            # If this block was already streamed, skip emitting it here.
            if already_streamed_to_chat_history:
                continue

            block.set_public_data(make_blocks_public)
            if block.text:
                context.chat_history.append_assistant_message(
                    text=block.text,
                    tags=block.tags,
                    mime_type=block.mime_type,
                )
            else:
                context.chat_history.append_assistant_message(
                    tags=block.tags,
                    url=block.raw_data_url or block.url or block.content_url or None,
                    mime_type=block.mime_type,
                )

    return chat_history_append_func


def _context_key_from_file(key: str, file: File) -> Optional[str]:
    for tag in file.tags:
        if tag.kind == TagKind.CHAT and tag.name == ChatTag.CONTEXT_KEYS:
            if value := tag.value:
                return value.get(key, None)
    return None


class AgentService(PackageService):
    """AgentService is a Steamship Package that can use an Agent, Tools, and a provided AgentContext to
    respond to user input."""

    agent: Optional[Agent]
    """The default agent for this agent service."""

    use_llm_cache: bool
    """Whether or not to cache LLM calls (for tool selection/direct responses) by default."""

    use_action_cache: bool
    """Whether or not to cache agent Actions (for tool runs) by default."""

    max_actions_per_run: int
    """The maximum number of actions to permit while the agent is reasoning.

    This is intended primarily to act as a backstop to prevent a condition in which the Agent decides to loop endlessly
    on tool runs that consume resources with a cost-basis (e.g. prompt completions, embedding operations, vector lookups)
    """

    max_actions_per_tool: Dict[str, int] = {}
    """The maximum number of actions to permit per tool name.

    This is intended primarily to act as a backstop to prevent a condition in which the Agent decides to loop endlessly
    on tool runs that consume resources with a cost-basis (e.g. prompt completions, embedding operations, vector lookups)
    """

    _agent_context: Optional[AgentContext] = None

    def __init__(
        self,
        use_llm_cache: Optional[bool] = False,
        use_action_cache: Optional[bool] = False,
        max_actions_per_run: Optional[int] = 5,
        max_actions_per_tool: Optional[Dict[str, int]] = None,
        agent: Optional[Agent] = None,
        **kwargs,
    ):
        self.use_llm_cache = use_llm_cache
        self.use_action_cache = use_action_cache
        self.max_actions_per_run = max_actions_per_run
        self.agent = agent
        self.max_actions_per_tool = max_actions_per_tool or {}
        super().__init__(**kwargs)

    ###############################################
    # Tool selection / execution
    ###############################################

    def next_action(
        self, agent: Agent, input_blocks: List[Block], context: AgentContext
    ) -> Action:
        action: Action = None
        if context.llm_cache:
            action = context.llm_cache.lookup(key=input_blocks)
        if action:
            logging.info(
                f"Using cached tool selection: calling {action.tool}.",
                extra={
                    AgentLogging.IS_MESSAGE: True,
                    AgentLogging.MESSAGE_TYPE: AgentLogging.THOUGHT,
                    AgentLogging.MESSAGE_AUTHOR: AgentLogging.AGENT,
                },
            )
        else:
            logging.info(
                "Selecting next action...",
                extra={
                    AgentLogging.IS_MESSAGE: True,
                    AgentLogging.MESSAGE_TYPE: AgentLogging.THOUGHT,
                    AgentLogging.MESSAGE_AUTHOR: AgentLogging.AGENT,
                },
            )
            action = agent.next_action(context=context)
            if context.llm_cache:
                context.llm_cache.update(key=input_blocks, value=action)

        logging.info(
            f"Selected next action: {action.tool}",
            extra={
                AgentLogging.IS_MESSAGE: True,
                AgentLogging.MESSAGE_TYPE: AgentLogging.THOUGHT,
                AgentLogging.MESSAGE_AUTHOR: AgentLogging.AGENT,
            },
        )

        # save action selection to history...
        return action

    def run_action(self, agent: Agent, action: Action, context: AgentContext):
        if isinstance(action, FinishAction):
            return

        if not agent:
            raise SteamshipError(
                "Missing agent. Not able to run action on behalf of missing agent."
            )

        if context.action_cache:
            # if cache and action is cached, use it. otherwise proceed normally.
            if output_blocks := context.action_cache.lookup(key=action):
                outputs = ",".join([f"{b.as_llm_input()}" for b in output_blocks])
                logging.info(
                    f"Tool {action.tool}: ({outputs}) [cached]",
                    extra={
                        AgentLogging.TOOL_NAME: action.tool,
                        AgentLogging.IS_MESSAGE: True,
                        AgentLogging.MESSAGE_TYPE: AgentLogging.OBSERVATION,
                        AgentLogging.MESSAGE_AUTHOR: AgentLogging.AGENT,
                    },
                )
                action.output = output_blocks
                agent.record_action_run(action, context)
                return

        tool = next((tool for tool in agent.tools if tool.name == action.tool), None)
        if not tool:
            raise SteamshipError(
                f"Could not find tool '{action.tool}' for action. Not able to run."
            )

        # TODO: Arrive at a solid design for the details of this structured log object
        inputs = ",".join([f"{b.as_llm_input()}" for b in action.input])
        logging.info(
            f"Running Tool {action.tool} ({inputs})",
            extra={
                AgentLogging.TOOL_NAME: action.tool,
                AgentLogging.IS_MESSAGE: True,
                AgentLogging.MESSAGE_TYPE: AgentLogging.ACTION,
                AgentLogging.MESSAGE_AUTHOR: AgentLogging.AGENT,
            },
        )
        blocks_or_task = tool.run(tool_input=action.input, context=context)
        if isinstance(blocks_or_task, Task):
            raise SteamshipError(
                "Tools return Tasks are not yet supported (but will be soon). "
                "Please use synchronous Tasks (Tools that return List[Block] for now."
            )
        else:
            outputs = ",".join([f"{b.as_llm_input()}" for b in blocks_or_task])
            logging.info(
                f"Tool {action.tool}: ({outputs})",
                extra={
                    AgentLogging.TOOL_NAME: action.tool,
                    AgentLogging.IS_MESSAGE: True,
                    AgentLogging.MESSAGE_TYPE: AgentLogging.OBSERVATION,
                    AgentLogging.MESSAGE_AUTHOR: AgentLogging.AGENT,
                },
            )
            action.output = blocks_or_task
            action.is_final = (
                tool.is_final
            )  # Permit the tool to decide if this action should halt the reasoning loop.
            agent.record_action_run(action, context)
            if context.action_cache and tool.cacheable:
                context.action_cache.update(key=action, value=action.output)

    def run_agent(self, agent: Agent, context: AgentContext):
        # First, some bookkeeping.

        # Clear any prior agent steps from set of completed steps.
        # This will allow the agent to select tools/dispatch actions based on a new context
        context.completed_steps = []

        # Set the pointer for the current action.
        # The agent will continue to take actions until it is ready to respond.
        action = self.next_action(
            agent=agent,
            input_blocks=[context.chat_history.last_user_message],
            context=context,
        )

        # Set the counter for the number of actions run.
        # This enables the agent to enforce a budget on actions to guard against running forever.
        number_of_actions_run = 0
        actions_per_tool = defaultdict(lambda: 0)

        while not action.is_final:
            # If we've exceeded our Action Budget, throw an error.
            if number_of_actions_run >= self.max_actions_per_run:
                raise SteamshipError(
                    message=(
                        f"Agent reached its Action budget of {self.max_actions_per_run} without arriving at a "
                        f"response. If you are the developer, checking the logs may reveal it was selecting "
                        f"unhelpful tools or receiving unhelpful responses from them."
                    )
                )

            if action.tool and action.tool in self.max_actions_per_tool:
                if (
                    actions_per_tool[action.tool]
                    > self.max_actions_per_tool[action.tool]
                ):
                    raise SteamshipError(
                        message=(
                            f"Agent reached its Action budget of {self.max_actions_per_tool[action.tool]} for tool {action.tool} without arriving at a response. If you are the developer, checking the logs may reveal it was selecting unhelpful tools or receiving unhelpful responses from them."
                        )
                    )

            # Run the next action and increment our counter
            self.run_action(agent=agent, action=action, context=context)
            number_of_actions_run += 1
            if action.tool:
                actions_per_tool[action.tool] += 1

            # Sometimes, running an action will result in it being dynamically set as a final action as a result of
            # the tool that performed the action's operation. E.g. a Tool wishes to have its output considered the
            # final and complete response.
            #
            # This is a distinct scenario from the next_action method returning a FinalAction, which is why this is
            # a break statement here rather than hidden within the while loop condition above: it's essentially an
            # "early exit" signal.
            if action.is_final:
                break

            # If we're still here, then the Agent has performed work and must decide what to do next. The next
            # action might be another tool, or it might be the Agent deciding to convert this Action's output into
            # A FinalAction object, which would then cause the while loop to terminate upon next iteration.
            action = self.next_action(
                agent=agent, input_blocks=action.output, context=context
            )

            # TODO: Arrive at a solid design for the details of this structured log object
            logging.info(
                f"Next Tool: {action.tool}",
                extra={
                    AgentLogging.TOOL_NAME: action.tool,
                    AgentLogging.IS_MESSAGE: False,
                    AgentLogging.MESSAGE_TYPE: AgentLogging.ACTION,
                    AgentLogging.MESSAGE_AUTHOR: AgentLogging.AGENT,
                },
            )

        agent.record_action_run(action, context)
        output_text_length = 0
        if action.output is not None:
            output_text_length = sum([len(block.text or "") for block in action.output])
        logging.info(
            f"Completed agent run. Result: {len(action.output or [])} blocks. {output_text_length} total text length. "
            f"Emitting on {len(context.emit_funcs)} functions."
        )
        for func in context.emit_funcs:
            logging.info(
                f"Emitting via function '{func.__name__}' for context: {context.id}"
            )
            func(action.output, context.metadata)

    def set_default_agent(self, agent: Agent):
        self.agent = agent

    def get_default_agent(self, throw_if_missing: bool = True) -> Optional[Agent]:
        """Return the default agent of this agent service.

        This is a helper wrapper to safeguard naming conventions that have formed.
        """
        if hasattr(self, "agent"):
            return self.agent
        elif hasattr(self, "_agent"):
            return self._agent
        else:
            if throw_if_missing:
                raise SteamshipError(
                    message="No Agent object found in the Agent Service. "
                    "Please name it either self.agent or self._agent."
                )
            else:
                return None

    def build_default_context(
        self, context_id: Optional[str] = None, **kwargs
    ) -> AgentContext:
        """Load the context for the agent.

        The AgentContext is a single place to implement (or override) the all context and state that will be used by
        the different components of the game.

        You can fetch many things from it using fetchers in `context_utils.py` such as:

        - get_game_state
        - get_server_settings
        - get_background_image_generator
        - etc

        This provides any piece of code, anywhere in the codebase, access to the correct objects
        for generating different kinds of assets and fetching/saving different kinds of state.

        INTERNAL NOTE:
        We provide a DIFFERENT VERSION of AgentService's context so that we can remove the
        # get_default_agent() call from it. In AgentService,
        this method depends upon get_default_agent. In this class, that dependency is flipped.
        """

        if self._agent_context is not None:
            return self._agent_context  # Used cached copy

        # AgentContexts serve to allow the AgentService to run agents
        # with appropriate information about the desired tasking.
        
        if context_id is not None:
            logging.warning(
                "This agent ALWAYS uses the context id `default` since it is a game occuping an entire workspace, not confined to a single chat history. "
                f"The provided context_id of {context_id} will be ignored. This is to prevent surprising state errors."
            )

        # NOTA BENE!
        context_id = "default"
    
        use_llm_cache = self.use_llm_cache
        if runtime_use_llm_cache := kwargs.get("use_llm_cache"):
            use_llm_cache = runtime_use_llm_cache

        use_action_cache = self.use_action_cache
        if runtime_use_action_cache := kwargs.get("use_action_cache"):
            use_action_cache = runtime_use_action_cache

        include_agent_messages = kwargs.get("include_agent_messages", True)
        include_llm_messages = kwargs.get("include_llm_messages", True)
        include_tool_messages = kwargs.get("include_tool_messages", True)

        context = AgentContext.get_or_create(
            client=self.client,
            context_keys={"id": f"{context_id}"},
            use_llm_cache=use_llm_cache,
            use_action_cache=use_action_cache,
            streaming_opts=StreamingOpts(
                include_agent_messages=include_agent_messages,
                include_llm_messages=include_llm_messages,
                include_tool_messages=include_tool_messages,
            ),
            initial_system_message="",  # None necessary
            searchable=False,
        )

        # Add a default LLM to the context, using the Agent's if it exists.
        llm = ChatOpenAI(client=self.client)
        context = with_llm(context=context, llm=llm)

        # Get the game state and add to context
        game_state = get_game_state(context)
        context = with_game_state(
            game_state, context
        )  # TODO: This shouldn't be necessary since get_game_state caches it.
        server_settings = get_server_settings(context)
        context = with_server_settings(
            server_settings, context
        )  # TODO: This shouldn't bve necessary since get_server_settings caches it.
        # TODO(doug): figure out how to make this selectable.
        self._agent_context = context
        return context

    def _history_file_for_context(
        self, context_id: Optional[str] = None, **kwargs
    ) -> File:

        # NOTA BENE!
        context_id = "default"

        #
        # # AgentContexts serve to allow the AgentService to run agents
        # # with appropriate information about the desired tasking.
        # if context_id is None:
        #     context_id = uuid.uuid4()

        ctx = AgentContext.get_or_create(
            self.client,
            context_keys={"id": f"{context_id}"},
            initial_system_message=self.get_default_agent().default_system_message(),
            searchable=False,
        )
        return ctx.chat_history.file

    def _streaming_context_id_and_file(
        self, context_id: Optional[str] = None, **kwargs
    ) -> Tuple[Optional[str], File]:
        history_file = self._history_file_for_context(context_id=context_id)
        ctx_id = context_id

        # if no context ID is provided, try to get a consistent ID from history_file
        if not ctx_id:
            ctx_id = _context_key_from_file(key="id", file=history_file)

        # if you can't find a consistent context_id, then something has gone wrong, preventing streaming
        if not ctx_id:
            # TODO(dougreid): this points to a slight flaw in the context_keys vs. context_id
            raise SteamshipError("Error setting up context: no id found for context.")

        return ctx_id, history_file

    @post("async_prompt")
    def async_prompt(
        self, prompt: Optional[str] = None, context_id: Optional[str] = None, **kwargs
    ) -> StreamingResponse:
        ctx_id, history_file = self._streaming_context_id_and_file(
            context_id=context_id, **kwargs
        )
        logging.info(f"/async_prompt called with message {prompt}")
        task = self.invoke_later(
            "/prompt", arguments={"prompt": prompt, "context_id": ctx_id, **kwargs}
        )
        return StreamingResponse(task=task, file=history_file)

    def _prompt(self, prompt: str, context: AgentContext) -> List[Block]:
        game_state = get_game_state(context)

        base_tags = []
        if game_state.current_quest:
            base_tags.append(QuestIdTag(game_state.current_quest))

        user_block = context.chat_history.append_user_message(prompt, tags=base_tags)
        if user_block.text == "":
            mark_block_as_excluded(user_block)
            
        agent: Optional[Agent] = self.get_default_agent()
        self.run_agent(agent, context)

    @post("prompt")
    def prompt(  # noqa: C901
        self, prompt: Optional[str] = None, context_id: Optional[str] = None, **kwargs
    ) -> List[Block]:
        """Run an agent with the provided text as the input."""
        with self.build_default_context(context_id, **kwargs) as context:
            prompt = prompt or kwargs.get("question") or "Hi."
            logging.info(f"/prompt called with message {prompt}")

            # AgentServices provide an emit function hook to access the output of running
            # agents and tools. The emit functions fire at after the supplied agent emits
            # a "FinishAction".
            #
            # Here, we show one way of accessing the output in a synchronous fashion. An
            # alternative way would be to access the final Action in the `context.completed_steps`
            # after the call to `run_agent()`.
            output_blocks = []

            def sync_emit(blocks: List[Block], meta: Metadata):
                nonlocal output_blocks
                output_blocks.extend(blocks)

            if sum(fn.__name__ == "sync_emit" for fn in context.emit_funcs) == 0:
                context.emit_funcs.append(sync_emit)

            # NOTE: we make blocks public on output here to allow for ease of testing and sharing
            if (
                sum(
                    fn.__name__ == "chat_history_append_func"
                    for fn in context.emit_funcs
                )
                == 0
            ):
                context.emit_funcs.append(
                    build_context_appending_emit_func(
                        context=context, make_blocks_public=True
                    )
                )

            had_exception = (
                True  # Not true, but it causes the loop to execute at least once.
            )
            max_exceptions_allowed = 4
            exception_count = 0
            while had_exception:
                try:
                    self._prompt(prompt, context)
                    had_exception = False
                except RunNextAgentException as e:
                    exception_count += 1
                    if exception_count > max_exceptions_allowed:
                        raise SteamshipError(message="Maximum agent switches exceeded")

                    logging.info(
                        "Got RunNextAgentException. Loading next agent.",
                        extra={
                            AgentLogging.IS_MESSAGE: True,
                            AgentLogging.MESSAGE_TYPE: AgentLogging.THOUGHT,
                            AgentLogging.MESSAGE_AUTHOR: AgentLogging.AGENT,
                        },
                    )
                    self.agent = None

                    had_exception = True
                    for block in e.action.output or []:
                        emit(output=block, context=context)

                    prompt = "Hi."
                    if e.action.input:
                        prompt = e.action.input[0].text
                except BaseException as e:
                    record_and_throw_unrecoverable_error(e, context)

            # timings = API_TIMINGS
            # pretty_print_timings(timings)

            # Return the response as a set of multi-modal blocks.
            return output_blocks
