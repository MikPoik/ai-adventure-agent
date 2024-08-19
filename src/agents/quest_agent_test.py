"""
WIP testing version
"""
import json
import logging
from datetime import datetime, timezone
from enum import Enum
from random import randint, random
from typing import Dict, List

from steamship import SteamshipError, Tag
from steamship.agents.logging import AgentLogging
from steamship.agents.schema import Action, AgentContext
from steamship.agents.schema.action import FinishAction

from generators.generator_context_utils import (
    get_music_generator,
    get_quest_background_image_generator,
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


class Likelihood(str, Enum):
    VERY_LIKELY = "VERY_LIKELY"
    LIKELY = "LIKELY"
    UNKNOWN = "UNKNOWN"
    UNLIKELY = "UNLIKELY"
    VERY_UNLIKELY = "VERY_UNLIKELY"


LIKELIHOOD_MAP: Dict[Difficulty, Dict[Likelihood, float]] = {
    Difficulty.EASY: {
        Likelihood.VERY_LIKELY: 0.05,
        Likelihood.LIKELY: 0.2,
        Likelihood.UNKNOWN: 0.35,
        Likelihood.UNLIKELY: 0.55,
        Likelihood.VERY_UNLIKELY: 0.7,
    },
    Difficulty.NORMAL: {
        Likelihood.VERY_LIKELY: 0.1,
        Likelihood.LIKELY: 0.3,
        Likelihood.UNKNOWN: 0.5,
        Likelihood.UNLIKELY: 0.7,
        Likelihood.VERY_UNLIKELY: 0.9,
    },
    Difficulty.HARD: {
        Likelihood.VERY_LIKELY: 0.2,
        Likelihood.LIKELY: 0.35,
        Likelihood.UNKNOWN: 0.6,
        Likelihood.UNLIKELY: 0.8,
        Likelihood.VERY_UNLIKELY: 0.95,
    },
}


class QuestAgent(InterruptiblePythonAgent):
    """
    The quest agent goes on a quest!

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
        #logging.warning("quests: "+ str(game_state.quests))
        #logging.warning("Quest descriptions: "+ str(game_state.quest_arc))
        
        if not game_state.quest_arc:
            game_state.quest_arc = generate_quest_arc(game_state.player,
                                                      context)

        if len(game_state.quest_arc) >= len(game_state.quests):
            quest_description = game_state.quest_arc[len(game_state.quests) - 1]

        
        else:
            logging.warning("QUEST DESCRIPTION IS NONE.")
            quest_description = None
        #print("QUEST ARC:\n\n"+str(game_state.quest_arc)+"\n\n")
        #print("QUESTS:\n\n"+str(game_state.quests)+"\n\n")
        # copy challenge description over to quest
        if quest_description:
            # TODO(dougreid): should we give these things IDs so that we only copy new ones?
            # or is this logic OK as a placeholder?
            if len(quest_description.challenges) != len(quest.challenges):
                for challenge in quest_description.challenges:
                    quest.challenges.append(
                        QuestChallenge(
                            name=challenge.name,
                            description=challenge.description,
                        ))

        logging.debug(
            "Running Quest Agent",
            extra={
                AgentLogging.IS_MESSAGE: True,
                AgentLogging.MESSAGE_TYPE: AgentLogging.AGENT,
                AgentLogging.MESSAGE_AUTHOR: AgentLogging.TOOL,
                AgentLogging.AGENT_NAME: self.__class__.__name__,
            },
        )

        if not quest.sent_intro:
            quest.sent_intro = True
            save_game_state(game_state, context)

            logging.debug(
                "Sending Intro Part 2",
                extra={
                    AgentLogging.IS_MESSAGE: True,
                    AgentLogging.MESSAGE_TYPE: AgentLogging.AGENT,
                    AgentLogging.MESSAGE_AUTHOR: AgentLogging.TOOL,
                    AgentLogging.AGENT_NAME: self.__class__.__name__,
                },
            )

            if quest_description is not None:
                optional_desc = ""
                if (quest_description.description
                        and quest_description.description.strip()):
                    optional_desc += f"\n{quest_description.description}"
                if (quest_description.other_information
                        and quest_description.other_information.strip()):
                    optional_desc += f"\n{quest_description.other_information}"

                #if len(optional_desc.strip()) > 0:
                #    optional_desc = ("\n\n## Current role-play goal:" +
                #                     optional_desc)
                #context.chat_history.append_system_message(
                #    text=
                #    f"\nRole-play scene:\n{quest_description.goal} "
                #    f"at {quest_description.location}. \n {optional_desc}\n",
                #    tags=[
                #        Tag(
                #            kind=TagKindExtensions.INSTRUCTIONS,
                #            name=InstructionsTag.QUEST,
                #        ),
                #        QuestIdTag(quest_id=quest.name),
                #    ],
                #)
                prompt = f"""Introduce yourself as {game_state.player.name}."""   
                logging.warning("Intro part 2")
            else:
                prompt = f"""Begin role-play as {game_state.player.name}"""
                logging.warning("Intro")

            
            block = send_story_generation(
                prompt=prompt,
                quest_name=quest.name,
                context=context,
            )


            await_streamed_block(block, context)
            
                        #self.create_problem(game_state,
                        #                    context,
                        #                    quest,
                        #                    quest_description=quest_description)
            save_game_state(game_state, context)
        else:
            logging.debug(
                "First Quest Story Block Skipped",
                extra={
                    AgentLogging.IS_MESSAGE: True,
                    AgentLogging.MESSAGE_TYPE: AgentLogging.AGENT,
                    AgentLogging.MESSAGE_AUTHOR: AgentLogging.TOOL,
                    AgentLogging.AGENT_NAME: self.__class__.__name__,
                },
            )

        
        logging.warning("ask user")
        if not quest.all_problems_solved():
            user_solution = await_ask(
                f"What do you say next?",
                context,
                key_suffix=
                f"{quest.name} solution {len(quest.user_problem_solutions)}"
                
            )
            #response_plan = self.generate_plan(game_state, context, quest, user_solution)


            #logging.warning("response plan: "+str(response_plan))

            image_request = None
            #image_request = self.is_solution_attempt(game_state,context,quest,user_solution)
            #print("image request: "+str(image_request))
            #image_request_json = json.loads(image_request)
            suggestion = ""#response_plan
            
            #quest.add_user_solution(user_solution)
            save_game_state(game_state, context)
            additional_info = f"Here's a plan to aid in {game_state.player.name}'s response:\n\"" + suggestion+"\"\n\n"
            #if image_request:
            #    additional_info="User has requested and image of you, and the image is being generated. Respond according to sending the image."

        
            response_block = self.respond_to_user(game_state,context,quest,quest_description,additional_info=additional_info)
            if image_request:
                print("image gen here")
                #if image_gen := get_quest_background_image_generator(context):
                #    image_gen.request_scene_image_generation(
                #        description=response_block.text, context=context)
                    
            if not quest.all_problems_solved():
                logging.warning("continue solving problems")
                #self.create_problem(game_state,
                #                    context,
                #                    quest,
                #                    quest_description=quest_description)
                user_solution = await_ask(
                    "What do you say next?",
                    context,
                    key_suffix=
                    f"{quest.name} solution {len(quest.user_problem_solutions)}"
                    
                    
                )
                #quest.add_user_solution(user_solution)
                

        if not quest.sent_outro:
            quest.sent_outro = True
            save_game_state(game_state, context)

            if quest_description is not None:
                prompt = f""""
Complete the story of the {player.name}'s current quest. {player.name} should achieve the goal of current quest, but NOT their overall goal.

## Quest goal
{quest_description.goal}

## Overall goal
{server_settings.adventure_goal}

## Note
- write a single short paragraph without line breaks.
- Tell the story using a tone of '{server_settings.narrative_tone}' and with a narrative voice of '{server_settings.narrative_voice}'."""
                #logging.warning("Outro: " +prompt)
            else:
                prompt = f"""Complete the story of the {player.name}'s current quest. {player.name} should not yet achieve their overall goal.

## Overall goal
{server_settings.adventure_goal}

## Note
- Write a single short paragraph without line breaks.
- Tell the story using a tone of '{server_settings.narrative_tone}' and with a narrative voice of '{server_settings.narrative_voice}'."""
            
                #logging.warning("Outro: " +prompt)
            story_end_block = send_story_generation(
                prompt,
                quest_name=quest.name,
                context=context,
            )
            await_streamed_block(story_end_block, context)

        quest.completed_timestamp = datetime.now(timezone.utc).isoformat()

        blocks = EndQuestTool().run([], context)
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
        quest_description: QuestDescription,
        additional_info: str = "",
    ):
        prompt = f"{additional_info}What does {game_state.player.name} say next to keep the conversation fresh,authentic,natural,creative and engaging? Provide response to user only."
        solution_block = send_story_generation(
            prompt=prompt,
            quest_name=quest.name,
            context=context,
        )
        return await_streamed_block(solution_block, context)

    
    def create_problem(
        self,
        game_state: GameState,
        context: AgentContext,
        quest: Quest,
        quest_description: QuestDescription,
    ):
        if len(quest.challenges) > 0:
            # this is a specified-challenges type of quest
            solved_challenges = sum(
                [1 if x.solution else 0 for x in quest.challenges])
            total_challenges = len(quest.challenges)
            if (isinstance(solved_challenges, int)
                    and solved_challenges < total_challenges):
                current_challenge = quest.challenges[solved_challenges]
                server_settings = get_server_settings(context=context)
                prompt = f"""
Introduce a problem to the role-play scene between the human and {game_state.player.name}, human should try to solve the problem.
                
- DO NOT solve the challenge for human.
- The role-play scene should continue the conversation.
- {game_state.player.name} should allow human to decide how to attempt to solve the challenge.

## Current role-play scene
- {quest.name}
- {quest_description.location}, {quest_description.goal}

## Introduce problem to current scene
- {current_challenge.name} 
- {current_challenge.description}

## Note
- write a single short paragraph without line breaks.
- Use a tone of '{server_settings.narrative_tone}' and a narrative voice of '{server_settings.narrative_voice}'."""
                logging.warning("Problem: " +prompt)
            else:
                raise SteamshipError(
                    "trying to solve more than the number of challenges.")
        elif len(quest.user_problem_solutions
                 ) == quest.num_problems_to_encounter - 1:
            # if last problem, try to make it make sense for wrapping things up
            server_settings = get_server_settings(context=context)
            prompt = f"""
Continue telling the story of {game_state.player.name}'s quest. Write about them encountering a 
challenge that prevents them from completing their current quest.

- Do not use the word 'challenge' directly.
- Do not mention any sort of ordering of challenges (examples: 'first' or 'next').
- Do not solve the challenge for {game_state.player.name}.
- The chapter MUST continue the current story arc of the quest.

## Note
- write a single short paragraph without line breaks.
- Tell the story using a tone of '{server_settings.narrative_tone}' and with a narrative voice of '{server_settings.narrative_voice}'."""
            #logging.warning("Problem: " +prompt)
        
        else:
            server_settings = get_server_settings(context=context)
            prompt = f"""
Introduce a problem to the role-play scene between the human and {game_state.player.name}, human should try to solve the problem. 

- The problem MUST NOT repeat (or be equivalent to) a prior problem faced by {game_state.player.name}.
- DO NOT introduce more than one problem.
- DO NOT use the words 'challenge' or 'problem' directly.
- DO NOT mention any sort of ordering of challenges (examples: 'first' or 'next').

## Scene 
{quest_description.location}, {quest_description.goal}

## Note
- write a single short paragraph without line breaks.
- Use a tone of '{server_settings.narrative_tone}' and a narrative voice of '{server_settings.narrative_voice}'."""
        #logging.warning("Problem: " +prompt)

        num_paragraphs = randint(1, 2)  # noqa: S311
        problem_block = send_story_generation(
            prompt=prompt,
            quest_name=quest.name,
            context=context,
        )
        updated_problem_block = await_streamed_block(problem_block, context)
        quest.current_problem = updated_problem_block.text

        if num_paragraphs > 1:
            # replace "Tell the" with "Continue telling the" and re-prompt
            if prompt.startswith("Write the role-play scene"):
                new_prompt = prompt.removeprefix("Write the role-play scene")
                prompt = f"Continue the role-play scene {new_prompt}"
                #logging.warning("\n\n ## Continue story ##\n\n")
            problem_block = send_story_generation(
                prompt=prompt,
                quest_name=quest.name,
                context=context,
            )
            #logging.warning("Telling prompt: "+prompt)
            updated_problem_block = await_streamed_block(
                problem_block, context)
            quest.current_problem = (
                f"{quest.current_problem}\n{updated_problem_block.text}")

        if image_gen := get_quest_background_image_generator(context):
            image_gen.request_scene_image_generation(
                description=updated_problem_block.text, context=context)
        if music_gen := get_music_generator(context):
            if server_settings.generate_music:
                music_gen.request_scene_music_generation(
                    description=updated_problem_block.text, context=context)
                
    def generate_plan(self, game_state: GameState, context: AgentContext,
        quest: Quest,user_input:str):
        
        prompt = f"""Current game state:
Event: 
Status: Ongoing
Goal: {game_state.quest_arc[0].goal}


How should the game progress? Should you introduce an event between {game_state.player.name} and user, or should you continue the current state? Has user achieved the current goal? Has user asked for a visual?

Provide a possible event and goal for the game to progress.
If there is an ongoing event, provide a status for it.
Do not generate dialogue, just the event and goal and status, if needed generate a visual image description also. 

If user is requesting for a visual of {game_state.player.name} you may provide one with image tool using Image: insert image description here

Provide response in format:
Event: short event description
Goal: short goal description
Status: short status description started/ongoing/completed
Tool: No Tool or tool call"""
    
        is_solution_attempt_response = generate_is_solution_attempt(
        prompt=prompt,
        quest_name=quest.name,
        context=context,
        )
        logging.warning("problem prompt: " +prompt)
        logging.warning(f"Plan response: {is_solution_attempt_response.text}")
        return is_solution_attempt_response.text.strip()

    
    def is_solution_attempt(self, game_state: GameState, context: AgentContext,
                            quest: Quest,user_input:str):
        #return "NO"
        prompt = f"""Given the task "Is user asking for a selfie? Generate image only if user is requesting image", consider the following list of available tools to solve the task effectively. Structure your response to utilize these tools optimally, detailing a sequence that leverages their capabilities. 
**Available Tools:**
1. *GenerateImage**: Useful to generate image. Input should be text string describing image.
2. *NoTool*: If task doesnt require a tool.

Use only the tools listed above.

**Response Components:**
1. **Evolved Thought**: Share your developed thought process, considering the available tools and how they could be orchestrated to address the task.
2. **Reasoning**: Explain the logic behind your evolved thought, particularly the choice and sequence of tools.
3. **Tools**: Specify which tools from the available list would be used, the order of their use, and any necessary inputs or parameters for each. Include a brief rationale for each tool's selection.
    - Your suggestions might look like this:
        - **Tool/Method Name**: Why it's suitable for a part of the process.
        - **Inputs/Parameters**: Specific inputs or operations required.
        - **Expected Outcome**: The anticipated result of using this tool.
If you dont need a tool, leave this empty.
4. **Answer**: Synthesize the insights and proposed tool sequence into a clear answer or solution.
5. **Follow-On**: Propose a next step for further exploration, which could involve analyzing the output, considering alternative tools, or extending the task.

Format your response as follows:
{{
"evolved_thought": "[YOUR EVOLVED THOUGHT HERE]",
"reasoning": "[YOUR REASONING HERE]",
"tools": [
{{
"name": "[TOOL/METHOD NAME]",
"inputs_parameters": "[INPUTS/PARAMETERS]",
"expected_outcome": "[EXPECTED OUTCOME]"
}},
],
"answer": "[YOUR ANSWER HERE]",
"follow-on": "[YOUR FOLLOW-ON IDEA OR QUESTION HERE]"
}}


```json"""
        
        is_solution_attempt_response = generate_is_solution_attempt(
            prompt=prompt,
            quest_name=quest.name,
            context=context,
        )
        logging.warning("problem prompt: " +prompt)
        logging.warning(
            f"Is solution attempt: {is_solution_attempt_response.text}")
        return is_solution_attempt_response.text

    def evaluate_solution(self, game_state: GameState, context: AgentContext,
                          quest: Quest):
        server_settings = get_server_settings(context)
        prompt = f"""Human tries to solve the current problem in the role-play scene, how likely is this to succeed?


## Solution suggestion
{quest.user_problem_solutions[-1]}.

## Qualification
- Please consider the context and {game_state.player.name}'s abilities, traits.
- Consider the solution when game difficulty is {server_settings.difficulty}.
- Consider if the solution qualifies the current goal of {quest.name}

## Note
- Return only the choice text.
- No other text.

Return only one option of ['VERY UNLIKELY', 'UNLIKELY', 'LIKELY', 'VERY LIKELY']
Option:"""
        
        #logging.warning("problem prompt: " +prompt)
        likelihood_block = generate_likelihood_estimation(
            prompt=prompt,
            quest_name=quest.name,
            context=context,
        )
        logging.warning("solution result: " + likelihood_block.text)
        likelihood_text = likelihood_block.text.upper().strip()
        likelihood_map = LIKELIHOOD_MAP.get(server_settings.difficulty)
        if "VERY UNLIKELY" in likelihood_text:
            required_roll = likelihood_map[Likelihood.VERY_UNLIKELY]
        elif "VERY LIKELY" in likelihood_text:
            required_roll = likelihood_map[Likelihood.VERY_LIKELY]
        elif "UNLIKELY" in likelihood_text:
            required_roll = likelihood_map[Likelihood.UNLIKELY]
        elif "LIKELY" in likelihood_text:
            required_roll = likelihood_map[Likelihood.LIKELY]
        else:
            required_roll = likelihood_map[Likelihood.UNKNOWN]

        # Add minor randomness, but don't drop below 0.05 (2 on d20) or go above 0.95 (20 on d20)
        required_roll_mod = 0.05 * (randint(-2, 2))  # noqa: S311
        required_roll = min(0.95, max(0.05, required_roll + required_roll_mod))
        # make sure we don't get weird floating point near values
        required_roll = round(required_roll, 2)

        roll = random()  # noqa: S311
        succeeded = roll > required_roll
        dice_roll_message = json.dumps({
            "required": required_roll,
            "rolled": roll,
            "success": succeeded,
            "mod": required_roll_mod,
        })
        context.chat_history.append_system_message(dice_roll_message,
                                                   tags=self.tags(
                                                       QuestTag.DICE_ROLL,
                                                       quest))
        return succeeded

    def generate_solution(
        self,
        game_state: GameState,
        context: AgentContext,
        quest: Quest,
        quest_goal: str,
    ):
        num_paragraphs = randint(1, 2)  # noqa: S311
        server_settings = get_server_settings(context=context)
        prompt = f"""Human tries to solve the role-play scene problem and it totally works. Continue role-play with what happens next in one paragraph.
- DO NOT have the human completing the scene goal of {quest_goal}

## Solution
{quest.user_problem_solutions[-1]}

## Note
- Use a tone of '{server_settings.narrative_tone}' and a narrative voice of '{server_settings.narrative_voice}'.
- write a single short paragraph."""
        
        #logging.warning("problem solution prompt: " +prompt)
        solution_block = send_story_generation(
            prompt=prompt,
            quest_name=quest.name,
            context=context,
        )
        await_streamed_block(solution_block, context)
        if num_paragraphs > 1:
            prompt = f"""Continue the the role-play about solving the problem, describe what happens in one paragraph.
- DO NOT have human completing the role-play scene goal of {quest_goal}.

## Solution
{quest.user_problem_solutions[-1]}

## Note
- Use a tone of '{server_settings.narrative_tone}' and a narrative voice of '{server_settings.narrative_voice}'.
- write a single short paragraph."""
            
            #logging.warning("problem solution prompt: " +prompt)
            solution_block = send_story_generation(
                prompt=prompt,
                quest_name=quest.name,
                context=context,
            )
            await_streamed_block(solution_block, context)

    def describe_failure(self, game_state: GameState, context: AgentContext,
                         quest: Quest):
        num_paragraphs = randint(1, 2)  # noqa: S311
        server_settings = get_server_settings(context=context)
        prompt = f"""Human tries to solve the role-play problem and it fails, continue the role-play for what happens.

## Suggested solution
{quest.user_problem_solutions[-1]}.


## Note
- Use a tone of '{server_settings.narrative_tone}' and a narrative voice of '{server_settings.narrative_voice}'.
- Write a single short paragraph."""
        #logging.warning("problem failure prompt: " +prompt)
        solution_block = send_story_generation(
            prompt=prompt,
            quest_name=quest.name,
            context=context,
        )
        await_streamed_block(solution_block, context)
        if num_paragraphs > 1:
            prompt = f"""Continue telling the role-play about failing to solve the problem, describe what happens.

## Suggested solution
{quest.user_problem_solutions[-1]}.


## Note
- Use a tone of '{server_settings.narrative_tone}' and a narrative voice of '{server_settings.narrative_voice}'.
- write a single short paragraph."""
            solution_block = send_story_generation(
                prompt=prompt,
                quest_name=quest.name,
                context=context,
            )
            #logging.warning("problem failure prompt: " +prompt)
            await_streamed_block(solution_block, context)

    def describe_non_solution(self, game_state: GameState,
                              context: AgentContext, quest: Quest):
        server_settings = get_server_settings(context=context)
        prompt = f"""Human doesn't yet try to solve the problem, instead the proposed following action. Describe what happens but do NOT solve the current problem below.
Do NOT solve the current problem.

## Current problem
{quest.current_problem}

## Current action
{quest.user_problem_solutions[-1]}.


## Note
- Use a tone of '{server_settings.narrative_tone}' and a narrative voice of '{server_settings.narrative_voice}'.
- write a single short paragraph."""
        solution_block = send_story_generation(
            prompt=prompt,
            quest_name=quest.name,
            context=context,
        )
        #logging.warning(
        #    f"problem non-solution prompt: {prompt}")
        await_streamed_block(solution_block, context)
