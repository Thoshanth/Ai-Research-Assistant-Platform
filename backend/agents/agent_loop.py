import os
import re
from groq import Groq
from dotenv import load_dotenv
from backend.agents.tools import TOOLS_DESCRIPTION, execute_tool
from backend.logger import get_logger

load_dotenv()
logger = get_logger("agents.loop")

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))

MAX_ITERATIONS = 5  # prevent infinite loops


def parse_agent_response(response_text: str) -> tuple[str, str, str]:
    """
    Parses the LLM's response to extract:
    - thought: what the agent is thinking
    - action: which tool to call
    - action_input: what to pass to the tool

    Expected LLM output format:
    Thought: I need to search for skills in the resume
    Action: search_documents
    Action Input: skills mentioned in resume
    """
    thought = ""
    action = ""
    action_input = ""

    lines = response_text.strip().split("\n")
    for line in lines:
        line = line.strip()
        if line.startswith("Thought:"):
            thought = line[len("Thought:"):].strip()
        elif line.startswith("Action:"):
            action = line[len("Action:"):].strip()
        elif line.startswith("Action Input:"):
            action_input = line[len("Action Input:"):].strip()

    return thought, action, action_input


def run_agent_loop(
    question: str,
    document_id: int = None,
    conversation_history: str = "",
) -> dict:
    """
    The core ReAct agent loop.

    Builds a prompt, asks the LLM what to do,
    parses the response, executes the tool,
    feeds the result back, and repeats until
    the LLM calls 'finish' or we hit MAX_ITERATIONS.

    Returns the final answer and a trace of all steps taken.
    """
    logger.info(f"Agent loop started | question='{question[:60]}' | max_iter={MAX_ITERATIONS}")

    # Full trace of the agent's reasoning — useful for debugging
    # and for showing users how the agent arrived at its answer
    trace = []

    # Build the initial system prompt
    system_prompt = f"""You are an intelligent research agent with access to tools.
Your job is to answer the user's question by reasoning step by step and using tools when needed.

{TOOLS_DESCRIPTION}

IMPORTANT RULES:
- Always start with a Thought explaining your reasoning
- Then specify Action and Action Input
- After seeing the Observation, think again
- When you have enough information, use: Action: finish, Action Input: your complete answer
- Never make up information — use tools to find real data
- Be concise and precise

Format every response EXACTLY like this:
Thought: [your reasoning]
Action: [tool name]
Action Input: [tool input]
"""

    # Conversation history adds context across sessions
    history_context = ""
    if conversation_history:
        history_context = f"\nPrevious conversation:\n{conversation_history}\n"

    # Messages list grows with each iteration
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": f"{history_context}Question: {question}"
        }
    ]

    final_answer = None

    for iteration in range(1, MAX_ITERATIONS + 1):
        logger.info(f"Agent iteration {iteration}/{MAX_ITERATIONS}")

        # Ask the LLM what to do next
        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=messages,
            max_tokens=512,
            temperature=0.1,
        )

        agent_response = response.choices[0].message.content
        logger.debug(f"Agent response:\n{agent_response}")

        # Parse thought, action, input
        thought, action, action_input = parse_agent_response(agent_response)

        step = {
            "iteration": iteration,
            "thought": thought,
            "action": action,
            "action_input": action_input,
        }

        # Check if agent wants to finish
        if action.lower() == "finish":
            final_answer = action_input
            step["observation"] = "Agent completed"
            trace.append(step)
            logger.info(f"Agent finished at iteration {iteration}")
            break

        # Execute the tool
        observation = execute_tool(action, action_input, document_id)
        step["observation"] = observation
        trace.append(step)

        logger.info(f"Iteration {iteration} | action='{action}' | obs_chars={len(observation)}")

        # Add the agent's response and tool result to message history
        # so the LLM sees the full context in the next iteration
        messages.append({"role": "assistant", "content": agent_response})
        messages.append({
            "role": "user",
            "content": f"Observation: {observation}\n\nContinue reasoning. If you have enough information, use Action: finish"
        })

    # If loop ended without finish, use last observation as answer
    if final_answer is None:
        logger.warning("Agent hit max iterations without finishing")
        final_answer = trace[-1]["observation"] if trace else "Could not complete the task."

    logger.info(f"Agent loop complete | iterations={len(trace)} | answer_chars={len(final_answer)}")

    return {
        "answer": final_answer,
        "iterations": len(trace),
        "trace": trace,
    }