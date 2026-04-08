import os
import json
from groq import Groq
from dotenv import load_dotenv
from backend.logger import get_logger

load_dotenv()
logger = get_logger("multiagent.critic")

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


def critic_node(state: dict) -> dict:
    """
    Critic Agent Node.

    Responsibilities:
    - Evaluate the Analyst's conclusions rigorously
    - Check for unsupported claims
    - Identify logical gaps
    - Either APPROVE the analysis or REJECT with specific feedback
    - If rejected, provide precise guidance for the Researcher

    Receives: question, research_findings, analysis
    Writes: critique, approved, feedback_for_researcher
    """
    question = state["question"]
    research_findings = state["research_findings"]
    analysis = state["analysis"]
    iteration = state.get("iterations", 1)

    logger.info(
        f"Critic starting | iteration={iteration} | "
        f"analysis_chars={len(analysis)}"
    )

    critic_prompt = f"""You are a Critical Reviewer responsible for quality control.
Evaluate whether the analysis adequately answers the question.

Original Question: {question}

Research Findings:
{research_findings}

Analysis to Review:
{analysis}

Evaluate the analysis strictly:

1. Does it directly answer the question? (Yes/No)
2. Are all conclusions supported by the research findings? (Yes/No)
3. Are there any unsupported claims or hallucinations? (Yes/No)
4. Is important information missing that would change the answer? (Yes/No)
5. Is the analysis logically consistent? (Yes/No)

Then make a decision:
- APPROVE: if analysis is complete, accurate, and well-supported
- REJECT: if there are significant gaps or unsupported claims

Return ONLY a JSON object:
{{
    "decision": "APPROVE" or "REJECT",
    "critique": "Your detailed evaluation",
    "feedback_for_researcher": "Specific information to search for (only if REJECT)",
    "quality_score": 1-10
}}"""

    response = _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a rigorous Critical Reviewer. Be strict but fair."
            },
            {"role": "user", "content": critic_prompt},
        ],
        max_tokens=1024,
        temperature=0.1,
    )

    raw = response.choices[0].message.content.strip()

    # Parse the JSON response
    try:
        if "```" in raw:
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        result = json.loads(raw.strip())

        decision = result.get("decision", "APPROVE").upper()
        critique = result.get("critique", "")
        feedback = result.get("feedback_for_researcher", "")
        quality_score = result.get("quality_score", 7)
        approved = decision == "APPROVE"

    except json.JSONDecodeError:
        logger.warning("Critic JSON parse failed — defaulting to APPROVE")
        approved = True
        critique = raw
        feedback = ""
        quality_score = 6

    logger.info(
        f"Critic decision | approved={approved} | "
        f"quality_score={quality_score} | iteration={iteration}"
    )

    return {
        "critique": critique,
        "approved": approved,
        "feedback_for_researcher": feedback,
        "iterations": iteration + 1,
        "messages": [{
            "agent": "Critic",
            "iteration": iteration,
            "decision": "APPROVED" if approved else "REJECTED",
            "quality_score": quality_score,
            "content": critique[:200] + "...",
        }],
    }