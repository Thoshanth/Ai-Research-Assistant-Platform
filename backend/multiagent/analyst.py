import os
from groq import Groq
from dotenv import load_dotenv
from backend.logger import get_logger

load_dotenv()
logger = get_logger("multiagent.analyst")

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


def analyst_node(state: dict) -> dict:
    """
    Analyst Agent Node.

    Responsibilities:
    - Read Researcher's findings deeply
    - Identify patterns and connections
    - Draw evidence-based conclusions
    - Structure a comprehensive analysis
    - Flag areas where research was insufficient

    Receives: question, research_findings
    Writes: analysis
    """
    question = state["question"]
    research_findings = state["research_findings"]
    iteration = state.get("iterations", 1)

    logger.info(
        f"Analyst starting | iteration={iteration} | "
        f"findings_chars={len(research_findings)}"
    )

    analysis_prompt = f"""You are a Senior Research Analyst.
Analyze the research findings below and provide deep, insightful analysis.

Original Question: {question}

Research Findings:
{research_findings}

Provide a comprehensive analysis covering:

1. DIRECT ANSWER
   - Answer the original question directly based on findings

2. KEY INSIGHTS
   - What patterns or connections do you see?
   - What is most significant in the findings?

3. EVIDENCE QUALITY
   - How well do the findings support your conclusions?
   - What is certain vs what is inferred?

4. LIMITATIONS
   - What important information is missing?
   - What would strengthen this analysis?

5. CONCLUSIONS
   - Summary of your analytical conclusions

Be analytical and specific. Base everything on the research findings provided.
Do not invent information not present in the findings."""

    response = _groq.chat.completions.create(
        model="llama-3.1-8b-instant",
        messages=[
            {
                "role": "system",
                "content": "You are a Senior Research Analyst. Provide deep, evidence-based analysis."
            },
            {"role": "user", "content": analysis_prompt},
        ],
        max_tokens=1500,
        temperature=0.2,
    )

    analysis = response.choices[0].message.content
    logger.info(f"Analyst complete | analysis_chars={len(analysis)}")

    return {
        "analysis": analysis,
        "messages": [{
            "agent": "Analyst",
            "iteration": iteration,
            "content": analysis[:200] + "...",
        }],
    }