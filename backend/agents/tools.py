import os
import ast
import operator
from groq import Groq
from dotenv import load_dotenv
from backend.rag.pipeline import query_pipeline
from backend.logger import get_logger

load_dotenv()
logger = get_logger("agents.tools")

_groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# ── Tool definitions ─────────────────────────────────────────────
# Each tool is a plain Python function.
# The agent calls these by name based on LLM decisions.

TOOLS_DESCRIPTION = """You have access to these tools:

1. search_documents(query: str)
   - Searches uploaded documents using RAG
   - Use for: questions about uploaded files, resumes, reports
   - Example: search_documents("skills mentioned in resume")

2. calculate(expression: str)
   - Evaluates a safe math expression
   - Use for: any numerical calculation
   - Example: calculate("(85 + 90 + 78) / 3")

3. summarize_document(document_id: int)
   - Returns a full summary of a specific document
   - Use for: "summarize this document" or overview requests
   - Example: summarize_document(7)

4. answer_general(question: str)
   - Answers from general knowledge, no document search
   - Use for: definitions, concepts, general facts
   - Example: answer_general("What is machine learning?")

5. finish(answer: str)
   - Returns the final answer to the user
   - Use ONLY when you have enough information to answer completely
   - Example: finish("The resume belongs to John, a CS student...")
"""


def search_documents(query: str, document_id: int = None) -> str:
    """Tool 1 — searches RAG pipeline"""
    logger.info(f"Tool: search_documents | query='{query}'")
    try:
        result = query_pipeline(
            question=query,
            document_id=document_id,
            top_k=3,
        )
        answer = result.get("answer", "No results found")
        sources = result.get("sources", [])
        source_names = [s.get("filename", "unknown") for s in sources]
        logger.info(f"search_documents returned | chars={len(answer)} | sources={source_names}")
        return f"Search result: {answer}\nSources: {', '.join(source_names)}"
    except Exception as e:
        logger.error(f"search_documents failed: {e}")
        return f"Search failed: {str(e)}"


def calculate(expression: str) -> str:
    """Tool 2 — safe math evaluator"""
    logger.info(f"Tool: calculate | expression='{expression}'")
    try:
        # Only allow safe math operations
        allowed_ops = {
            ast.Add: operator.add,
            ast.Sub: operator.sub,
            ast.Mult: operator.mul,
            ast.Div: operator.truediv,
            ast.Pow: operator.pow,
            ast.Mod: operator.mod,
        }

        def safe_eval(node):
            if isinstance(node, ast.Constant):
                return node.value
            elif isinstance(node, ast.BinOp):
                op = allowed_ops.get(type(node.op))
                if op is None:
                    raise ValueError(f"Unsupported operation: {type(node.op)}")
                return op(safe_eval(node.left), safe_eval(node.right))
            elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
                return -safe_eval(node.operand)
            else:
                raise ValueError(f"Unsupported expression type: {type(node)}")

        tree = ast.parse(expression, mode='eval')
        result = safe_eval(tree.body)
        logger.info(f"calculate result: {result}")
        return f"Calculation result: {result}"
    except Exception as e:
        logger.error(f"calculate failed: {e}")
        return f"Calculation error: {str(e)}"


def summarize_document(document_id: int) -> str:
    """Tool 3 — summarizes a full document"""
    logger.info(f"Tool: summarize_document | doc_id={document_id}")
    try:
        from backend.database.db import SessionLocal, DocumentRecord
        db = SessionLocal()
        try:
            record = db.query(DocumentRecord).filter(
                DocumentRecord.id == document_id
            ).first()
            if not record:
                return f"Document {document_id} not found"
            text_preview = record.extracted_text[:3000]
        finally:
            db.close()

        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Summarize the following document clearly and concisely in 3-5 sentences."
                },
                {
                    "role": "user",
                    "content": f"Document content:\n{text_preview}"
                }
            ],
            max_tokens=512,
            temperature=0.1,
        )
        summary = response.choices[0].message.content
        logger.info(f"summarize_document complete | chars={len(summary)}")
        return f"Document summary: {summary}"
    except Exception as e:
        logger.error(f"summarize_document failed: {e}")
        return f"Summarization failed: {str(e)}"


def answer_general(question: str) -> str:
    """Tool 4 — answers from general knowledge"""
    logger.info(f"Tool: answer_general | question='{question[:50]}'")
    try:
        response = _groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {
                    "role": "system",
                    "content": "Answer the question clearly from your general knowledge."
                },
                {"role": "user", "content": question}
            ],
            max_tokens=512,
            temperature=0.3,
        )
        answer = response.choices[0].message.content
        logger.info(f"answer_general complete | chars={len(answer)}")
        return answer
    except Exception as e:
        logger.error(f"answer_general failed: {e}")
        return f"Failed to answer: {str(e)}"


def execute_tool(tool_name: str, tool_input: str, document_id: int = None) -> str:
    """
    Central dispatcher — receives tool name and input from the agent loop
    and calls the right function.
    """
    logger.info(f"Executing tool | name='{tool_name}' | input='{tool_input[:60]}'")

    if tool_name == "search_documents":
        return search_documents(tool_input, document_id)
    elif tool_name == "calculate":
        return calculate(tool_input)
    elif tool_name == "summarize_document":
        try:
            doc_id = int(tool_input.strip())
        except ValueError:
            doc_id = document_id or 1
        return summarize_document(doc_id)
    elif tool_name == "answer_general":
        return answer_general(tool_input)
    else:
        logger.warning(f"Unknown tool requested: '{tool_name}'")
        return f"Unknown tool: {tool_name}"