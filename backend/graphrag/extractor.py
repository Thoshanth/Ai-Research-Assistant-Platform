import os
import json
from groq import Groq
from dotenv import load_dotenv
from backend.logger import get_logger

load_dotenv()
logger = get_logger("graphrag.extractor")

_groq = Groq(api_key=os.getenv("GROQ_API_KEY"))


def extract_entities_and_relations(text: str) -> dict:
    """
    Sends a text chunk to the LLM and asks it to extract
    all entities and relationships in structured JSON format.

    This is the core of knowledge graph construction —
    turning unstructured text into structured graph data.

    Returns:
    {
        "entities": [
            {"name": "Python", "type": "Technology"},
            {"name": "Thoshanth", "type": "Person"},
        ],
        "relations": [
            {"source": "Thoshanth", "relation": "KNOWS", "target": "Python"},
        ]
    }
    """
    logger.debug(f"Extracting entities | text_chars={len(text)}")

    prompt = f"""Extract all entities and relationships from this text.

Text:
{text}

Return ONLY a valid JSON object with this exact structure:
{{
    "entities": [
        {{"name": "entity name", "type": "Person|Organization|Technology|Skill|Location|Concept|Project|Other"}}
    ],
    "relations": [
        {{"source": "entity1 name", "relation": "RELATION_TYPE", "target": "entity2 name"}}
    ]
}}

Common relation types: HAS_SKILL, WORKS_AT, STUDIES_AT, BUILT, KNOWS, IS_A, USED_IN, RELATED_TO, LOCATED_IN, PART_OF, CREATED_BY

Rules:
- Only extract entities clearly mentioned in the text
- Keep entity names concise and consistent
- Use UPPERCASE for relation types
- Return valid JSON only, no explanation
- If no entities found return {{"entities": [], "relations": []}}"""

    try:
        response = _groq.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=1024,
            temperature=0,
        )

        raw = response.choices[0].message.content.strip()

        # Clean up common LLM formatting issues
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        result = json.loads(raw)

        entities = result.get("entities", [])
        relations = result.get("relations", [])

        logger.info(
            f"Extraction complete | "
            f"entities={len(entities)} | "
            f"relations={len(relations)}"
        )
        return result

    except json.JSONDecodeError as e:
        logger.warning(f"JSON parse failed: {e} | returning empty")
        return {"entities": [], "relations": []}
    except Exception as e:
        logger.error(f"Extraction failed: {e}", exc_info=True)
        return {"entities": [], "relations": []}


def extract_from_chunks(chunks: list[str]) -> dict:
    """
    Runs entity extraction on a list of chunks and
    merges all results into one unified entity/relation set.

    Deduplicates entities by name so the same entity
    appearing in multiple chunks is only one graph node.
    """
    all_entities = {}  # name → type (deduped)
    all_relations = []  # list of all relations

    for i, chunk in enumerate(chunks):
        logger.info(f"Extracting chunk {i+1}/{len(chunks)}")

        if len(chunk.strip()) < 20:
            continue

        result = extract_entities_and_relations(chunk)

        # Deduplicate entities by name
        for entity in result.get("entities", []):
            name = entity.get("name", "").strip()
            entity_type = entity.get("type", "Other")
            if name and name not in all_entities:
                all_entities[name] = entity_type

        # Add all relations
        for relation in result.get("relations", []):
            source = relation.get("source", "").strip()
            rel_type = relation.get("relation", "").strip()
            target = relation.get("target", "").strip()
            if source and rel_type and target:
                all_relations.append({
                    "source": source,
                    "relation": rel_type,
                    "target": target,
                })

    logger.info(
        f"Merged extraction | "
        f"unique_entities={len(all_entities)} | "
        f"total_relations={len(all_relations)}"
    )

    return {
        "entities": [
            {"name": k, "type": v} for k, v in all_entities.items()
        ],
        "relations": all_relations,
    }