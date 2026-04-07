import json
import networkx as nx
from pathlib import Path
from backend.logger import get_logger

logger = get_logger("graphrag.store")

GRAPH_DIR = Path("graph_data")
GRAPH_DIR.mkdir(exist_ok=True)


def get_graph_path(document_id: int) -> Path:
    return GRAPH_DIR / f"graph_doc_{document_id}.json"


def build_graph(extracted_data: dict, document_id: int) -> nx.DiGraph:
    """
    Builds a directed graph from extracted entities and relations.

    Directed graph (DiGraph) means edges have direction:
    Thoshanth → KNOWS → Python
    (not the same as Python → KNOWS → Thoshanth)

    Each node stores its entity type as metadata.
    Each edge stores the relation type as metadata.
    """
    G = nx.DiGraph()

    # Add entity nodes
    for entity in extracted_data.get("entities", []):
        name = entity["name"]
        entity_type = entity["type"]
        G.add_node(name, type=entity_type, document_id=document_id)

    # Add relation edges
    for relation in extracted_data.get("relations", []):
        source = relation["source"]
        rel_type = relation["relation"]
        target = relation["target"]

        # Add nodes if they don't exist yet
        if source not in G:
            G.add_node(source, type="Unknown", document_id=document_id)
        if target not in G:
            G.add_node(target, type="Unknown", document_id=document_id)

        G.add_edge(source, target, relation=rel_type)

    logger.info(
        f"Graph built | nodes={G.number_of_nodes()} | "
        f"edges={G.number_of_edges()} | doc_id={document_id}"
    )
    return G


def save_graph(G: nx.DiGraph, document_id: int):
    """
    Saves graph to disk as JSON using node-link format.
    This makes the graph persistent across server restarts.
    """
    path = get_graph_path(document_id)
    data = nx.node_link_data(G)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    logger.info(f"Graph saved | path='{path}'")


def load_graph(document_id: int) -> nx.DiGraph:
    """
    Loads a previously saved graph from disk.
    Raises ValueError if graph doesn't exist yet.
    """
    path = get_graph_path(document_id)
    if not path.exists():
        raise ValueError(
            f"No graph found for document {document_id}. "
            f"Run POST /graphrag/build/{document_id} first."
        )
    with open(path, "r") as f:
        data = json.load(f)
    G = nx.node_link_graph(data, directed=True)
    logger.info(
        f"Graph loaded | nodes={G.number_of_nodes()} | "
        f"edges={G.number_of_edges()} | doc_id={document_id}"
    )
    return G


def search_graph(
    G: nx.DiGraph,
    query_entities: list[str],
    max_hops: int = 2,
) -> dict:
    """
    Traverses the graph starting from query entities.

    max_hops=2 means we look at:
    - Direct connections (1 hop): Thoshanth → Python
    - Connections of connections (2 hops): Python → FastAPI → REST

    Returns all found nodes and edges as context.
    """
    found_nodes = set()
    found_edges = []

    for entity in query_entities:
        # Find the closest matching node (case-insensitive)
        matched_node = None
        for node in G.nodes():
            if entity.lower() in node.lower() or node.lower() in entity.lower():
                matched_node = node
                break

        if not matched_node:
            logger.debug(f"Entity '{entity}' not found in graph")
            continue

        logger.info(f"Traversing from node '{matched_node}'")

        # BFS traversal up to max_hops
        visited = {matched_node}
        current_frontier = {matched_node}

        for hop in range(max_hops):
            next_frontier = set()
            for node in current_frontier:
                # Outgoing edges
                for _, neighbor, data in G.out_edges(node, data=True):
                    found_edges.append({
                        "source": node,
                        "relation": data.get("relation", "RELATED_TO"),
                        "target": neighbor,
                    })
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)

                # Incoming edges
                for neighbor, _, data in G.in_edges(node, data=True):
                    found_edges.append({
                        "source": neighbor,
                        "relation": data.get("relation", "RELATED_TO"),
                        "target": node,
                    })
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)

            found_nodes.update(visited)
            current_frontier = next_frontier

    # Build context from found nodes and edges
    node_info = []
    for node in found_nodes:
        node_data = G.nodes[node]
        node_info.append(
            f"{node} (type: {node_data.get('type', 'Unknown')})"
        )

    edge_info = []
    seen_edges = set()
    for edge in found_edges:
        key = (edge["source"], edge["relation"], edge["target"])
        if key not in seen_edges:
            edge_info.append(
                f"{edge['source']} --[{edge['relation']}]--> {edge['target']}"
            )
            seen_edges.add(key)

    logger.info(
        f"Graph search complete | "
        f"nodes_found={len(found_nodes)} | "
        f"edges_found={len(seen_edges)}"
    )

    return {
        "nodes": node_info,
        "edges": edge_info,
        "entities_searched": query_entities,
    }


def get_graph_summary(G: nx.DiGraph) -> dict:
    """
    Returns a summary of the entire graph for exploration.
    """
    entity_types = {}
    for node, data in G.nodes(data=True):
        t = data.get("type", "Unknown")
        entity_types[t] = entity_types.get(t, 0) + 1

    relation_types = {}
    for _, _, data in G.edges(data=True):
        r = data.get("relation", "UNKNOWN")
        relation_types[r] = relation_types.get(r, 0) + 1

    # Most connected nodes (hubs)
    degree_dict = dict(G.degree())
    top_nodes = sorted(
        degree_dict.items(), key=lambda x: x[1], reverse=True
    )[:10]

    return {
        "total_nodes": G.number_of_nodes(),
        "total_edges": G.number_of_edges(),
        "entity_types": entity_types,
        "relation_types": relation_types,
        "most_connected_entities": [
            {"entity": n, "connections": d} for n, d in top_nodes
        ],
    }