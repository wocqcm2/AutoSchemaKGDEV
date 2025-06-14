from neo4j import GraphDatabase
import faiss
import numpy as np
from collections import defaultdict
from typing import List
import time
import logging
from atlas_rag.reader.llm_generator import LLMGenerator
from atlas_rag.retrieval.embedding_model import BaseEmbeddingModel
from atlas_rag.billion.retriever.base import BaseLargeKGEdgeRetriever

class LargeKGToGRetriever(BaseLargeKGEdgeRetriever):
    def __init__(self, keyword: str, neo4j_driver: GraphDatabase, 
                 llm_generator: LLMGenerator, sentence_encoder: BaseEmbeddingModel, 
                 node_index: faiss.Index, logger: logging.Logger = None):
        """
        Initialize the LargeKGToGRetriever for billion-level KG retrieval using Neo4j.

        Args:
            keyword (str): Identifier for the KG dataset (e.g., 'cc_en').
            neo4j_driver (GraphDatabase): Neo4j driver for database access.
            llm_generator (LLMGenerator): LLM for NER, rating, and reasoning.
            sentence_encoder (BaseEmbeddingModel): Encoder for generating embeddings.
            node_index (faiss.Index): FAISS index for node embeddings.
            logger (Logger, optional): Logger for verbose output.
        """
        self.keyword = keyword
        self.neo4j_driver = neo4j_driver
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.node_faiss_index = node_index
        self.verbose = logger is not None
        self.logger = logger 

    def convert_numeric_id_to_name(self, numeric_id):
        if numeric_id.isdigit():
            return self.gds_driver.util.asNode(self.gds_driver.find_node_id(["Node"], {"numeric_id": numeric_id})).get('name')
        else:
            return numeric_id

    def ner(self, text: str) -> List[str]:
        """
        Extract named entities from the query text using the LLM.

        Args:
            text (str): The query text.

        Returns:
            List[str]: List of extracted entities.
        """
        entities = self.llm_generator.large_kg_ner(text)
        if self.verbose:
            self.logger.info(f"Extracted entities: {entities}")
        return entities

    def retrieve_topk_nodes(self, query: str, top_k_nodes: int = 5) -> List[str]:
        """
        Retrieve top-k nodes similar to entities in the query.

        Args:
            query (str): The user query.
            top_k_nodes (int): Number of nodes to retrieve per entity.

        Returns:
            List[str]: List of node numeric_ids.
        """
        start_time = time.time()
        entities = self.ner(query)
        if self.verbose:
            ner_time = time.time() - start_time
            self.logger.info(f"NER took {ner_time:.2f} seconds, entities: {entities}")
        if not entities:
            entities = [query]

        initial_nodes = []
        for entity in entities:
            entity_embedding = self.sentence_encoder.encode([entity])
            D, I = self.node_faiss_index.search(entity_embedding, top_k_nodes)
            if self.verbose:
                self.logger.info(f"Entity: {entity}, FAISS Distances: {D}, Indices: {I}")
            initial_nodes.extend([str(i) for i in I[0]])
        # no need filtering as ToG pruning will handle it.
        topk_nodes = list(set(initial_nodes))
        if len(topk_nodes) > 2 * len(entities):
            topk_nodes = topk_nodes[:2 * len(entities)]
        if self.verbose:
            self.logger.info(f"Top-k nodes: {topk_nodes}")
        return topk_nodes

    def expand_paths(self, P: List[List[str]], width: int, query: str) -> List[List[str]]:
        """
        Expand each path by adding neighbors of the last node.

        Args:
            P (List[List[str]]): Current list of paths, where each path is a list of alternating node_ids and relation types.

        Returns:
            List[List[str]]: List of expanded paths.
        """
        last_node_ids = [path[-1] for path in P if path]
        if not last_node_ids:
            return []

        with self.neo4j_driver.session() as session:
            # Query outgoing relationships
            start_time = time.time()
            outgoing_query = """
            MATCH (n:Node)-[r:Relation]->(m:Node)
            WHERE n.numeric_id IN $last_node_ids
            RETURN n.numeric_id AS source, r.relation AS rel_type, m.numeric_id AS target, m.name AS target_name, 'Node' AS target_type
            """
            outgoing_result = session.run(outgoing_query, last_node_ids=last_node_ids)
            outgoing = [(record["source"], record["rel_type"], record["target"], record["target_name"], record["target_type"]) 
                        for record in outgoing_result]
            if self.verbose:
                outgoing_time = time.time() - start_time
                self.logger.info(f"Outgoing relationships query took {outgoing_time:.2f} seconds, count: {len(outgoing)}")

            # Query incoming relationships
            start_time = time.time()
            incoming_query = """
            MATCH (m:Node)-[r:Relation]->(n:Node)
            WHERE m.numeric_id IN $last_node_ids
            RETURN m.numeric_id AS source, r.relation AS rel_type, n.numeric_id AS target, n.name AS target_name, 'Node' AS target_type
            """
            incoming_result = session.run(incoming_query, last_node_ids=last_node_ids)
            incoming = [(record["source"], record["rel_type"], record["target"], record["target_name"], record["target_type"]) 
                        for record in incoming_result]
            if self.verbose:
                incoming_time = time.time() - start_time
                self.logger.info(f"Incoming relationships query took {incoming_time:.2f} seconds, count: {len(incoming)}")

            # Query outgoing Node -> Text relationships
            start_time = time.time()
            outgoing_text_query = """
            MATCH (n:Node)-[r:Source]->(t:Text)
            WHERE n.numeric_id IN $last_node_ids
            RETURN n.numeric_id AS source, 'Source' AS rel_type, t.numeric_id AS target, 'Text' AS target_type
            """
            outgoing_text_result = session.run(outgoing_text_query, last_node_ids=last_node_ids)
            outgoing_text = [(record["source"], record["rel_type"], record["target"], record["target_name"], record["target_type"])
                            for record in outgoing_text_result]
            if self.verbose:
                outgoing_text_time = time.time() - start_time
                self.logger.info(f"Outgoing Node->Text relationships query took {outgoing_text_time:.2f} seconds, count: {len(outgoing_text)}")

            # Query incoming Text -> Node relationships
            start_time = time.time()
            incoming_text_query = """
            MATCH (t:Text)-[r:Source]->(n:Node)
            WHERE n.numeric_id IN $last_node_ids
            RETURN t.numeric_id AS source, 'Source' AS rel_type, n.numeric_id AS target, 'Text' AS target_type
            """
            incoming_text_result = session.run(incoming_text_query, last_node_ids=last_node_ids)
            incoming_text = [(record["source"], record["rel_type"], record["target"], record["target_type"])
                            for record in incoming_text_result]
            if self.verbose:
                incoming_text_time = time.time() - start_time
                self.logger.info(f"Incoming Text->Node relationships query took {incoming_text_time:.2f} seconds, count: {len(incoming_text)}")

        last_node_ids_to_new_paths = defaultdict(list)
        for path in P:
            last_node = path[-1]  # Tuple (id, name, type)
            last_node_id = last_node[0]
            # Outgoing Node -> Node
            for source, rel_type, target, target_name, target_type in outgoing:
                if source == last_node_id and target not in [n[0] for n in path if isinstance(n, tuple)]:
                    new_path = path + [rel_type, (target, target_name, target_type)]
                    last_node_ids_to_new_paths[last_node_id].append(new_path)
            # Outgoing Node -> Text
            for source, rel_type, target, target_name, target_type in outgoing_text:
                if source == last_node_id and target not in [n[0] for n in path if isinstance(n, tuple)]:
                    new_path = path + [rel_type, (target, target_name, target_type)]
                    last_node_ids_to_new_paths[last_node_id].append(new_path)
            # Incoming Node -> Node
            for source, rel_type, target, source_name, source_type in incoming:
                if target == last_node_id and source not in [n[0] for n in path if isinstance(n, tuple)]:
                    new_path = path + [rel_type, (source, source_name, source_type)]
                    last_node_ids_to_new_paths[last_node_id].append(new_path)
            # Incoming Text -> Node
            for source, rel_type, target, source_name, source_type in incoming_text:
                if target == last_node_id and source not in [n[0] for n in path if isinstance(n, tuple)]:
                    new_path = path + [rel_type, (source, source_name, source_type)]
                    last_node_ids_to_new_paths[last_node_id].append(new_path)

        total_new_paths = sum(len(new_paths) for new_paths in last_node_ids_to_new_paths.values())
        top_width_new_paths = []
        if total_new_paths > len(last_node_ids) * width:
            for last_node_id, new_paths in last_node_ids_to_new_paths.items():
                if len(new_paths) > width:
                    path_embeddings = self.sentence_encoder.encode([self.path_to_string(path) for path in new_paths])
                    query_embeddings = self.sentence_encoder.encode([query])
                    scores = np.dot(path_embeddings, query_embeddings.T).flatten()
                    top_indices = np.argsort(scores)[-width:]
                    new_paths = [new_paths[i] for i in top_indices]
                top_width_new_paths.extend(new_paths)
        else:
            for new_paths in last_node_ids_to_new_paths.values():
                top_width_new_paths.extend(new_paths)

        if self.verbose:
            self.logger.info(f"Expanded paths count: {len(top_width_new_paths)}")
        return top_width_new_paths

    def path_to_string(self, path: List[str]) -> str:
        """
        Convert a path to a human-readable string for LLM rating.

        Args:
            path (List[str]): Path as a list of node_ids and relation types.

        Returns:
            str: String representation of the path.
        """
        if len(path) < 1:
            return ""
        path_str = []
        with self.neo4j_driver.session() as session:
            for i in range(0, len(path), 2):
                node_id = path[i]
                result = session.run("MATCH (n:Node {numeric_id: $node_id}) RETURN n.name", node_id=node_id)
                node_name = result.single()["n.name"] if result.single() else node_id
                if i + 1 < len(path):
                    rel_type = path[i + 1]
                    path_str.append(f"{node_name} ---> {rel_type} --->")
                else:
                    path_str.append(node_name)
        return " ".join(path_str).strip()

    def prune(self, query: str, P: List[List[str]], topN: int = 5) -> List[List[str]]:
        """
        Prune paths to keep the top N based on LLM relevance ratings.

        Args:
            query (str): The user query.
            P (List[List[str]]): List of paths to prune.
            topN (int): Number of paths to retain.

        Returns:
            List[List[str]]: Top N paths.
        """
        path_strings = [self.path_to_string(path) for path in P]
        ratings = []
        for path_str in path_strings:
            prompt = f"Please rate the following path based on relevance to the query (1-5, 0 if not relevant).\nQuery: {query}\nPath: {path_str}"
            messages = [
                {"role": "system", "content": "Provide a single integer rating (0-5)."},
                {"role": "user", "content": prompt}
            ]
            response = self.llm_generator._generate_response(messages)
            rating = int(response.strip()) if response.strip().isdigit() else 0
            ratings.append(rating)
        sorted_paths = [path for _, path in sorted(zip(ratings, P), key=lambda x: x[0], reverse=True)]
        top_paths = sorted_paths[:topN]
        if self.verbose:
            self.logger.info(f"Pruned to top {topN} paths: {[self.path_to_string(p) for p in top_paths]}")
        return top_paths

    def reasoning(self, query: str, P: List[List[str]]) -> bool:
        """
        Check if the current paths are sufficient to answer the query.

        Args:
            query (str): The user query.
            P (List[List[str]]): Current list of paths.

        Returns:
            bool: True if sufficient, False otherwise.
        """
        triples = []
        with self.neo4j_driver.session() as session:
            for path in P:
                for i in range(0, len(path) - 2, 2):
                    node1_id = path[i]
                    rel = path[i + 1]
                    node2_id = path[i + 2]
                    node1_result = session.run("MATCH (n:Node {numeric_id: $node_id}) RETURN n.name", node_id=node1_id)
                    node1_name = node1_result.single()["n.name"] if node1_result.single() else node1_id
                    node2_result = session.run("MATCH (n:Node {numeric_id: $node_id}) RETURN n.name", node_id=node2_id)
                    node2_name = node2_result.single()["n.name"] if node2_result.single() else node2_id
                    triples.append(f"({node1_name}, {rel}, {node2_name})")
        triples_str = ". ".join(triples)
        prompt = f"Are these triples sufficient to answer the query?\nQuery: {query}\nTriples: {triples_str}"
        messages = [
            {"role": "system", "content": "Answer Yes or No."},
            {"role": "user", "content": prompt}
        ]
        response = self.llm_generator._generate_response(messages)
        if self.verbose:
            self.logger.info(f"Reasoning result: {response}")
        return "yes" in response.lower()

    def retrieve_passages(self, query: str, topN: int = 5, Dmax: int = 3, Wmax: int = 3) -> List[str]:
        """
        Retrieve the top N paths to answer the query.

        Args:
            query (str): The user query.
            topN (int): Number of paths to return.
            Dmax (int): Maximum depth of path expansion.
            Wmax (int): Maximum width of path expansion.

        Returns:
            List[str]: List of triples as strings.
        """
        if self.verbose:
            self.logger.info(f"Retrieving paths for query: {query}")
        
        initial_nodes = self.retrieve_topk_nodes(query, top_k_nodes=topN)
        if not initial_nodes:
            if self.verbose:
                self.logger.info("No initial nodes found.")
            return []

        P = [[node] for node in initial_nodes]
        for D in range(Dmax + 1):
            if self.verbose:
                self.logger.info(f"Depth {D}, Current paths: {len(P)}")
            P = self.expand_paths(P, Wmax, query)
            if not P:
                if self.verbose:
                    self.logger.info("No paths to expand.")
                break
            P = self.prune(query, P, topN)
            if self.reasoning(query, P):
                if self.verbose:
                    self.logger.info("Paths sufficient, stopping expansion.")
                break

        # Extract final triples
        triples = []
        with self.neo4j_driver.session() as session:
            for path in P:
                for i in range(0, len(path) - 2, 2):
                    node1_id = path[i]
                    rel = path[i + 1]
                    node2_id = path[i + 2]
                    node1_result = session.run("MATCH (n:Node {numeric_id: $node_id}) RETURN n.name", node_id=node1_id)
                    node1_name = node1_result.single()["n.name"] if node1_result.single() else node1_id
                    node2_result = session.run("MATCH (n:Node {numeric_id: $node_id}) RETURN n.name", node_id=node2_id)
                    node2_name = node2_result.single()["n.name"] if node2_result.single() else node2_id
                    triples.append(f"({node1_name}, {rel}, {node2_name})")
        
        if self.verbose:
            self.logger.info(f"Final triples: {triples}")
        return triples
