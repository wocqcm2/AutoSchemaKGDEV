from difflib import get_close_matches
import json
import json
import faiss
from openai import OpenAI, NOT_GIVEN
from pathlib import Path
from neo4j import GraphDatabase
from copy import deepcopy
import time
import jsonschema
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

from graphdatascience import GraphDataScience
import logging
from typing import Dict
from sentence_transformers import SentenceTransformer
from prompt_template import ner_prompt, validate_keyword_output, keyword_filtering_prompt
from tenacity import retry, stop_after_delay, stop_after_attempt, wait_fixed
import string

keyword_to_port_dict = {
        "en_simple_wiki_v0": 8011,
        "pes2o_abstract": 8012,
        "cc_en": 8013,
        "demo": 7687,
}

current_dir = Path.cwd()
keyword_to_index_paths = {
    'cc_en':{
        'node':f"{current_dir.parent.parent}/import/Dulce/triples_csv/triple_nodes_cc_en_from_json.index",
        'edge':f"{current_dir.parent.parent}/import/Dulce/triples_csv/triple_edges_cc_en_from_json.index",
        'text':f"{current_dir.parent.parent}/import/Dulce/triples_csv/text_nodes_cc_en_from_json_with_emb.index",
    },
    'pes2o_abstract':{
        'node':f"{current_dir.parent.parent}/import/Dulce/triples_csv/triple_nodes_pes2o_abstract_from_json_non_norm.index",
        'edge':f"{current_dir.parent.parent}/import/Dulce/triples_csv/triple_edges_pes2o_abstract_from_json_non_norm.index",
        'text':f"{current_dir.parent.parent}/import/Dulce/triples_csv/text_nodes_pes2o_abstract_from_json_with_emb_non_norm.index",
    },
    'en_simple_wiki_v0':{
        'node':f"{current_dir.parent.parent}/import/Dulce/triples_csv/triple_nodes_en_simple_wiki_v0_from_json_non_norm.index",
        'edge':f"{current_dir.parent.parent}/import/Dulce/triples_csv/triple_edges_en_simple_wiki_v0_from_json_non_norm.index",
        'text':f"{current_dir.parent.parent}/import/Dulce/triples_csv/text_nodes_en_simple_wiki_v0_from_json_with_emb_non_norm.index",
    },
    'demo':{
        'node':f"{current_dir.parent.parent}/import/Dulce/triples_csv/triple_nodes_Dulce_from_json.index",
        'edge':f"{current_dir.parent.parent}/import/Dulce/triples_csv/triple_edges_Dulce_from_json.index",
        'text':f"{current_dir.parent.parent}/import/Dulce/triples_csv/text_nodes_Dulce_from_json.index",
    },
}

def build_projection_graph(driver: GraphDataScience, keyword = 'cc_en'):
    if keyword == 'cc_en':
        filtered_graph =  "largekgrag_graph"
        project_graph_1 = "intermediate_graph"

        is_project_graph_1_exist = False

        result = driver.graph.list()
        for index, row in result.iterrows():
            if row['graphName'] == project_graph_1:
                is_project_graph_1_exist = True
        
        is_filtered_graph_exist = False
        result = driver.graph.list()
        for index, row in result.iterrows():
            if row['graphName'] == filtered_graph:
                is_filtered_graph_exist = True
        if not is_project_graph_1_exist and not is_filtered_graph_exist:
            start_time = time.time()
            node_properties = ["Node"]
            relation_projection = [ "Relation"]
            result = driver.graph.project(
                project_graph_1,
                node_properties,
                relation_projection
            )
            graph = driver.graph.get(project_graph_1)
            driver.degree.mutate(graph, mutateProperty="degree")
            logging.info(f"Projection graph {project_graph_1} created in {time.time() - start_time:.2f} seconds")
        if not is_filtered_graph_exist:
            # Compute node degrees and filter
            graph = driver.graph.get(project_graph_1)
            start_time = time.time()
            logging.info(f"Degree distribution: {graph.degree_distribution()}")
            
            degree_distribution = graph.degree_distribution()
            # if filter to have 99.99% of the nodes
            # degree_threshold = degree_distribution['p999']
            degree_threshold = 5000.0
            driver.graph.filter(
                filtered_graph,
                graph,
                node_filter=f"n.degree <= {degree_threshold}",  # Exclude nodes with degree > p99.9
                relationship_filter="*"  # Keep all relationships between remaining nodes
            )
            driver.graph.drop(project_graph_1)
            logging.info(f"Filtered graph {filtered_graph} created in {time.time() - start_time:.2f} seconds")
    else:
        project_graph_1 = "largekgrag_graph"
        is_project_graph_1_exist = False
        # is_project_graph_2_exist = False
        result = driver.graph.list()
        for index, row in result.iterrows():
            if row['graphName'] == project_graph_1:
                is_project_graph_1_exist = True
            # if row['graphName'] == project_graph_2:
            #     is_project_graph_2_exist = True
        
        if not is_project_graph_1_exist:
            start_time = time.time()
            node_properties = ["Node"]
            relation_projection = [ "Relation"]
            result = driver.graph.project(
                project_graph_1,
                node_properties,
                relation_projection
            )
            graph = driver.graph.get(project_graph_1)
            logging.info(f"Projection graph {project_graph_1} created in {time.time() - start_time:.2f} seconds")

def build_neo4j_label_index(driver):
    with driver.session() as session:
        index_name = f"NodeNumericIDIndex"
        # Check if the index already exists
        existing_indexes = session.run("SHOW INDEXES").data()
        index_exists = any(index['name'] == index_name for index in existing_indexes)
        # Drop the index if it exists
        if not index_exists:
            start_time = time.time()
            session.run(f"CREATE INDEX {index_name} FOR (n:Node) ON (n.numeric_id)")
            print(f"Index {index_name} created in {time.time() - start_time:.2f} seconds")
            
        index_name = f"TextNumericIDIndex"
        index_exists = any(index['name'] == index_name for index in existing_indexes)
        if not index_exists:
            start_time = time.time()
            session.run(f"CREATE INDEX {index_name} FOR (t:Text) ON (t.numeric_id)")
            print(f"Index {index_name} created in {time.time() - start_time:.2f} seconds")
        
        index_name = f"EntityEventEdgeNumericIDIndex"
        index_exists = any(index['name'] == index_name for index in existing_indexes)
        if not index_exists:
            start_time = time.time()
            session.run(f"CREATE INDEX {index_name} FOR ()-[r:Relation]-() on (r.numeric_id)")
            print(f"Index {index_name} created in {time.time() - start_time:.2f} seconds")

def load_indexes(path_dict):
    for key, value in path_dict.items():
        if key == 'node':
            node_index = faiss.read_index(value, faiss.IO_FLAG_MMAP)
            logging.info(f"Node index loaded from {value}")
        elif key == 'edge':
            edge_index = faiss.read_index(value, faiss.IO_FLAG_MMAP)
            logging.info(f"Edge index loaded from {value}")
        elif key == 'text':
            passage_index = faiss.read_index(value, faiss.IO_FLAG_MMAP)
            logging.info(f"Passage index loaded from {value}")
    return node_index, edge_index, passage_index


class MiniLM():
    def __init__(self,sentence_encoder:SentenceTransformer):
        self.sentence_encoder = sentence_encoder

    def encode(self, query, **kwargs):
        return self.sentence_encoder.encode(query)
    
class LLMGenerator():
    def __init__(self, pipeline, client: OpenAI, is_gpt=False, is_api=False):
        self.is_gpt = is_gpt
        self.is_api = is_api
        self.client = client
        self.pipeline = pipeline
        self.model_name = 'meta-llama/Llama-3.3-70B-Instruct-Turbo'

    def _generate_response(self, messages, do_sample=True, 
                           max_new_tokens=32768,
                           temperature = 0.7,
                           frequency_penalty = None,
                           response_format = None
                           ):
        if self.is_gpt:
            return self.gpt_api_call(messages)
        else:
            if self.is_api:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_new_tokens,
                    temperature = temperature,
                    top_p=NOT_GIVEN,
                    frequency_penalty= NOT_GIVEN if frequency_penalty is None else frequency_penalty,
                    response_format = response_format if response_format is not None else {"type": "text"},
                )
                logging.info(f"API response: {response}")
                return response.choices[0].message.content
            
            response = self.pipeline(messages, do_sample=do_sample, max_new_tokens=max_new_tokens)
            return response[0]["generated_text"][-1]["content"]
    

    def generate(self, question, max_new_tokens=2048):
        messages = [
            {"role": "system", "content": self.cot_system_instruction_no_doc},
            {"role": "user", "content": question},
        ]
        return self._generate_response(messages, max_new_tokens=max_new_tokens)

    def generate_with_context(self, question, context, max_new_tokens=2048):
        messages = [
            {"role": "system", "content": self.cot_system_instruction},
            {"role": "user", "content": f"{context}\n\n{question}"},
        ]
        return self._generate_response(messages, max_new_tokens=max_new_tokens)

    def generate_with_context_kg(self, question, context, max_new_tokens=2048):
        messages = [
            {"role": "system", "content": self.cot_system_instruction_kg},
            {"role": "user", "content": f"{context}\n\n{question}"},
        ]
        return self._generate_response(messages, max_new_tokens=max_new_tokens)
    
    def generate_with_custom_messages(self, custom_messages, do_sample=True, max_new_tokens=1024, temperature=0.8, frequency_penalty = None, response_format = None):
        return self._generate_response(custom_messages, do_sample, max_new_tokens, temperature, frequency_penalty, response_format)
    
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def ner(self, text):
        messages = deepcopy(ner_prompt)
        messages.append(
            {
                "role": "user", 
                "content": f"[[ ## question ## ]]\n{text}" 
            }
        )
        
        # Generate raw response from LLM
        raw_response = self._generate_response(messages, max_new_tokens=4096, temperature=0.7, frequency_penalty=1.1, response_format={"type": "json_object"})
        
        try:
            # Validate and clean the response
            cleaned_data = validate_keyword_output(raw_response)
            logging.info(f"NER validated keywords: {cleaned_data['keywords']}")
            return cleaned_data['keywords']
        
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            logging.error(f"Keyword validation failed: {str(e)}")
            return []  # Fallback to empty list or raise custom exception
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def filter_keywords_with_entity(self, question, keywords):
        messages = deepcopy(keyword_filtering_prompt)
        
        messages.append({
            "role": "user",
            "content": f"""[[ ## question ## ]]
            {question}
            [[ ## keywords_before_filter ## ]]
            {keywords}"""
        })
        
        try:
            logging.info(f"largekgRAG: Before Filter Keywords: {keywords}")
            response = self.generate_with_custom_messages(messages, response_format={"type": "json_object"}, temperature=0.0, max_new_tokens=2048)
            
            # Validate and clean the response
            cleaned_data = validate_keyword_output(response)
            logging.info(f"Filtered Keywords: {cleaned_data['keywords']}")
            
            return cleaned_data['keywords']
        except Exception as e:
            logging.error(f"Unexpected Error: {str(e)}")
            return keywords
        
class LargeKGRetriever_API():
    def __init__(self, neo4j_drivers: Dict[str, GraphDatabase], gds_drivers: Dict[str, GraphDataScience], 
                llm_generator:LLMGenerator, sentence_encoder:MiniLM, 
                node_indexes:Dict[str, faiss.Index], passage_indexes:Dict[str, faiss.Index], keyword: str): 
        # load each kg resources
        self.keyword = keyword
        self.neo4j_drivers = neo4j_drivers
        self.gds_drivers = gds_drivers
        self.node_indexes = node_indexes
        # self.edge_indexes = edge_indexes
        self.passage_indexes = passage_indexes
        
        # istantiate one kg resources
        self.neo4j_driver = self.neo4j_drivers[keyword]
        self.gds_driver = self.gds_drivers[keyword]
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.node_faiss_index = self.node_indexes[keyword]
        # self.edge_faiss_index = self.edge_indexes[keyword]
        self.passage_faiss_index = self.passage_indexes[keyword]
        self.verbose = False
        
        self.ppr_weight_threshold = 0.00005

    def set_resources(self, keyword):
        self.neo4j_driver = self.neo4j_drivers[keyword]
        self.gds_driver = self.gds_drivers[keyword]
        self.node_faiss_index = self.node_indexes[keyword]
        # self.edge_faiss_index = self.edge_indexes[keyword]
        self.passage_faiss_index = self.passage_indexes[keyword]
        self.keyword = keyword
    
    def set_model(self, model):
        self.llm_generator.model_name = model
    
    def ner(self, text):
        return self.llm_generator.ner(text)
    
    def convert_numeric_id_to_name(self, numeric_id):
        if numeric_id.isdigit():
            return self.gds_driver.util.asNode(self.gds_driver.find_node_id(["Node"], {"numeric_id": numeric_id})).get('name')
        else:
            return numeric_id
    def has_intersection(self, word_set, input_string):
        cleaned_string = input_string.translate(str.maketrans('', '', string.punctuation)).lower()
        if self.keyword == 'cc_en':
            # Check if any phrase in word_set is a substring of cleaned_string
            for phrase in word_set:
                if phrase in cleaned_string:
                    return True
            return False
        else:
            # Check if any word in word_set is present in the cleaned_string's words
            words_in_string = set(cleaned_string.split())
            return not word_set.isdisjoint(words_in_string)

    def pagerank(self, personalization_dict, topN=5, sampling_area=200, verbose=False):
        graph = self.gds_driver.graph.get('largekgrag_graph')
        node_count = graph.node_count()
        sampling_ratio = sampling_area / node_count
        aggregation_node_dict = []
        ppr_weight_threshold = self.ppr_weight_threshold
        start_time = time.time()
        
        # Pre-filter nodes based on ppr_weight threshold
        # Precompute word sets based on keyword
        if self.keyword == 'cc_en':
            filtered_personalization = {
                node_id: ppr_weight 
                for node_id, ppr_weight in personalization_dict.items() 
                if ppr_weight >= ppr_weight_threshold
            }
            stop_words = set(stopwords.words('english'))
            word_set_phrases = set()
            word_set_words = set()
            for node_id, ppr_weight in filtered_personalization.items():
                name = self.gds_driver.util.asNode(node_id)['name']
                if name:
                    cleaned_phrase = name.translate(str.maketrans('', '', string.punctuation)).lower().strip()
                    if cleaned_phrase:
                        # Process for 'cc_en': remove stop words and add cleaned phrase
                        filtered_words = [word for word in cleaned_phrase.split() if word not in stop_words]
                        if filtered_words:
                            cleaned_phrase_filtered = ' '.join(filtered_words)
                            word_set_phrases.add(cleaned_phrase_filtered)
        
            word_set = word_set_phrases if self.keyword == 'cc_en' else word_set_words
            logging.info(f"Optimized word set: {word_set}")
        else:
            filtered_personalization = personalization_dict
        # Process each node in the filtered personalization dict
        for node_id, ppr_weight in filtered_personalization.items():
            try:
                self.gds_driver.graph.drop('rwr_sample')
                start_time = time.time()
                G_sample, _ = self.gds_driver.graph.sample.rwr("rwr_sample", graph, concurrency=4, samplingRatio = sampling_ratio, startNodes = [node_id],
                                                               restartProbability = 0.4, logProgress = False)
                logging.info(f"largekgRAG : Sampled graph for node {node_id} in {time.time() - start_time:.2f} seconds")
                start_time = time.time()
                result = self.gds_driver.pageRank.stream(
                    G_sample, maxIterations=30, sourceNodes=[node_id], logProgress=False
                ).sort_values("score", ascending=False)
                logging.info(f"largekgRAG : PageRank calculated for node {node_id} in {time.time() - start_time:.2f} seconds")
                start_time = time.time()
                # if self.keyword == 'cc_en':
                if self.keyword != 'cc_en':
                    result = result[result['score'] > 0.0].nlargest(50, 'score').to_dict('records')
                for entry in result.to_dict('records'):
                    if self.keyword == 'cc_en':
                        node_name = self.gds_driver.util.asNode(entry['nodeId'])['name']
                        if not self.has_intersection(word_set, node_name):
                            continue
                    
                    numeric_id = self.gds_driver.util.asNode(entry['nodeId'])['numeric_id']
                    aggregation_node_dict.append({
                        'nodeId': numeric_id,
                        'score': entry['score'] * ppr_weight
                    })

            except Exception as e:
                logging.error(f"Node is filtered out: {self.gds_driver.util.asNode(node_id)['name']}")

    
        
        aggregation_node_dict = sorted(aggregation_node_dict, key=lambda x: x['score'], reverse=True)[:25]
        logging.info(f"Time taken to sample and calculate PageRank: {time.time() - start_time:.2f} seconds")
        start_time = time.time()
        with self.neo4j_driver.session() as session:
            intermediate_time = time.time()
            # Step 1: Distribute entity scores to connected text nodes and find the top 5
            query_scores = """
            UNWIND $entries AS entry
            MATCH (n:Node {numeric_id: entry.nodeId})-[:Source]->(t:Text)
            WITH t.numeric_id AS textId, SUM(entry.score) AS total_score
            ORDER BY total_score DESC
            LIMIT $topN
            RETURN textId, total_score
            """
            # Execute query to aggregate scores
            result_scores = session.run(query_scores, entries=aggregation_node_dict, topN=topN)
            top_numeric_ids = []
            top_scores = []

            # Extract the top text node IDs and scores
            for record in result_scores:
                top_numeric_ids.append(record["textId"])
                top_scores.append(record["total_score"])

            # Step 2: Use top numeric IDs to retrieve the original text
            logging.info(f"Time taken to prepare query 1 : {time.time() - intermediate_time:.2f} seconds")
            intermediate_time = time.time()
            logging.info(len(top_numeric_ids))
            query_text = """
            UNWIND $textIds AS textId
            MATCH (t:Text {numeric_id: textId})
            RETURN t.original_text AS text, t.numeric_id AS textId
            """
            result_texts = session.run(query_text, textIds=top_numeric_ids)
            topN_passages = []
            score_dict = dict(zip(top_numeric_ids, top_scores))

            # Combine original text with scores
            for record in result_texts:
                original_text = record["text"]
                text_id = record["textId"]
                score = score_dict.get(text_id, 0)
                topN_passages.append((original_text, score))
            logging.info(f"Time taken to prepare query 2 : {time.time() - intermediate_time:.2f} seconds")
        # Sort passages by score
        topN_passages = sorted(topN_passages, key=lambda x: x[1], reverse=True)
        top_texts = [item[0] for item in topN_passages][:topN]
        top_scores = [item[1] for item in topN_passages][:topN]

        logging.info(f"Neo4j Query Time: {time.time() - start_time:.2f} seconds")
        return top_texts, top_scores
    
    def retrieve_topk_nodes(self, query, top_k_nodes = 2):
        # extract entities from the query
        entities = self.ner(query)
        if len(entities) == 0:
            entities = [query]
        num_entities = len(entities)
        initial_nodes = []
        logging.info(f"largekgRAG : Number per ner {top_k_nodes}")
        for entity in entities:
            entity_embedding = self.sentence_encoder.encode([entity])
            D, I = self.node_faiss_index.search(entity_embedding, top_k_nodes)
            initial_nodes += [str(i)for i in I[0]]
        name_id_map = {}
        for node_id in initial_nodes:
            name = self.convert_numeric_id_to_name(node_id)
            name_id_map[name] = node_id  
            
        topk_nodes = list(set(initial_nodes))
        # convert the numeric id to string and filter again then return numeric id
        keywords_before_filter = [self.convert_numeric_id_to_name(n) for n in initial_nodes]
        filtered_keywords = self.llm_generator.filter_keywords_with_entity(query, keywords_before_filter)
    
        # Second pass: Add filtered keywords
        filtered_top_k_nodes = []
        filter_log_dict = {}
        match_threshold = 0.8
        logging.info(f"largekgRAG : Filtered Before Match Keywords Candidate: {filtered_keywords}")
        for keyword in filtered_keywords:
            # Check for an exact match first
            if keyword in name_id_map:
                filtered_top_k_nodes.append(name_id_map[keyword])
                filter_log_dict[keyword] = name_id_map[keyword]
            else:
                # Look for close matches using difflib's get_close_matches
                close_matches = get_close_matches(keyword, name_id_map.keys(), n=1, cutoff=match_threshold)
                
                if close_matches:
                    # If a close match is found, add the corresponding node
                    filtered_top_k_nodes.append(name_id_map[close_matches[0]])
                
                filter_log_dict[keyword] = name_id_map[close_matches[0]] if close_matches else None

        logging.info(f"largekgRAG : Filtered After Match Keywords Candidate: {filter_log_dict}")
        
        topk_nodes = list(set(filtered_top_k_nodes))
        if len(topk_nodes) > 2 * num_entities:
            topk_nodes = topk_nodes[:2 * num_entities]
        return topk_nodes
    def _process_text(self, text):
        """Normalize text for containment checks (lowercase, alphanumeric+spaces)"""
        text = text.lower()
        text = ''.join([c for c in text if c.isalnum() or c.isspace()])
        return set(text.split())
    
    def retrieve_personalization_dict(self, query, number_of_source_nodes_per_ner=5, verbose = False):
        topk_nodes = self.retrieve_topk_nodes(query, number_of_source_nodes_per_ner)
        if topk_nodes == []:
            logging.info(f"largekgRAG : No nodes found for query: {query}")
            return {}
        logging.info(f"largekgRAG : Topk nodes: {[self.convert_numeric_id_to_name(node_id) for node_id in topk_nodes]}")
        freq_dict_for_nodes = {}
        query = """
            UNWIND $nodes AS node
            MATCH (n1:Node {numeric_id: node})-[r:Source]-(n2:Text)
            RETURN n1.numeric_id as numeric_id, COUNT(DISTINCT n2.text_id) AS fileCount
        """
        with self.neo4j_driver.session() as session:
            result = session.run(query, nodes=topk_nodes)
            for record in result:
                freq_dict_for_nodes[record["numeric_id"]] = record["fileCount"]
        # Create the personalization dictionary
        personalization_dict = {self.gds_driver.find_node_id(["Node"],{"numeric_id": numeric_id}): 1 / file_count for numeric_id, file_count in freq_dict_for_nodes.items()}
        logging.info(f"largekgRAG : Personalization dict's number of node: {len(personalization_dict)}")
        return personalization_dict

       
    def retrieve_passages(self, query, topN=5, number_of_source_nodes_per_ner = 2, sampling_area = 200, verbose = True):
        personalization_dict = self.retrieve_personalization_dict(query, number_of_source_nodes_per_ner, verbose)
        if personalization_dict == {}:
            return [], [0]
        topN_passages, topN_scores = self.pagerank(personalization_dict, topN, sampling_area = sampling_area, verbose = verbose)
        self.verbose = False
        return topN_passages, topN_scores

def load_api_start_up(sentence_transformer:MiniLM, llm_generator: LLMGenerator, keyword: str)->LargeKGRetriever_API:
    user = "neo4j"
    password = "admin2024"
    neo4j_driver_dict = {}
    neo4j_gds_driver_dict = {}
    node_faiss_index_dict = {}
    # edge_faiss_index_dict = {}
    passage_faiss_index_dict = {}
    
    uri = f"bolt://127.0.0.1:{keyword_to_port_dict[keyword]}"
    neo4j_driver = GraphDatabase.driver(uri, auth=(user, password))
    gds_driver = GraphDataScience(neo4j_driver)
    neo4j_driver_dict[keyword] = neo4j_driver
    neo4j_gds_driver_dict[keyword] = gds_driver
    # build label index and projection graph
    build_neo4j_label_index(neo4j_driver)
    build_projection_graph(gds_driver, keyword)
    node_index, edge_index, passage_index = load_indexes(keyword_to_index_paths[keyword])
    node_faiss_index_dict[keyword] = node_index
    # edge_faiss_index_dict[keyword] = edge_index
    passage_faiss_index_dict[keyword] = passage_index
        

    largekg_api = LargeKGRetriever_API(neo4j_driver_dict, neo4j_gds_driver_dict,
                                        llm_generator, sentence_transformer, 
                                        # node_faiss_index_dict, edge_faiss_index_dict, passage_faiss_index_dict)
                                        node_faiss_index_dict, passage_faiss_index_dict, keyword)
    return largekg_api

    