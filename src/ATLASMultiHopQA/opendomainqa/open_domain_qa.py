import networkx as nx
import json
import random
from tqdm import tqdm
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import json
from tqdm import tqdm
import random
import argparse
import torch
import pickle
import networkx as nx
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import re
from openai import OpenAI, AzureOpenAI
from tenacity import retry, wait_fixed, stop_after_delay, stop_after_attempt
from embedding_model import BaseEmbeddingModel, MiniLM, NvEmbed
from concurrent.futures import ThreadPoolExecutor, as_completed
import configparser
from datetime import datetime
from filter_template import messages as filter_messages, validate_filter_output
import logging
from logging.handlers import RotatingFileHandler
from copy import deepcopy
from rag_qa_prompt import prompt_template
import torch.nn.functional as F
from together import Together


# from https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/qa/qa_reader.py
# prompts from hipporag qa_reader
cot_system_instruction = ('As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. If the information is not enough, you can use your own knowledge to answer the question.'
                          'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                          'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')
cot_system_instruction_no_doc = ('As an advanced reading comprehension assistant, your task is to analyze the questions and then answer them. '
                                 'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                 'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')

# This is the instruction for the KG-based QA task
cot_system_instruction_kg = ('As an advanced reading comprehension assistant, your task is to analyze extracted information and corresponding questions meticulously. If the knowledge graph information is not enough, you can use your own knowledge to answer the question. '
                                'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')

config = configparser.ConfigParser()
config.read('config.ini')


deepinfra_models = [
    'meta-llama/Meta-Llama-3.1-8B-Instruct',
    'meta-llama/Llama-3.3-70B-Instruct',
    'mistralai/Mistral-7B-Instruct-v0.3',
    'mistralai/Mixtral-8x22B-Instruct-v0.1',
    'deepseek-ai/DeepSeek-R1',
    'meta-llama/Llama-3.3-70B-Instruct-Turbo'
]
azure_models = [
    'gpt-4o',
    'gpt-35-turbo'
]

def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val

def normalize_embeddings(embeddings):
    """Normalize the embeddings to unit length (L2 norm)."""
    if isinstance(embeddings, torch.Tensor):
        # Handle PyTorch tensors
        norm_emb = F.normalize(embeddings, p=2, dim=1).detach().cpu().numpy()
    elif isinstance(embeddings, np.ndarray):
        # Handle numpy arrays
        norm_emb = F.normalize(torch.tensor(embeddings), p=2, dim=1).detach().cpu().numpy()
    else:
        raise TypeError(f"Unsupported input type: {type(embeddings)}. Must be torch.Tensor or np.ndarray")
    
    return norm_emb

class QAJudger():
    def __init__(self):
        pass

    def split_answer(self, generated_text):
        if "Answer:" in generated_text:
            generated_text = generated_text.split("Answer:")[-1]
        elif "answer:" in generated_text:
            generated_text = generated_text.split("answer:")[-1]
        # if answer is none
        if not generated_text:
            return "none"
        return generated_text

        
    def judge(self, generated_text, reference_text):

        # evaluate in two different methods, EM and F1
        # The EM score is the percentage of questions that were answered exactly correctly
        # The F1 score is the average of the F1 score of each question. The F1 score is the harmonic mean of precision and recall
        # The precision is the number of correct words in the answer divided by the number of words in the answer
        # The recall is the number of correct words in the answer divided by the number of words in the reference answer
        # The F1 score is 2 * (precision * recall) / (precision + recall)

        # generated_text = self.split_answer(generated_text)
        
        generated_text = generated_text.lower()
        if "." in generated_text:
            generated_text = generated_text.replace(".", "")
        
        reference_text = reference_text.lower()
        if "." in reference_text:
            reference_text = reference_text.replace(".", "")
        
        # remove any extra spaces
        generated_text = " ".join(generated_text.split())
        reference_text = " ".join(reference_text.split())

        # remove punctuations
        generated_text = re.sub(r'[^\w\s]', '', generated_text)
        reference_text = re.sub(r'[^\w\s]', '', reference_text)


        # to lower case
        generated_text = generated_text.lower()
        reference_text = reference_text.lower()

        generated_text = generated_text.split()
        reference_text = reference_text.split()

        correct = 0
        for word in generated_text:
            if word in reference_text:
                correct += 1
        
        precision = correct / len(generated_text)
        recall = correct / len(reference_text)

        if precision + recall == 0:
            f1 = 0
        else:
            f1 = 2 * (precision * recall) / (precision + recall)

        em = 1 if generated_text == reference_text else 0

        return em, f1

class LLMGenerator():
    def __init__(self, model_name):
        self.model_name = model_name
        if model_name in deepinfra_models:
            # self.client = OpenAI(api_key=config['settings']['TOGETHER_API_KEY'],base_url="https://api.together.xyz/v1") 
            self.client = OpenAI(api_key=config['settings']['DEEPINFRA_API_KEY'],base_url="https://api.deepinfra.com/v1/openai") 
        elif model_name in azure_models:
            self.client = AzureOpenAI(api_key=config['settings']['AZURE_API_KEY'],base_url="https://api.azure.com/v1/openai")
        self.cot_system_instruction = "".join(cot_system_instruction)
        self.cot_system_instruction_no_doc = "".join(cot_system_instruction_no_doc)
        self.cot_system_instruction_kg = "".join(cot_system_instruction_kg)

    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(5))
    def _generate_response(self, messages, max_new_tokens=32768, temperature=0.7):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(5))
    def filter_generation(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            top_p=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"},
            # Additional parameters for stability
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content
    def _generate_batch_response(self, batch_messages, max_new_tokens=32768, temperature=0.7):
        # Use ThreadPoolExecutor for concurrent requests if using API
        with ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(self._generate_response, msg, max_new_tokens, temperature): idx 
                for idx, msg in enumerate(batch_messages)
            }
            results = [None] * len(batch_messages)  # Pre-allocate results list
            for future in as_completed(future_to_index):
                index = future_to_index[future]  # Get the original index
                results[index] = future.result()  # Place the result in the correct position
        return results

    def generate(self, question, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": self.cot_system_instruction_no_doc},
            {"role": "user", "content": question},
        ]
        return self._generate_response(messages, max_new_tokens=max_new_tokens)

    def generate_with_context(self, question, context, max_new_tokens=1024, frequency_penalty=None, temperature = 0.7, seed = None):
        messages = [
            {"role": "system", "content": self.cot_system_instruction},
            {"role": "user", "content": f"{context}\n\n{question}\nThought:"},
        ]
        # return self._generate_response(messages, max_new_tokens=max_new_tokens, frequency_penalty=frequency_penalty, temperature = temperature, seed = seed)
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)

    def generate_with_context_one_shot(self, question, context, max_new_tokens=4096):
        messages = deepcopy(prompt_template)
        messages.append(
            {"role": "user", "content": f"{context}\n\nQuestions:{question}\nThought:"},
            
        )
        logger.info(f"Generating with context: {messages}")
        return self._generate_response(messages, max_new_tokens=max_new_tokens)
    def generate_with_context_kg(self, question, context, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": self.cot_system_instruction_kg},
            {"role": "user", "content": f"{context}\n\n{question}"},
        ]
        return self._generate_response(messages, max_new_tokens=max_new_tokens)

    def filter_triples_with_entity(self,question, nodes, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": """
            Your task is to filter text cnadidates based on their relevance to a given query.
            The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information.
            You must only select relevant texts from the provided candidate list that have connection to the query, aiding in reasoning and providing an accurate answer.
            The output should be in JSON format, e.g., {"nodes": [e1, e2, e3, e4]}, and if no texts are relevant, return an empty list, {"nodes": []}.
            Do not include any explanations, additional text, or context Only provide the JSON output.
            Do not change the content of each object in the list. You must only use text from the candidate list and cannot generate new text."""},

            {"role": "user", "content": f"""{question} \n Output Before Filter: {nodes} \n Output After Filter:"""}
        ]
        try:
            response = json.loads(self._generate_response(messages, max_new_tokens=max_new_tokens))
            # loop through the reponse json and check if all node is original nodes else go to exception
            return response
        except Exception as e:
            # Log the error if necessary
            logger.error(f"HippoRAG 2: error parsing {e}")
            # If all retries fail, return the original triples
            return json.loads(nodes)

    def filter_triples_with_entity_event(self,question, triples):
        messages = deepcopy(filter_messages)
        messages.append(
            {"role": "user", "content": f"""[ ## question ## ]]
{question}

[[ ## fact_before_filter ## ]]
{triples}"""})
        
        try:
            logger.info(f"HippoRAG2: {messages}")
            response = self.filter_generation(messages)
            cleaned_data = validate_filter_output(response)
            logger.info(f"HippoRAG2: Filtered triples: {cleaned_data}")
            return cleaned_data['fact']
        except Exception as e:
            # Log the error if necessary
            logger.error(f"HippoRAG 2: error parsing {e}")
            # If all retries fail, return the original triples
            return []
    def generate_with_custom_messages(self, custom_messages, max_new_tokens=1024):
        return self._generate_response(custom_messages, max_new_tokens)
    
    def ner(self, text):
        messages = [
            {"role": "system", "content": "Please extract the entities from the following question and output them separated by comma, in the following format: entity1, entity2, ..."},
            {"role": "user", "content": f"Extract the named entities from: {text}"},
        ]
        return self._generate_response(messages)
 
# We will not compute embeddings here, we will use the pre-computed embeddings
def load_all_data(keyword, include_events, include_concept, encoder_model_name):
    # Define paths for loading data
    node_index_path = f"./processed_data/precompute/{keyword}_event{include_events}_concept{include_concept}_{encoder_model_name}_node_faiss.index"
    node_list_path = f"./processed_data/precompute/{keyword}_event{include_events}_concept{include_concept}_node_list.pkl"
    edge_index_path = f"./processed_data/precompute/{keyword}_event{include_events}_concept{include_concept}_{encoder_model_name}_edge_faiss.index"
    edge_list_path = f"./processed_data/precompute/{keyword}_event{include_events}_concept{include_concept}_edge_list.pkl"
    node_embeddings_path = f"./processed_data/precompute/{keyword}_event{include_events}_concept{include_concept}_{encoder_model_name}_node_embeddings.pkl"
    edge_embeddings_path = f"./processed_data/precompute/{keyword}_event{include_events}_concept{include_concept}_{encoder_model_name}_edge_embeddings.pkl"
    text_embeddings_path = f"./processed_data/precompute/{keyword}_{encoder_model_name}_text_embeddings.pkl"
    text_index_path = f"./processed_data/precompute/{keyword}_text_faiss.index"
    
    text_list_path = f"./processed_data/precompute/{keyword}_text_list.pkl"
    original_text_dict_path = f"./processed_data/precompute/{keyword}_original_text_dict_with_node_id.pkl"
        
    
    # Check if all required files exist
    required_paths = [
    text_index_path,
    node_list_path, edge_list_path,
    text_list_path, original_text_dict_path,
    node_index_path, edge_index_path,
    text_embeddings_path, node_embeddings_path, edge_embeddings_path
    ]

    # Check if all required files exist
    missing_paths = [path for path in required_paths if not os.path.exists(path)]

    if missing_paths:
        print("Missing files:")
        for path in missing_paths:
            print(path)
        raise FileNotFoundError("One or more required files are missing.")
    
    # Load data
    text_faiss_index = faiss.read_index(text_index_path)

    with open(node_list_path, "rb") as f:
        node_list = pickle.load(f)

    with open(edge_list_path, "rb") as f:
        edge_list = pickle.load(f)

    with open(text_list_path, "rb") as f:
        original_text_list = pickle.load(f)

    with open(original_text_dict_path, "rb") as f:
        original_text_dict_with_node_id = pickle.load(f)

    node_faiss_index = faiss.read_index(node_index_path)
    edge_faiss_index = faiss.read_index(edge_index_path)

    # Load text embeddings
    with open(text_embeddings_path, "rb") as f:
        text_embeddings = pickle.load(f)
    with open(node_embeddings_path, "rb") as f:
        node_embeddings = pickle.load(f)
    with open(edge_embeddings_path, "rb") as f:
        edge_embeddings = pickle.load(f)
    return {
        "text_faiss_index": text_faiss_index,
        "node_list": node_list,
        "edge_list": edge_list,
        "original_text_list": original_text_list,
        "original_text_dict_with_node_id": original_text_dict_with_node_id,
        "node_faiss_index": node_faiss_index,
        "edge_faiss_index": edge_faiss_index,
        "text_embeddings": text_embeddings,
        "node_embeddings": node_embeddings,
        "edge_embeddings": edge_embeddings,
    }

def calculate_recalls(result_list):
    # Initialize counters
    hippo_partial_recall2_count = 0
    hippo_partial_recall5_count = 0
    hippo_all_recall2_count = 0
    hippo_all_recall5_count = 0
    
    hippo2_all_recall2_count = 0
    hippo2_all_recall5_count = 0
    hippo2_partial_recall2_count = 0
    hippo2_partial_recall5_count = 0

    # tog_partial_recall2_count = 0
    # tog_partial_recall5_count = 0
    # tog_all_recall2_count = 0
    # tog_all_recall5_count = 0

    # edge_partial_recall2_count = 0
    # edge_partial_recall5_count = 0
    # edge_all_recall2_count = 0
    # edge_all_recall5_count = 0
    
    

    total_samples = len(result_list)

    for result in result_list:
        gold_file_ids = set(result.get("gold_file_ids", []))

        # 1) HIPPO
        hippo_id = result.get("hippo_id", [])
        top2 = set(hippo_id[:2])
        top5 = set(hippo_id[:5])
        if top2.intersection(gold_file_ids):
            hippo_partial_recall2_count += 1
        if top5.intersection(gold_file_ids):
            hippo_partial_recall5_count += 1
        if gold_file_ids.issubset(top2):
            hippo_all_recall2_count += 1
        if gold_file_ids.issubset(top5):
            hippo_all_recall5_count += 1

        # 2) TOG
        # tog_id = result.get("tog_id", [])
        # top2 = set(tuple(x) if isinstance(x, list) else x for x in tog_id[:2])
        # top5 = set(tuple(x) if isinstance(x, list) else x for x in tog_id[:5])
        # if top2.intersection(gold_file_ids):
        #     tog_partial_recall2_count += 1
        # if top5.intersection(gold_file_ids):
        #     tog_partial_recall5_count += 1
        # if gold_file_ids.issubset(top2):
        #     tog_all_recall2_count += 1
        # if gold_file_ids.issubset(top5):
        #     tog_all_recall5_count += 1
        
        # 3) EDGES
        # retrieved_edges_id = result.get("retrieved_edges_id", [])
        # top2 = set(retrieved_edges_id[:2])
        # top5 = set(retrieved_edges_id[:5])
        # if top2.intersection(gold_file_ids):
        #     edge_partial_recall2_count += 1
        # if top5.intersection(gold_file_ids):
        #     edge_partial_recall5_count += 1
        # if gold_file_ids.issubset(top2):
        #     edge_all_recall2_count += 1
        # if gold_file_ids.issubset(top5):
        #     edge_all_recall5_count += 1
        
        # 4) HIPPO2
        hippo2_id = result.get("hippo2_id", [])
        top2 = set(hippo2_id[:2])
        top5 = set(hippo2_id[:5])
        if top2.intersection(gold_file_ids):
            hippo2_partial_recall2_count += 1
        if top5.intersection(gold_file_ids):
            hippo2_partial_recall5_count += 1
        if gold_file_ids.issubset(top2):
            hippo2_all_recall2_count += 1
        if gold_file_ids.issubset(top5):
            hippo2_all_recall5_count += 1
            

    # Calculate final recall ratios
    hippo_partial_recall2 = hippo_partial_recall2_count / total_samples
    hippo_partial_recall5 = hippo_partial_recall5_count / total_samples
    hippo_all_recall2 = hippo_all_recall2_count / total_samples
    hippo_all_recall5 = hippo_all_recall5_count / total_samples

    # tog_partial_recall2 = tog_partial_recall2_count / total_samples
    # tog_partial_recall5 = tog_partial_recall5_count / total_samples
    # tog_all_recall2 = tog_all_recall2_count / total_samples
    # tog_all_recall5 = tog_all_recall5_count / total_samples

    # edge_partial_recall2 = edge_partial_recall2_count / total_samples
    # edge_partial_recall5 = edge_partial_recall5_count / total_samples
    # edge_all_recall2 = edge_all_recall2_count / total_samples
    # edge_all_recall5 = edge_all_recall5_count / total_samples
    
    hippo2_partial_recall2 = hippo2_partial_recall2_count / total_samples
    hippo2_partial_recall5 = hippo2_partial_recall5_count / total_samples
    hippo2_all_recall2 = hippo2_all_recall2_count / total_samples
    hippo2_all_recall5 = hippo2_all_recall5_count / total_samples

    return {
        "hippo_partial_recall2": hippo_partial_recall2,
        "hippo_partial_recall5": hippo_partial_recall5,
        "hippo_all_recall2": hippo_all_recall2,
        "hippo_all_recall5": hippo_all_recall5,
        
        # "tog_partial_recall2": tog_partial_recall2,
        # "tog_partial_recall5": tog_partial_recall5,
        # "tog_all_recall2": tog_all_recall2,
        # "tog_all_recall5": tog_all_recall5,
        
        # "edge_partial_recall2": edge_partial_recall2,
        # "edge_partial_recall5": edge_partial_recall5,
        # "edge_all_recall2": edge_all_recall2,
        # "edge_all_recall5": edge_all_recall5,
        
        "hippo2_partial_recall2": hippo2_partial_recall2,
        "hippo2_partial_recall5": hippo2_partial_recall5,
        "hippo2_all_recall2": hippo2_all_recall2,
        "hippo2_all_recall5": hippo2_all_recall5,
    }

class SimpleGraphRetriever():

    def __init__(self, KG:nx.DiGraph, llm_generator:LLMGenerator, sentence_encoder:BaseEmbeddingModel, node_index:faiss.Index, edge_index:faiss.Index, node_list:list, edge_list:list):
        self.KG = KG
        self.node_list = node_list
        self.edge_list = edge_list
        
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder

        self.node_faiss_index = node_index
        self.edge_faiss_index = edge_index


    def retrive_topk_edges(self, query, topk=5):
        # retrieve the top k edges
        topk_edges = []
        query_embedding = self.sentence_encoder.encode([query], query_type='edge')
        D, I = self.edge_faiss_index.search(query_embedding, topk)

        topk_edges += [self.edge_list[i] for i in I[0]]
        edge_file_ids_list = [self.KG.edges[edge]["file_id"] for edge in topk_edges]

        topk_edges_with_data = [(edge[0], self.KG.edges[edge]["relation"], edge[1]) for edge in topk_edges]
        string_edge_edges = [f"{self.KG.nodes[edge[0]]['id']}  {edge[1]}  {self.KG.nodes[edge[2]]['id']}." for edge in topk_edges_with_data]
        
        return string_edge_edges, edge_file_ids_list
    
class SimpleTextRetriever():
    def __init__(self, passage_list:list, sentence_encoder:BaseEmbeddingModel, text_embeddings):  
        self.sentence_encoder = sentence_encoder
        self.passage_list = passage_list
        self.text_embeddings = text_embeddings
        
    def retrieve_topk_passages(self, query, topk=3):
        query_emb = self.sentence_encoder.encode([query], query_type="passage")
        sim_scores = self.text_embeddings @ query_emb[0].T
        topk_indices = np.argsort(sim_scores)[-topk:][::-1]  # Get indices of top-k scores

        # Retrieve top-k passages
        topk_passages = [self.passage_list[i] for i in topk_indices]
        return topk_passages

class TogRetriever():
    def __init__(self, KG:nx.DiGraph | nx.MultiDiGraph, llm_generator, sentence_encoder, node_list, edge_list, node_embeddings, edge_embeddings):
        self.KG = KG
        self.is_multigraph = isinstance(KG, nx.MultiDiGraph)

        # Filter out nodes with type 'concept'
        # self.KG = KG.subgraph([node for node in KG.nodes() if KG.nodes[node]['type'] != 'concept'])
        self.max_paths = 50


        self.node_list = list(KG.nodes)
        if self.is_multigraph:
            # For MultiDiGraph, we need to handle multiple edges between same nodes
            self.edge_list = list(KG.edges(keys=True))  # Include edge keys
            self.edge_list_with_relation = [(edge[0], KG.edges[edge]["relation"], edge[1]) for edge in self.edge_list]
            self.edge_list_string = [f"{edge[0]}  {KG.edges[edge]['relation']}  {edge[1]}" for edge in self.edge_list]
        else:
            # For DiGraph, edges are simple tuples
            self.edge_list = list(KG.edges)
            self.edge_list_with_relation = [(edge[0], KG.edges[edge]["relation"], edge[1]) for edge in self.edge_list]
            self.edge_list_string = [f"{edge[0]}  {KG.edges[edge]['relation']}  {edge[1]}" for edge in self.edge_list]
        
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder        

        self.node_embeddings = node_embeddings
        self.edge_embeddings = edge_embeddings

    

    def ner(self, text):
        messages = [
            {"role": "system", "content": "Please extract the entities from the following question and output them separated by comma, in the following format: entity1, entity2, ..."},
            {"role": "user", "content": f"Extract the named entities from: Are Portland International Airport and Gerald R. Ford International Airport both located in Oregon?"},
            {"role": "system", "content": "Portland International Airport, Gerald R. Ford International Airport, Oregon"},
            {"role": "user", "content": f"Extract the named entities from: {text}"},
        ]

        
        response = self.llm_generator._generate_response(messages)
        generated_text = response
        # print(generated_text)
        return generated_text
    


    def retrieve_topk_nodes(self, query, topN=5):
        # extract entities from the query
        entities = self.ner(query)
        entities = entities.split(", ")

        if len(entities) == 0:
            # If the NER cannot extract any entities, we 
            # use the query as the entity to do approximate search
            entities = [query]

        # evenly distribute the topk for each entity
        topk_for_each_entity = topN//len(entities)
    
        # retrieve the top k nodes
        topk_nodes = []

        for entity_index, entity in enumerate(entities):
            if entity in self.node_list:
                topk_nodes.append(entity)
    
        for entity_index, entity in enumerate(entities): 
            topk_for_this_entity = topk_for_each_entity + 1
            
            entity_embedding = self.sentence_encoder.encode([entity])
            # Calculate similarity scores using dot product
            scores = self.node_embeddings @ entity_embedding[0].T
            # Get top-k indices
            top_indices = np.argsort(scores)[-topk_for_this_entity:][::-1]
            topk_nodes += [self.node_list[i] for i in top_indices]
            
        topk_nodes = list(set(topk_nodes))

        if len(topk_nodes) > 2*topN:
            topk_nodes = topk_nodes[:2*topN]
        return topk_nodes

    def print_path(self, path):
        path_string = ""
        for index, node_or_relation in enumerate(path):
            if index % 2 == 0:
                id_path = self.KG.nodes[node_or_relation]["id"]
            else:
                id_path = node_or_relation
            path_string += f"{id_path} --->"
        path_string = path_string[:-5]
        print(path_string)

    def print_paths(self, paths):
        for path in paths:
            self.print_path(path)
    
    
    
    def shuffle_paths(self, paths):
        # shuffle the paths
        random.shuffle(paths)
        return paths
    

    def retrieve_path(self, query, topN=3, Dmax=3):
        """ 
        Retrieve the top N paths that connect the entities in the query.
        Dmax is the maximum depth of the search.
        """

        # in the first step, we retrieve the top k nodes
        initial_nodes = self.retrieve_topk_nodes(query, topN=topN)
        E = initial_nodes
        P = [ [e] for e in E]
        D = 0

        while D <= Dmax:
            P = self.search(query, P)
            # print(f"P size: {len(P)}")

            # self.print_paths(P)

            P = self.shuffle_paths(P)
            if len(P) > self.max_paths:
                P = P[:self.max_paths]

            # print(f"P size: {len(P)}")

            P = self.prune(query, P, topN)
            
            if self.reasoning(query, P):
                generated_text = self.generate(query, P)
                break
            
            D += 1
        
        if D > Dmax:    
            generated_text = self.generate(query, P)
        
        # print(generated_text)
        return generated_text
    


    def search(self, query, P):
        new_paths = []
        for path in P:
            tail_entity = path[-1]
            
            if self.is_multigraph:
                # For MultiDiGraph, get all edges including their keys
                successors_edges = list(self.KG.edges(tail_entity, keys=True))
                predecessors_edges = list(self.KG.in_edges(tail_entity, keys=True))
                
                # Extract unique neighbors
                successors = set(edge[1] for edge in successors_edges)
                predecessors = set(edge[0] for edge in predecessors_edges)
            else:
                # For DiGraph, simple neighbor lookup
                successors = list(self.KG.successors(tail_entity))
                predecessors = list(self.KG.predecessors(tail_entity))

            # Remove entities already in the path
            successors = [neighbor for neighbor in successors if neighbor not in path]
            predecessors = [neighbor for neighbor in predecessors if neighbor not in path]

            # remove the nodes that have type 'concept'
            successors = [neighbor for neighbor in successors if self.KG.nodes[neighbor]['type'] != 'concept']
            predecessors = [neighbor for neighbor in predecessors if self.KG.nodes[neighbor]['type'] != 'concept']

            # unique the successors and predecessors
            successors = list(set(successors))
            predecessors = list(set(predecessors))

            if len(successors) == 0 and len(predecessors) == 0:
                new_paths.append(path)
                continue

            if self.is_multigraph:
                # Handle successors for MultiDiGraph
                for edge in successors_edges:
                    if edge[1] in successors:  # Only process if neighbor not in path
                        relation = self.KG.edges[edge]["relation"]
                        new_path = path + [relation, edge[1]]
                        new_paths.append(new_path)
                
                # Handle predecessors for MultiDiGraph
                for edge in predecessors_edges:
                    if edge[0] in predecessors:  # Only process if neighbor not in path
                        relation = self.KG.edges[edge]["relation"]
                        new_path = path + [relation, edge[0]]
                        new_paths.append(new_path)
            else:
                # Handle successors for DiGraph
                for neighbor in successors:
                    relation = self.KG.edges[(tail_entity, neighbor)]["relation"]
                    new_path = path + [relation, neighbor]
                    new_paths.append(new_path)
                
                # Handle predecessors for DiGraph
                for neighbor in predecessors:
                    relation = self.KG.edges[(neighbor, tail_entity)]["relation"]
                    new_path = path + [relation, neighbor]
                    new_paths.append(new_path)
        
        # Remove duplicate paths by converting to tuples and back to lists
        new_paths = [list(path) for path in set(tuple(path) for path in new_paths)]

       
        
        return new_paths
    
    def prune(self, query, P, topN=3):
        ratings = []

        for path in P:
            path_string = ""
            for index, node_or_relation in enumerate(path):
                if index % 2 == 0:
                    id_path = self.KG.nodes[node_or_relation]["id"]
                else:
                    id_path = node_or_relation
                path_string += f"{id_path} --->"
            path_string = path_string[:-5]

            prompt = f"Please rating the following path based on the relevance to the question. The ratings should be in the range of 1 to 5. 1 for least relevant and 5 for most relevant. Only provide the rating, do not provide any other information. The output should be a single integer number. If you think the path is not relevant, please provide 0. If you think the path is relevant, please provide a rating between 1 and 5. \n Query: {query} \n path: {path_string}" 

            messages = [{"role": "system", "content": "Answer the question following the prompt."},
            {"role": "user", "content": f"{prompt}"}]

            response = self.llm_generator._generate_response(messages)
            # print(response)
            rating = int(response)
            ratings.append(rating)
            
        # sort the paths based on the ratings
        sorted_paths = [path for _, path in sorted(zip(ratings, P), reverse=True)]
        
        return sorted_paths[:topN]

    def reasoning(self, query, P):
        triples = []
        for path in P:
            for i in range(0, len(path)-2, 2):

                # triples.append((path[i], path[i+1], path[i+2]))
                triples.append((self.KG.nodes[path[i]]["id"], path[i+1], self.KG.nodes[path[i+2]]["id"]))
        
        triples_string = [f"({triple[0]}, {triple[1]}, {triple[2]})" for triple in triples]
        triples_string = ". ".join(triples_string)

        prompt = f"Given a question and the associated retrieved knowledge graph triples (entity, relation, entity), you are asked to answer whether it's sufficient for you to answer the question with these triples and your knowledge (Yes or No). Query: {query} \n Knowledge triples: {triples_string}"
        
        messages = [{"role": "system", "content": "Answer the question following the prompt."},
        {"role": "user", "content": f"{prompt}"}]

        response = self.llm_generator._generate_response(messages)
        return "yes" in response.lower()

    def generate(self, query, P):
        triples = []
        for path in P:
            for i in range(0, len(path)-2, 2):
                # triples.append((path[i], path[i+1], path[i+2]))
                triples.append((self.KG.nodes[path[i]]["id"], path[i+1], self.KG.nodes[path[i+2]]["id"]))
        
        triples_string = [f"({triple[0]}, {triple[1]}, {triple[2]})" for triple in triples]
        triples_string = ". ".join(triples_string)
        
        messages = [
            {"role": "system", "content": cot_system_instruction_kg},
            {"role": "user", "content": f"{triples_string}\n\n{query}"},
        ]
        
        response = self.llm_generator._generate_response(messages)
        return triples_string, response

class HippoRAGRetriever():
    def __init__(self, KG:nx.DiGraph, passage_dict:dict, llm_generator:LLMGenerator, sentence_encoder:BaseEmbeddingModel, node_list:list, node_embeddings:faiss.Index, file_id_to_node_id:dict):
        self.passage_dict = passage_dict
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.node_embeddings = node_embeddings
        self.node_list = node_list
        self.file_id_to_node_id = file_id_to_node_id
        self.KG = KG.subgraph(self.node_list)
        self.node_name_list = [self.KG.nodes[node]["id"] for node in self.node_list]

        
    def retrieve_personalization_dict(self, query, topN=10):

        # extract entities from the query
        entities = self.llm_generator.ner(query)
        entities = entities.split(", ")
        logger.info(f"HippoRAG NER Entities: {entities}")
        # print("Entities:", entities)

        if len(entities) == 0:
            # If the NER cannot extract any entities, we 
            # use the query as the entity to do approximate search
            entities = [query]
    
        # evenly distribute the topk for each entity
        topk_for_each_entity = topN//len(entities)
    
        # retrieve the top k nodes
        topk_nodes = []

        for entity_index, entity in enumerate(entities):
            if entity in self.node_name_list:
                # get the index of the entity in the node list
                index = self.node_name_list.index(entity)
                topk_nodes.append(self.node_list[index])
            else:
                topk_for_this_entity = 1
                
                # print("Topk for this entity:", topk_for_this_entity)
                
                entity_embedding = self.sentence_encoder.encode([entity], query_type="search")
                scores = self.node_embeddings@entity_embedding[0].T
                index_matrix = np.argsort(scores)[-topk_for_this_entity:][::-1]
               
                topk_nodes += [self.node_list[i] for i in index_matrix]
        
        logger.info(f"HippoRAG Topk Nodes: {[self.KG.nodes[node]['id'] for node in topk_nodes]}")
        
        topk_nodes = list(set(topk_nodes))

        # assert len(topk_nodes) <= topN
        if len(topk_nodes) > 2*topN:
            topk_nodes = topk_nodes[:2*topN]

        
        # print("Topk nodes:", topk_nodes)
        # find the number of docs that one work appears in
        freq_dict_for_nodes = {}
        for node in topk_nodes:
            node_data = self.KG.nodes[node]
            # print(node_data)
            file_ids = node_data["file_id"]
            file_ids_list = file_ids.split(",")
            #uniq this list
            file_ids_list = list(set(file_ids_list))
            freq_dict_for_nodes[node] = len(file_ids_list)

        personalization_dict = {node: 1 / freq_dict_for_nodes[node]  for node in topk_nodes}

        # print("personalization dict: ")
        return personalization_dict

    def retrieve_passages(self, query, topN=5):
        personaliation_dict = self.retrieve_personalization_dict(query, topN=2*topN)
        
        # retrieve the top N passages
        pr = nx.pagerank(self.KG, personalization=personaliation_dict)

        for node in pr:
            pr[node] = round(pr[node], 4)
            if pr[node] < 0.001:
                pr[node] = 0
        
        passage_probabilities_sum = {}
        for node in pr:
            node_data = self.KG.nodes[node]
            file_ids = node_data["file_id"]
            # for each file id check through each text_id
            file_ids_list = file_ids.split(",")
            #uniq this list
            file_ids_list = list(set(file_ids_list))
            # file id to node id
            
            for file_id in file_ids_list:
                if file_id == 'concept_file':
                    continue
                for node_id in self.file_id_to_node_id[file_id]:
                    if node_id not in passage_probabilities_sum:
                        passage_probabilities_sum[node_id] = 0
                    passage_probabilities_sum[node_id] += pr[node]
        
        sorted_passages = sorted(passage_probabilities_sum.items(), key=lambda x: x[1], reverse=True)
        top_passages = sorted_passages[:topN]
        top_passages, scores = zip(*top_passages)

        passag_contents = [self.passage_dict[passage_id] for passage_id in top_passages]
        
        return passag_contents, scores, top_passages

class HippoRAG2Retriever():
    def __init__(self, KG:nx.DiGraph, passage_dict:dict, llm_generator:LLMGenerator, sentence_encoder:BaseEmbeddingModel, 
                 node_list:list, node_embeddings:np.ndarray,
                 edge_list:list, edge_embeddings:np.ndarray, edge_faiss_index:faiss.Index,
                 text_embeddings:np.ndarray, text_id_list: list, node_id_to_file_id:dict,
                 hipporag2mode:str):
        self.passage_dict = passage_dict
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.node_embeddings = node_embeddings
        self.node_list = node_list
        self.edge_list = edge_list
        self.edge_embeddings = edge_embeddings
        self.text_embeddings = text_embeddings
        self.edge_faiss_index = edge_faiss_index
        self.text_id_list = text_id_list
        self.node_id_to_file_id = node_id_to_file_id

        print("HippoRAG2: KG passages nodes:", len(self.passage_dict))
        print("HippoRAG2: KG passages nodes:", len(self.text_id_list))
        
        self.KG = KG.subgraph(self.node_list + text_id_list)

        if hipporag2mode == "query2edge":
            self.retrieve_node_fn = self.query2edge
        elif hipporag2mode == "query2node":
            self.retrieve_node_fn = self.query2node
        elif hipporag2mode == "ner2node":
            self.retrieve_node_fn = self.ner2node
        else:
            raise ValueError(f"Invalid mode: {hipporag2mode}. Choose from 'query2edge', 'query2node', or 'query2passage'.")
    
    def ner(self, text):
        return self.llm_generator.ner(text)
    
    def ner2node(self, query, topN = 10):
        entities = self.ner(query)
        entities = entities.split(", ")

        if len(entities) == 0:
            entities = [query]
        # retrieve the top k nodes
        topk_nodes = []
        node_score_dict = {}
        for entity_index, entity in enumerate(entities):
            topk_for_this_entity = 1
            entity_embedding = self.sentence_encoder.encode([entity], query_type="search")
            scores = min_max_normalize(self.node_embeddings@entity_embedding[0].T)
            index_matrix = np.argsort(scores)[-topk_for_this_entity:][::-1]
            similarity_matrix = [scores[i] for i in index_matrix]
            for index, sim_score in zip(index_matrix, similarity_matrix):
                node = self.node_list[index]
                if node not in topk_nodes:
                    topk_nodes.append(node)
                    node_score_dict[node] = sim_score
                    
        topk_nodes = list(set(topk_nodes))
        result_node_score_dict = {}
        if len(topk_nodes) > 2*topN:
            topk_nodes = topk_nodes[:2*topN]
            for node in topk_nodes:
                if node in node_score_dict:
                    result_node_score_dict[node] = node_score_dict[node]
        return result_node_score_dict
    
    def query2node(self, query, topN = 10):
        query_emb = self.sentence_encoder.encode([query], query_type="entity")
        scores = min_max_normalize(self.node_embeddings@query_emb[0].T)
        index_matrix = np.argsort(scores)[-topN:][::-1]
        similarity_matrix = [scores[i] for i in index_matrix]
        result_node_score_dict = {}
        for index, sim_score in zip(index_matrix, similarity_matrix):
            node = self.node_list[index]
            result_node_score_dict[node] = sim_score

        return result_node_score_dict
    
    def query2edge(self, query, topN = 10):
        query_emb = self.sentence_encoder.encode([query], query_type="edge")
        scores = min_max_normalize(self.edge_embeddings@query_emb[0].T)
        index_matrix = np.argsort(scores)[-topN:][::-1]
        log_edge_list = []
        for index in index_matrix:
            edge = self.edge_list[index]
            edge_str = [self.KG.nodes[edge[0]]['id'], self.KG.edges[edge]['relation'], self.KG.nodes[edge[1]]['id']]
            log_edge_list.append(edge_str)
        logger.info(f"HippoRAG2: Before Fitler Edges: {log_edge_list}")
        similarity_matrix = [scores[i] for i in index_matrix]
        # construct the edge list
        before_filter_edge_json = {}
        before_filter_edge_json['fact'] = []
        for index, sim_score in zip(index_matrix, similarity_matrix):
            edge = self.edge_list[index]
            edge_str = [self.KG.nodes[edge[0]]['id'], self.KG.edges[edge]['relation'], self.KG.nodes[edge[1]]['id']]
            before_filter_edge_json['fact'].append(edge_str)
        filtered_facts = self.llm_generator.filter_triples_with_entity_event(query, json.dumps(before_filter_edge_json, ensure_ascii=False))
        if len(filtered_facts) == 0:
            return {}
        # use filtered facts to get the edge id and check if it exists in the original candidate list.
        node_score_dict = {}
        log_edge_list = []
        for edge in filtered_facts:
            edge_str = f'{edge[0]} {edge[1]} {edge[2]}'
            search_emb = self.sentence_encoder.encode([edge_str], query_type="search")
            D, I = self.edge_faiss_index.search(search_emb, 1)
            filtered_index = I[0][0]
            # get the edge and the original score
            edge = self.edge_list[filtered_index]
            log_edge_list.append([self.KG.nodes[edge[0]]['id'], self.KG.edges[edge]['relation'], self.KG.nodes[edge[1]]['id']])
            head, tail = edge[0], edge[1]
            sim_score = scores[filtered_index]
            logger.info(f"HippoRAG2: Edge: {edge_str}, Score: {sim_score}")
            if head not in node_score_dict:
                node_score_dict[head] = [sim_score]
            else:
                node_score_dict[head].append(sim_score)
            if tail not in node_score_dict:
                node_score_dict[tail] = [sim_score]
            else:
                node_score_dict[tail].append(sim_score)
        # average the scores
        logger.info(f"HippoRAG2: Filtered edges: {log_edge_list}")
        
        # take average of the scores
        for node in node_score_dict:
            node_score_dict[node] = sum(node_score_dict[node]) / len(node_score_dict[node])
        
        return node_score_dict
    
    def query2passage(self, query, weight_adjust = 0.05):
        query_emb = self.sentence_encoder.encode([query], query_type="passage")
        sim_scores = self.text_embeddings @ query_emb[0].T
        sim_scores = min_max_normalize(sim_scores)*weight_adjust # converted to probability
        # create dict of passage id and score
        return dict(zip(self.text_id_list, sim_scores))
    
    def retrieve_personalization_dict(self, query, topN=10):
        node_dict = self.retrieve_node_fn(query, topN=30)
        text_dict = self.query2passage(query, weight_adjust=0.05)
  
        return node_dict, text_dict

    def retrieve_passages(self, query, topN=5):
        node_dict, text_dict = self.retrieve_personalization_dict(query, topN=topN)
          
        personalization_dict = {}
        if len(node_dict) == 0:
            # return topN text passages
            sorted_passages = sorted(text_dict.items(), key=lambda x: x[1], reverse=True)
            sorted_passages = sorted_passages[:topN]
            sorted_passages_contents = []
            sorted_scores = []
            sorted_passage_ids = []
            for passage_id, score in sorted_passages:
                sorted_passages_contents.append(self.passage_dict[passage_id])
                sorted_scores.append(float(score))
                sorted_passage_ids.append(self.node_id_to_file_id[passage_id])
            return sorted_passages_contents, sorted_scores, sorted_passage_ids
            
        personalization_dict.update(node_dict)
        personalization_dict.update(text_dict)
        # retrieve the top N passages
        pr = nx.pagerank(self.KG, personalization=personalization_dict, alpha = 0.9, max_iter=2000, tol=1e-7)

        # get the top N passages based on the text_id list and pagerank score
        text_dict_score = {}
        for node in self.text_id_list:
            # filter out nodes that have 0 score
            if pr[node] > 0.0:
                text_dict_score[node] = pr[node]
            
        # return topN passages
        sorted_passages_ids = sorted(text_dict_score.items(), key=lambda x: x[1], reverse=True)
        sorted_passages_ids = sorted_passages_ids[:topN]
        
        sorted_passages_contents = []
        sorted_scores = []
        sorted_passage_ids = []
        for passage_id, score in sorted_passages_ids:
            sorted_passages_contents.append(self.passage_dict[passage_id])
            sorted_scores.append(score)
            sorted_passage_ids.append(self.node_id_to_file_id[passage_id])
        return sorted_passages_contents, sorted_scores, sorted_passage_ids


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # "2wikimultihopqa", "hotpotqa", "musique"
    parser.add_argument("--keyword", type=str, default="hotpotqa")
    parser.add_argument("--model", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    parser.add_argument("--include_events", action="store_true", default=False)
    parser.add_argument("--include_concept", action="store_true", default=False)
    parser.add_argument("--encoder_model_name", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--hipporag2mode", type=str, default='query2edge', choices=['query2edge'], help="Choose the mode for HippoRAG2 retriever")
    parser.add_argument("--method", type=str, default='hipporag2', choices=['hipporag', 'hipporag2', 'tog'], help="Choose which method to use for retrieval")
    print(f'Loading experiment config {parser.parse_args()}')
    args = parser.parse_args()
    
    keyword = args.keyword
    model_name = args.model
    include_events = args.include_events
    include_concept = args.include_concept
    encoder_model_name = args.encoder_model_name
    hipporag2mode = args.hipporag2mode
    method = args.method

    log_file_path = f'./log/{keyword}_event{args.include_events}_concept{args.include_concept}_{method}.log'
    logger = logging.getLogger("MyLogger")
    logger.setLevel(logging.INFO)
    max_bytes = 50 * 1024 * 1024  # 50 MB
    if not os.path.exists(log_file_path):
        os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
    handler = RotatingFileHandler(log_file_path, maxBytes=max_bytes, backupCount=5)
    logger.addHandler(handler)
    # load the data
    if encoder_model_name == "nvidia/NV-Embed-v2":
        loading_dir_encoder_name = "NV-Embed-v2"
    else:
        loading_dir_encoder_name = encoder_model_name
        
    data = load_all_data(keyword, include_events, include_concept, loading_dir_encoder_name)
    node_list = data["node_list"]
    edge_list = data["edge_list"]
    original_text_list = data["original_text_list"]
    original_text_dict_with_node_id = data["original_text_dict_with_node_id"]
    edge_faiss_index = data["edge_faiss_index"]
    text_embeddings = data["text_embeddings"]
    node_embeddings = data["node_embeddings"]
    edge_embeddings = data["edge_embeddings"]
    node_embeddings = normalize_embeddings(np.array(node_embeddings).astype(np.float32))
    text_embeddings = normalize_embeddings(np.array(text_embeddings).astype(np.float32))
    edge_embeddings = normalize_embeddings(np.array(edge_embeddings).astype(np.float32))
    
    if len(node_list) != len(node_embeddings):
        print(f"Node list length: {len(node_list)}, Node embeddings length: {len(node_embeddings)}")
    assert len(node_list) == len(node_embeddings)

    # Check edge list, embeddings, and FAISS index
    if len(edge_list) != len(edge_embeddings) or len(edge_embeddings) != edge_faiss_index.ntotal:
        print(f"Edge list length: {len(edge_list)}, Edge embeddings length: {len(edge_embeddings)}, FAISS index total: {edge_faiss_index.ntotal}")
    assert len(edge_list) == len(edge_embeddings) == edge_faiss_index.ntotal

    # Check original text list and text embeddings
    if len(original_text_list) != len(text_embeddings):
        print(f"Original text list length: {len(original_text_list)}, Text embeddings length: {len(text_embeddings)}")
    assert len(original_text_list) == len(text_embeddings)
    
    # load the KG and filter based on stored edges
    graph_dir = f"./processed_data/kg_graphml/{keyword}_concept.graphml"
    KG : nx.DiGraph = nx.read_graphml(graph_dir)
    # loop through KG to get the node if using the file id, while retaining the order of the original text list

    text_id_list = []
    node_id_to_file_id = {} # For checking recall
    file_id_to_node_id = {}
    for node_id in tqdm(list(KG.nodes)):
        if keyword == "musique" and KG.nodes[node_id]['type']=="passage":
            node_id_to_file_id[node_id] = KG.nodes[node_id]["id"]
        else:
            node_id_to_file_id[node_id] = KG.nodes[node_id]["file_id"]
        
        if KG.nodes[node_id]['file_id'] not in file_id_to_node_id:
            file_id_to_node_id[KG.nodes[node_id]['file_id']] = []
        if KG.nodes[node_id]['type'] == "passage":
            file_id_to_node_id[KG.nodes[node_id]['file_id']].append(node_id)
    for node_id in tqdm(list(original_text_dict_with_node_id.keys())):
            text_id_list.append(node_id)
    assert len(text_id_list) == len(original_text_dict_with_node_id)
    print("Number of text ids:", len(text_id_list))
    print('Number of nodes in the KG:', len(KG.nodes))
    print('Number of edges in the KG:', len(KG.edges))

    llm_judge = QAJudger()
    llm_generator = LLMGenerator(model_name)
    # embedding model is used for computing embeddings for nodes and edges
    if encoder_model_name == "nvidia/NV-Embed-v2":
        sentence_encdoder = AutoModel.from_pretrained("nvidia/NV-Embed-v2", device_map="auto", trust_remote_code=True)
        embedding_model = NvEmbed(sentence_encdoder)
    else:
        sentence_encdoder = SentenceTransformer(encoder_model_name, device="cuda:0")
        embedding_model = MiniLM(sentence_encdoder)

    # edge_retriever = SimpleGraphRetriever(KG, llm_generator, embedding_model, node_faiss_index, edge_faiss_index, node_list, edge_list)
    # passage_retriever = SimpleTextRetriever(original_text_list, embedding_model, text_embeddings) 
    # tog_retriever = TogRetriever(KG, original_text_dict_with_node_id, llm_generator, embedding_model, node_list, node_faiss_index, edge_faiss_index)
    # hippo_rag_retriever = HippoRAGRetriever(KG, original_text_dict_with_node_id, llm_generator, embedding_model, node_list, node_embeddings, file_id_to_node_id)
    # hippo_rag_2_retriever = HippoRAG2Retriever(KG, original_text_dict_with_node_id, 
    #                                            llm_generator, embedding_model, 
    #                                            node_list, node_embeddings, 
    #                                            edge_list, edge_embeddings, edge_faiss_index,
    #                                            text_embeddings, text_id_list, node_id_to_file_id,
    #                                            hipporag2mode=hipporag2mode)

    # keyword = "hotpotqa"
    time_stamp = datetime.now().strftime("%Y%m%d%H%M%S")

    question_data_dir = f"./data/{keyword}.json"

    result_list = []

    # Initialize retrievers based on selected method
    if method == 'tog':
        retriever = TogRetriever(KG, llm_generator, embedding_model, node_list, edge_list, node_embeddings, edge_embeddings)
    elif method == 'hipporag':
        retriever = HippoRAGRetriever(KG, original_text_dict_with_node_id, llm_generator, embedding_model, node_list, node_embeddings, file_id_to_node_id)
    elif method == 'hipporag2':
        retriever = HippoRAG2Retriever(KG, original_text_dict_with_node_id, 
                                     llm_generator, embedding_model, 
                                     node_list, node_embeddings, 
                                     edge_list, edge_embeddings, edge_faiss_index,
                                     text_embeddings, text_id_list, node_id_to_file_id,
                                     hipporag2mode=hipporag2mode)

    with open(question_data_dir, "r") as f:
        data = json.load(f)
        print(f"Data loaded from {question_data_dir}")
        data = data[:1]

        for sample in tqdm(data):
            question = sample["question"]
            answer = sample["answer"]

            gold_file_ids = []
            if keyword == "hotpotqa" or keyword == "2wikimultihopqa":
                supporting_facts = sample["supporting_facts"]
                for fact in supporting_facts:
                    file_id = fact[0]
                    gold_file_ids.append(file_id)
            elif keyword == "musique":
                paragraphs = sample["paragraphs"]
                for paragraph in paragraphs:
                    is_supporting_fact = paragraph["is_supporting"]
                    if is_supporting_fact:
                        gold_file_ids.append(paragraph["paragraph_text"])
            else:
                print("Keyword not supported")
                continue
                
            logger.info(f"Question: {question}")
            
            # Conduct retrieval based on selected method
            if method == 'tog':
                tog_triples, tog_generated_answer = retriever.retrieve_path(question, topN=3, Dmax=3)
                tog_short_answer = llm_judge.split_answer(tog_generated_answer)
                tog_em, tog_f1 = llm_judge.judge(tog_short_answer, answer)
                
                result = {
                    "question": question,
                    "answer": answer,
                    "tog_triples": tog_triples,
                    "tog_generated_answer": tog_generated_answer,
                    "tog_short_answer": tog_short_answer,
                    "tog_em": tog_em,
                    "tog_f1": tog_f1,
                    "gold_file_ids": gold_file_ids
                }
                
            elif method == 'hipporag':
                hippo_passages, hippo_scores, hippo_id = retriever.retrieve_passages(question, topN=5)
                hippo_response_text = "\nTitle:".join(hippo_passages)
                generated_answer_with_hippo = llm_generator.generate_with_context(question, hippo_response_text, max_new_tokens=2048, temperature=0.5)
                hippo_short_answer = llm_judge.split_answer(generated_answer_with_hippo)
                hippo_em, hippo_f1 = llm_judge.judge(hippo_short_answer, answer)
                
                result = {
                    "question": question,
                    "answer": answer,
                    "hippo_passages": hippo_passages,
                    "hippo_id": hippo_id,
                    "hippo_scores": hippo_scores,
                    "hippo_generated_answer": generated_answer_with_hippo,
                    "hippo_short_answer": hippo_short_answer,
                    "hippo_em": hippo_em,
                    "hippo_f1": hippo_f1,
                    "gold_file_ids": gold_file_ids
                }
                
            elif method == 'hipporag2':
                hippo2_passages, hippo2_scores, hippo2_id = retriever.retrieve_passages(question, topN=5)
                hippo2_response_text = "Title: " + "\nTitle: ".join(hippo2_passages)
                generated_answer_with_hippo2 = llm_generator.generate_with_context(question, hippo2_response_text, max_new_tokens=2048, temperature=0.5)
                hippo2_short_answer = llm_judge.split_answer(generated_answer_with_hippo2)
                hippo2_em, hippo2_f1 = llm_judge.judge(hippo2_short_answer, answer)
                
                result = {
                    "question": question,
                    "answer": answer,
                    "hippo2_passages": hippo2_passages,
                    "hippo2_id": hippo2_id,
                    "hippo2_scores": hippo2_scores,
                    "hippo2_generated_answer": generated_answer_with_hippo2,
                    "hippo2_short_answer": hippo2_short_answer,
                    "hippo2_em": hippo2_em,
                    "hippo2_f1": hippo2_f1,
                    "gold_file_ids": gold_file_ids
                }
            
            result_list.append(result)

    # Calculate averages based on selected method
    if method == 'tog':
        average_em = sum([result["tog_em"] for result in result_list]) / len(result_list)
        average_f1 = sum([result["tog_f1"] for result in result_list]) / len(result_list)
        summary_dict = {
            "keyword": keyword,
            "average_f1_with_tog": average_f1,
            "average_em_with_tog": average_em,
        }
    elif method == 'hipporag':
        average_em = sum([result["hippo_em"] for result in result_list]) / len(result_list)
        average_f1 = sum([result["hippo_f1"] for result in result_list]) / len(result_list)
        summary_dict = {
            "keyword": keyword,
            "average_f1_with_hippo": average_f1,
            "average_em_with_hippo": average_em,
        }
    elif method == 'hipporag2':
        average_em = sum([result["hippo2_em"] for result in result_list]) / len(result_list)
        average_f1 = sum([result["hippo2_f1"] for result in result_list]) / len(result_list)
        summary_dict = {
            "keyword": keyword,
            "average_f1_with_hippo2": average_f1,
            "average_em_with_hippo2": average_em,
        }

    current_time = datetime.now()
    formatted_time = current_time.strftime("%Y%m%d%H%M%S")

    summary_file = f"./result/{keyword}/summary_{formatted_time}_event{include_events}_concept{include_concept}_{loading_dir_encoder_name}_{method}.json"
    if not os.path.exists(os.path.dirname(summary_file)):
        os.makedirs(os.path.dirname(summary_file), exist_ok=True)
    f_summary = open(summary_file, "w")

    json.dump(summary_dict, f_summary)
    f_summary.write("\n")
    
    result_dir = f"./result/{keyword}/result_{formatted_time}_event{include_events}_concept{include_concept}_{loading_dir_encoder_name}_{method}.json"
    if not os.path.exists(os.path.dirname(result_dir)):
        os.makedirs(os.path.dirname(result_dir), exist_ok=True)
    with open(result_dir, "w") as f:
        for result in result_list:
            json.dump(result, f)
            f.write("\n")

    f_summary.close()









