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
from embedding_model import BaseEmbeddingModel
from atlas_rag.reader.llm_generator import LLMGenerator, cot_system_instruction_kg
 
def min_max_normalize(x):
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    
    # Handle the case where all values are the same (range is zero)
    if range_val == 0:
        return np.ones_like(x)  # Return an array of ones with the same shape as x
    
    return (x - min_val) / range_val
 
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
    def __init__(self, KG:nx.DiGraph, llm_generator, sentence_encoder, node_list, edge_list, node_embeddings, edge_embeddings):
        self.KG = KG

        self.node_list = list(KG.nodes)
        self.edge_list = list(KG.edges)
        self.edge_list_with_relation = [(edge[0], KG.edges[edge]["relation"], edge[1])  for edge in self.edge_list]
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

    def retrieve_path(self, query, topN=5, Dmax=4):
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
            sucessors = list(self.KG.successors(tail_entity))
            predecessors = list(self.KG.predecessors(tail_entity))

            # print(f"tail_entity: {tail_entity}")
            # print(f"sucessors: {sucessors}")
            # print(f"predecessors: {predecessors}")

            # # print the attributes of the tail_entity
            # print(f"attributes of the tail_entity: {self.KG.nodes[tail_entity]}")
           
            # remove the entity that is already in the path
            sucessors = [neighbour for neighbour in sucessors if neighbour not in path]
            predecessors = [neighbour for neighbour in predecessors if neighbour not in path]

            if len(sucessors) == 0 and len(predecessors) == 0:
                new_paths.append(path)
                continue
            for neighbour in sucessors:
                relation = self.KG.edges[(tail_entity, neighbour)]["relation"]
                new_path = path + [relation, neighbour]
                new_paths.append(new_path)
            
            for neighbour in predecessors:
                relation = self.KG.edges[(neighbour, tail_entity)]["relation"]
                new_path = path + [relation, neighbour]
                new_paths.append(new_path)
        
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
                 hipporag2mode:str = "query2edge"):
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
            if head not in node_score_dict:
                node_score_dict[head] = [sim_score]
            else:
                node_score_dict[head].append(sim_score)
            if tail not in node_score_dict:
                node_score_dict[tail] = [sim_score]
            else:
                node_score_dict[tail].append(sim_score)
        
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










