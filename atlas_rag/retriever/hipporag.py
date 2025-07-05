from tqdm import tqdm
import networkx as nx
import numpy as np
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from logging import Logger
from typing import Optional
from atlas_rag.retriever.base import BasePassageRetriever
from atlas_rag.retriever.inference_config import InferenceConfig


class HippoRAGRetriever(BasePassageRetriever):
    def __init__(self, llm_generator:LLMGenerator, sentence_encoder:BaseEmbeddingModel, 
                 data:dict,  inference_config: Optional[InferenceConfig] = None, logger = None, **kwargs):
        self.passage_dict = data["text_dict"]
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder
        self.node_embeddings = data["node_embeddings"]
        self.node_list = data["node_list"]
        file_id_to_node_id = {}
        self.KG = data["KG"]
        for node_id in tqdm(list(self.KG.nodes)):
            if self.KG.nodes[node_id]['type'] == "passage":
                if self.KG.nodes[node_id]['file_id'] not in file_id_to_node_id:
                    file_id_to_node_id[self.KG.nodes[node_id]['file_id']] = []
                file_id_to_node_id[self.KG.nodes[node_id]['file_id']].append(node_id)
        self.file_id_to_node_id = file_id_to_node_id
        
        self.KG:nx.DiGraph = self.KG.subgraph(self.node_list)
        self.node_name_list = [self.KG.nodes[node]["id"] for node in self.node_list]
        
        
        self.logger :Logger = logger
        if self.logger is None:
            self.logging = False
        else:
            self.logging = True
        
        self.inference_config = inference_config if inference_config is not None else InferenceConfig()  
        
    def retrieve_personalization_dict(self, query, topN=10):

        # extract entities from the query
        entities = self.llm_generator.ner(query)
        entities = entities.split(", ")
        if self.logging:
            self.logger.info(f"HippoRAG NER Entities: {entities}")
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
        
        if self.logging:
            self.logger.info(f"HippoRAG Topk Nodes: {[self.KG.nodes[node]['id'] for node in topk_nodes]}")
        
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

    def retrieve(self, query, topN=5, **kwargs):
        topN_nodes = self.inference_config.topk_nodes
        personaliation_dict = self.retrieve_personalization_dict(query, topN=topN_nodes)
        
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
        
        return passag_contents, top_passages