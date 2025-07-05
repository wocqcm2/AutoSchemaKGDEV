from typing import Dict
import numpy as np
from atlas_rag.vectorstore.embedding_model import BaseEmbeddingModel
from atlas_rag.llm_generator.llm_generator import LLMGenerator
from atlas_rag.retriever.base import BaseEdgeRetriever, BasePassageRetriever
class SimpleGraphRetriever(BaseEdgeRetriever):

    def __init__(self, llm_generator:LLMGenerator, sentence_encoder:BaseEmbeddingModel, 
                 data:dict):
        
        self.KG = data["KG"]
        self.node_list = data["node_list"]
        self.edge_list = data["edge_list"]
        
        self.llm_generator = llm_generator
        self.sentence_encoder = sentence_encoder

        self.node_faiss_index = data["node_faiss_index"]
        self.edge_faiss_index = data["edge_faiss_index"]


    def retrieve(self, query, topN=5, **kwargs):
        # retrieve the top k edges
        topk_edges = []
        query_embedding = self.sentence_encoder.encode([query], query_type='edge')
        D, I = self.edge_faiss_index.search(query_embedding, topN)

        topk_edges += [self.edge_list[i] for i in I[0]]

        topk_edges_with_data = [(edge[0], self.KG.edges[edge]["relation"], edge[1]) for edge in topk_edges]
        string_edge_edges = [f"{self.KG.nodes[edge[0]]['id']}  {edge[1]}  {self.KG.nodes[edge[2]]['id']}" for edge in topk_edges_with_data]

        return string_edge_edges, ["N/A" for _ in range(len(string_edge_edges))]
    
class SimpleTextRetriever(BasePassageRetriever):
    def __init__(self, passage_dict:Dict[str,str], sentence_encoder:BaseEmbeddingModel, data:dict):  
        self.sentence_encoder = sentence_encoder
        self.passage_dict = passage_dict
        self.passage_list = list(passage_dict.values())
        self.passage_keys = list(passage_dict.keys())
        self.text_embeddings = data["text_embeddings"]
        
    def retrieve(self, query, topN=5, **kwargs):
        query_emb = self.sentence_encoder.encode([query], query_type="passage")
        sim_scores = self.text_embeddings @ query_emb[0].T
        topk_indices = np.argsort(sim_scores)[-topN:][::-1]  # Get indices of top-k scores

        # Retrieve top-k passages
        topk_passages = [self.passage_list[i] for i in topk_indices]
        topk_passages_ids = [self.passage_keys[i] for i in topk_indices]
        return topk_passages, topk_passages_ids
