from sentence_transformers import SentenceTransformer
from transformers import AutoModel
import torch.nn.functional as F
from abc import ABC, abstractmethod

class BaseEmbeddingModel(ABC):
    def __init__(self, model):
        self.model = model

    @abstractmethod
    def encode(self, query, **kwargs):
        """Abstract method to encode queries."""
        pass
    
class NvEmbed(BaseEmbeddingModel):
    def __init__(self, sentence_encoder:AutoModel):
        self.sentence_encoder = sentence_encoder
    def encode(self, query, query_type = 'passage', **kwargs):
        '''
        HippoRAG orignal prompts:
        'ner_to_node': 'Given a phrase, retrieve synonymous or relevant phrases that best match this phrase.',
        'query_to_node': 'Given a question, retrieve relevant phrases that are mentioned in this question.',
        'query_to_fact': 'Given a question, retrieve relevant triplet facts that matches this question.',
        'query_to_sentence': 'Given a question, retrieve relevant sentences that best answer the question.',
        'query_to_passage': 'Given a question, retrieve relevant documents that best answer the question.',
        '''
        normalize_embeddings = kwargs.get('normalize_embeddings', True)
        if query_type == 'passage':
            prompt_prefix = 'Given a question, retrieve relevant documents that best answer the question.'
        elif query_type == 'entity':
            prompt_prefix = 'Given a question, retrieve relevant phrases that are mentioned in this question.'
        elif query_type == 'edge':
            prompt_prefix = 'Given a question, retrieve relevant triplet facts that matches this question.'
        elif query_type == 'fill_in_edge':
            prompt_prefix = 'Given a triples with only head and relation, retrieve relevant triplet facts that best fill the atomic query.'
        elif query_type == 'search':
            query_embeddings = self.sentence_encoder.encode(query)
            if normalize_embeddings:
                query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
            return query_embeddings.detach().cpu().numpy()
        else:
            raise ValueError(f"Unknown query type: {query_type}. Supported types are: passage, entity, edge, search.")
        query_prefix = f"Instruct: {prompt_prefix}\nQuery: "
        query_embeddings = self.sentence_encoder.encode(query, instruction=query_prefix)
        
        # Normalize the embeddings
        if normalize_embeddings:
            query_embeddings = F.normalize(query_embeddings, p=2, dim=1).detach().cpu().numpy()
        return query_embeddings

class SentenceEmbedding(BaseEmbeddingModel):
    def __init__(self,sentence_encoder:SentenceTransformer):
        self.sentence_encoder = sentence_encoder

    def encode(self, query, **kwargs):
        normalize_embeddings = kwargs.get('normalize_embeddings', True)
        return self.sentence_encoder.encode(query, normalize_embeddings=normalize_embeddings)
   