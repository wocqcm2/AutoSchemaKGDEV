import os
import json
import numpy as np
from logging import Logger
from atlas_rag.retriever.rag_model import BaseRetriever, BaseEdgeRetriever, BasePassageRetriever
from typing import List
from datetime import datetime
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
import torch.nn.functional as F
from atlas_rag.retriever.embedding_model import NvEmbed, SentenceEmbedding
from atlas_rag.reader.llm_generator import LLMGenerator
from atlas_rag.evaluation.evaluation import QAJudger
from dataclasses import dataclass



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

@dataclass
class BenchMarkConfig:
    """
    Configuration class for benchmarking.

    Attributes:
        dataset_name (str): Name of the dataset. Default is "hotpotqa".
        question_file (str): Path to the question file. Default is "hotpotqa".
        graph_file (str): Path to the graph file. Default is "hotpotqa_concept.graphml".
        include_events (bool): Whether to include events. Default is False.
        include_concept (bool): Whether to include concepts. Default is False.
        reader_model_name (str): Name of the reader model. Default is "meta-llama/Llama-2-7b-chat-hf".
        encoder_model_name (str): Name of the encoder model. Default is "nvidia/NV-Embed-v2".
    """
    dataset_name: str = "hotpotqa"
    question_file: str = "hotpotqa"
    graph_file: str = "hotpotqa_concept.graphml"
    include_events: bool = False
    include_concept: bool = False
    reader_model_name: str = "meta-llama/Llama-2-7b-chat-hf"
    encoder_model_name: str = "nvidia/NV-Embed-v2"
        

class RAGBenchmark:
    def __init__(self, config:BenchMarkConfig,logging=False):
        self.config = config
        self.logging = logging

    def load_encoder_model(self, encoder_model_name, **kwargs):
        if encoder_model_name == "nvidia/NV-Embed-v2":
            sentence_encoder = AutoModel.from_pretrained("nvidia/NV-Embed-v2", **kwargs)
            return NvEmbed(sentence_encoder)
        else:
            sentence_encoder = SentenceTransformer(encoder_model_name, **kwargs)
            return SentenceEmbedding(sentence_encoder)

    def benchmark(self, retrievers:List[BaseRetriever], 
                  llm_generator:LLMGenerator,
                  dataset_name = "hotpotqa",
                  question_file="hotpotqa",
                  graph_file="hotpotqa_concept.graphml",
                  logger:Logger = None,
                  number_of_samples= -1,
                  ):
        qa_judge = QAJudger()
        result_list = []
        with open(question_file, "r") as f:
            data = json.load(f)
            print(f"Data loaded from {question_file}")
        if number_of_samples > 0:
            data = data[:number_of_samples]
            print(f"Using only the first {number_of_samples} samples from the dataset")
        for sample in tqdm(data):
            question = sample["question"]
            answer = sample["answer"]

            gold_file_ids = []
            if dataset_name in ("hotpotqa", "2wikimultihopqa"):
                for fact in sample["supporting_facts"]:
                    gold_file_ids.append(fact[0])
            elif dataset_name == "musique":
                for paragraph in sample["paragraphs"]:
                    if paragraph["is_supporting"]:
                        gold_file_ids.append(paragraph["paragraph_text"])
            else:
                print("Dataset not supported")
                continue
            
            result = {
                "question": question,
                "answer": answer,
                "gold_file_ids": gold_file_ids,
            } 
            
            if logger is not None:
                logger.info(f"Question: {question}")
            for retriever in retrievers:
                sorted_context, sorted_context_ids = retriever.retrieve(question, topN=5)
                
                if isinstance(retriever, BaseEdgeRetriever):
                    sorted_context = ". ".join(sorted_context)
                    llm_generated_answer = llm_generator.generate_with_context_kg(question, sorted_context, max_new_tokens=2048, temperature=0.5)
                elif isinstance(retriever, BasePassageRetriever):
                    sorted_context = "\n".join(sorted_context)
                    llm_generated_answer = llm_generator.generate_with_context(question, sorted_context, max_new_tokens=2048, temperature=0.5)
                
                short_answer = qa_judge.split_answer(llm_generated_answer)
                em, f1 = qa_judge.judge(short_answer, answer)
                
                
                result[f"{retriever.__class__.__name__ }_em"] = em
                result[f"{retriever.__class__.__name__ }_f1"] = f1
                result[f"{retriever.__class__.__name__ }_passages"] = sorted_context
                result[f"{retriever.__class__.__name__ }_id"] = sorted_context_ids
                result[f"{retriever.__class__.__name__ }_generated_answer"] = llm_generated_answer
                result[f"{retriever.__class__.__name__ }short_answer"] = short_answer
                
                # Calculate recall
                if dataset_name in ("hotpotqa", "2wikimultihopqa"):
                    recall_2, recall_5 = qa_judge.recall(sorted_context_ids, gold_file_ids)
                elif dataset_name == "musique":
                    recall_2, recall_5 = qa_judge.recall(sorted_context, gold_file_ids)
                
                result[f"{retriever.__class__.__name__ }_recall@2"] = recall_2
                result[f"{retriever.__class__.__name__ }_recall@5"] = recall_5
                
            result_list.append(result)


        self.save_results(result_list, [retriever.__class__.__name__ for retriever in retrievers])
    

    def save_results(self, result_list, retriever_names:List[str]):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d%H%M%S")
        
        dataset_name = self.config.dataset_name
        include_events = self.config.include_events
        include_concept = self.config.include_concept
        encoder_model_name = self.config.encoder_model_name
        reader_model_name = self.config.reader_model_name
        
        summary_file = f"./result/{dataset_name}/summary_{formatted_time}_event{include_events}_concept{include_concept}_{encoder_model_name}_{reader_model_name}.json"
        if not os.path.exists(os.path.dirname(summary_file)):
            os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        
        result_dir = f"./result/{dataset_name}/result_{formatted_time}_event{include_events}_concept{include_concept}_{encoder_model_name}_{reader_model_name}.json"
        if not os.path.exists(os.path.dirname(result_dir)):
            os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        
        summary_dict = self.calculate_summary(result_list, retriever_names)
        
        with open(summary_file, "w") as f_summary:
            json.dump(summary_dict, f_summary)
            f_summary.write("\n")

        with open(result_dir, "w") as f:
            for result in result_list:
                json.dump(result, f)
                f.write("\n")
    
    def calculate_summary(self, result_list, method):
        summary_dict = {}
        for retriever_name in method:
            if not all(f"{retriever_name}_em" in result for result in result_list):
                raise ValueError(f"Missing {retriever_name}_em in results")
            if not all(f"{retriever_name}_f1" in result for result in result_list):
                raise ValueError(f"Missing {retriever_name}_f1 in results")
            if not all(f"{retriever_name}_recall@2" in result for result in result_list):
                raise ValueError(f"Missing {retriever_name}_recall@2 in results")
            if not all(f"{retriever_name}_recall@5" in result for result in result_list):
                raise ValueError(f"Missing {retriever_name}_recall@5 in results")
            
            average_em = sum([result[f"{retriever_name}_em"] for result in result_list]) / len(result_list)
            average_f1 = sum([result[f"{retriever_name}_f1"] for result in result_list]) / len(result_list)
            average_recall_2 = sum([result[f"{retriever_name}_recall@2"] for result in result_list]) / len(result_list)
            average_recall_5 = sum([result[f"{retriever_name}_recall@5"] for result in result_list]) / len(result_list)
            summary_dict.update( {
                "average_f1": average_f1,
                "average_em": average_em,
                "average_recall@2": average_recall_2,
                "average_recall@5": average_recall_5,
            })
        return summary_dict
