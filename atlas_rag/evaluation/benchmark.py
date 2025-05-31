import os
import json
import numpy as np
import logging
from datetime import datetime
from configparser import ConfigParser
import networkx as nx
from transformers import AutoModel
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import argparse
import torch
import torch.nn.functional as F

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
class RAGBenchmark:
    def __init__(self):
        self.config = ConfigParser()
        self.config.read('hipporag_opendomainqa/config.ini')

    def setup_logger(self, keyword, include_events, include_concept, method):
        log_file_path = f'./log/{keyword}_event{include_events}_concept{include_concept}_{method}.log'
        logger = logging.getLogger("RAGBenchmarkLogger")
        logger.setLevel(logging.INFO)
        max_bytes = 50 * 1024 * 1024  # 50 MB
        if not os.path.exists(os.path.dirname(log_file_path)):
            os.makedirs(os.path.dirname(log_file_path), exist_ok=True)
        handler = logging.handlers.RotatingFileHandler(log_file_path, maxBytes=max_bytes, backupCount=5)
        logger.addHandler(handler)
        return logger

    def load_encoder_model(self, encoder_model_name):
        if encoder_model_name == "nvidia/NV-Embed-v2":
            sentence_encoder = AutoModel.from_pretrained("nvidia/NV-Embed-v2", device_map="auto", trust_remote_code=True)
            return NvEmbed(sentence_encoder)
        else:
            sentence_encoder = SentenceTransformer(encoder_model_name, device="cuda")
            return MiniLM(sentence_encoder)

    def load_data(self, keyword, include_events, include_concept, encoder_model_name):
        loading_dir_encoder_name = "NV-Embed-v2" if encoder_model_name == "nvidia/NV-Embed-v2" else encoder_model_name
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
        
        # Sanity checks
        assert len(node_list) == len(node_embeddings), "Mismatch in node list and embeddings lengths."
        assert len(edge_list) == len(edge_embeddings) == edge_faiss_index.ntotal, "Mismatch in edge list, embeddings, and FAISS index."

        return (data, node_list, edge_list, original_text_list, original_text_dict_with_node_id, edge_faiss_index, text_embeddings, node_embeddings, edge_embeddings)

    def benchmark(self, dataset_name="hotpotqa", model_name="meta-llama/Llama-3.3-70B-Instruct-Turbo", include_events=False, include_concept=False, encoder_model_name="nvidia/NV-Embed-v2", hipporag2mode='query2edge', method='hipporag2'):
        logger = self.setup_logger(dataset_name, include_events, include_concept, method)
        embedding_model = self.load_encoder_model(encoder_model_name)
        data, node_list, edge_list, original_text_list, original_text_dict_with_node_id, edge_faiss_index, text_embeddings, node_embeddings, edge_embeddings = self.load_data(dataset_name, include_events, include_concept, encoder_model_name)
        
        # Load Knowledge Graph
        graph_dir = f"./processed_data/kg_graphml/{dataset_name}_concept.graphml"
        KG = nx.read_graphml(graph_dir)
        
        # Prepare mappings
        text_id_list = []
        node_id_to_file_id = {}
        file_id_to_node_id = {}
        for node_id in tqdm(list(KG.nodes)):
            node_id_to_file_id[node_id] = KG.nodes[node_id]["id"] if dataset_name == "musique" and KG.nodes[node_id]['type'] == "passage" else KG.nodes[node_id]["file_id"]
            if KG.nodes[node_id]['file_id'] not in file_id_to_node_id:
                file_id_to_node_id[KG.nodes[node_id]['file_id']] = []
            if KG.nodes[node_id]['type'] == "passage":
                file_id_to_node_id[KG.nodes[node_id]['file_id']].append(node_id)
        for node_id in tqdm(list(original_text_dict_with_node_id.keys())):
            text_id_list.append(node_id)
        
        assert len(text_id_list) == len(original_text_dict_with_node_id)
        
        # Initialize Judger and Generator
        llm_judge = QAJudger()
        llm_generator = LLMGenerator(model_name)

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
        
        question_data_dir = f"./data/{dataset_name}.json"
        result_list = []
        
        with open(question_data_dir, "r") as f:
            data = json.load(f)
            print(f"Data loaded from {question_data_dir}")
        
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
            
            logger.info(f"Question: {question}")
            result = self.perform_retrieval(retriever, method, question, answer, gold_file_ids)
            result_list.append(result)
        
        self.save_results(result_list, method, dataset_name, include_events, include_concept, encoder_model_name, loading_dir_encoder_name)
    
    def perform_retrieval(self, retriever, method, question, answer, gold_file_ids):
        if method == 'tog':
            tog_triples, tog_generated_answer = retriever.retrieve_path(question, topN=5, Dmax=3)
            tog_short_answer = llm_judge.split_answer(tog_generated_answer)
            tog_em, tog_f1 = llm_judge.judge(tog_short_answer, answer)
            return {
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
            return {
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
            return {
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

    def save_results(self, result_list, method, dataset_name, include_events, include_concept, encoder_model_name, loading_dir_encoder_name):
        current_time = datetime.now()
        formatted_time = current_time.strftime("%Y%m%d%H%M%S")
        
        summary_file = f"./result/{dataset_name}/summary_{formatted_time}_event{include_events}_concept{include_concept}_{loading_dir_encoder_name}_{method}.json"
        if not os.path.exists(os.path.dirname(summary_file)):
            os.makedirs(os.path.dirname(summary_file), exist_ok=True)
        
        result_dir = f"./result/{dataset_name}/result_{formatted_time}_event{include_events}_concept{include_concept}_{loading_dir_encoder_name}_{method}.json"
        if not os.path.exists(os.path.dirname(result_dir)):
            os.makedirs(os.path.dirname(result_dir), exist_ok=True)
        
        summary_dict = self.calculate_summary(result_list, method)
        
        with open(summary_file, "w") as f_summary:
            json.dump(summary_dict, f_summary)
            f_summary.write("\n")

        with open(result_dir, "w") as f:
            for result in result_list:
                json.dump(result, f)
                f.write("\n")
    
    def calculate_summary(self, result_list, method):
        average_em = sum([result[f"{method}_em"] for result in result_list]) / len(result_list)
        average_f1 = sum([result[f"{method}_f1"] for result in result_list]) / len(result_list)
        return {
            "average_f1": average_f1,
            "average_em": average_em,
        }

if __name__ == "__main__":
    benchmark = RAGBenchmark()
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("--dataset_name", type=str, default="hotpotqa")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.3-70B-Instruct-Turbo")
    parser.add_argument("--include_events", action="store_true", default=False)
    parser.add_argument("--include_concept", action="store_true", default=False)
    parser.add_argument("--encoder_model_name", type=str, default="nvidia/NV-Embed-v2")
    parser.add_argument("--hipporag2mode", type=str, default='query2edge', choices=['query2edge'], help="Choose the mode for HippoRAG2 retriever")
    parser.add_argument("--method", type=str, default='hipporag2', choices=['hipporag', 'hipporag2', 'tog'], help="Choose which method to use for retrieval")

    args = parser.parse_args()
    benchmark.benchmark(args.dataset_name, args.model_name, args.include_events, args.include_concept, args.encoder_model_name, args.hipporag2mode, args.method)