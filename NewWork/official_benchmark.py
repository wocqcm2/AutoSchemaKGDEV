#!/usr/bin/env python3
"""
Official AutoSchemaKG Benchmark for HotpotQA Knowledge Graph
使用AutoSchemaKG官方测试系统和标准数据集
计算论文中的标准指标：EM, F1, Recall@2, Recall@5
"""

import os
import sys
import json
import logging
from pathlib import Path

# 添加路径
sys.path.append('..')
sys.path.append('.')

from config_loader import ConfigLoader, create_model_client
from atlas_rag.evaluation.benchmark import RAGBenchmark, BenchMarkConfig
from atlas_rag.vectorstore.embedding_model import SentenceEmbedding
from atlas_rag.retriever.simple_retriever import SimpleGraphRetriever, SimpleTextRetriever
from atlas_rag.retriever.hipporag import HippoRAGRetriever  
from atlas_rag.retriever.hipporag2 import HippoRAG2Retriever
from atlas_rag.retriever.tog import TogRetriever
from atlas_rag.logging import setup_logger
from sentence_transformers import SentenceTransformer
import networkx as nx
import pandas as pd
import pickle
import numpy as np
import faiss


class HotpotKGDataLoader:
    """加载HotpotQA知识图谱数据，适配官方benchmark格式"""
    
    def __init__(self, kg_path: str = "output/hotpot_kg"):
        self.kg_path = Path(kg_path)
        self.concepts_csv = self.kg_path / "concept_csv" / "concepts_hotpot_kg.csv"
        self.relationships_csv = self.kg_path / "concept_csv" / "relationships_hotpot_kg.csv"
        self.graph_pkl = self.kg_path / "graph" / "hotpot_kg.pkl"
        
    def load_kg_data(self):
        """加载知识图谱数据"""
        print("🔄 Loading HotpotQA Knowledge Graph...")
        
        # 加载NetworkX图
        with open(self.graph_pkl, 'rb') as f:
            G = pickle.load(f)
        
        # 加载CSV数据
        concepts_df = pd.read_csv(self.concepts_csv)
        relationships_df = pd.read_csv(self.relationships_csv)
        
        # 构建节点列表和数据
        node_list = []
        node_dict = {}
        text_dict = {}
        
        for idx, row in concepts_df.iterrows():
            node_id = str(idx)
            node_data = {
                'id': row['id'],
                'text': row['text'],
                'type': row['type'],
                'abstraction_level': row['abstraction_level']
            }
            node_list.append(node_id)
            node_dict[node_id] = node_data
            text_dict[node_id] = row['text']
            
            # 确保节点在图中
            if node_id not in G.nodes:
                G.add_node(node_id, **node_data)
        
        # 构建边列表
        edge_list = []
        for idx, row in relationships_df.iterrows():
            source_name = row['source']
            target_name = row['target']
            
            # 查找对应的节点ID
            source_id = None
            target_id = None
            
            for node_idx, concept_row in concepts_df.iterrows():
                if concept_row['id'] == source_name:
                    source_id = str(node_idx)
                if concept_row['id'] == target_name:
                    target_id = str(node_idx)
            
            if source_id and target_id:
                edge_list.append((source_id, target_id))
                
                # 重要：同时添加到NetworkX图中！
                edge_data = {
                    'relation': row['relation'],
                    'description': row.get('relation_type', row['relation'])
                }
                G.add_edge(source_id, target_id, **edge_data)
        
        # 验证图的结构
        print(f"✅ Loaded: {len(node_list)} nodes, {len(edge_list)} edges")
        print(f"🔍 NetworkX Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        
        # 验证一些边是否真的在图中
        if edge_list:
            sample_edges = edge_list[:min(3, len(edge_list))]
            for edge in sample_edges:
                if G.has_edge(*edge):
                    print(f"✅ Sample edge {edge} exists in graph")
                else:
                    print(f"❌ Sample edge {edge} NOT in graph!")
        
        # 验证节点数据
        if node_list:
            sample_node = node_list[0]
            if sample_node in G.nodes:
                node_data = G.nodes[sample_node]
                print(f"✅ Sample node {sample_node}: {list(node_data.keys())}")
            else:
                print(f"❌ Sample node {sample_node} NOT in graph!")
        
        # 为官方retriever添加更多必需的数据字段
        return {
            'KG': G,
            'node_list': node_list,
            'edge_list': edge_list,
            'text_dict': text_dict,
            'original_text_dict_with_node_id': text_dict,
            'node_dict': node_dict,
            'edge_dict': {edge: {'relation': G.edges[edge].get('relation', 'related_to')} for edge in edge_list if G.has_edge(*edge)},
            'original_text_dict': text_dict,
            'passage_dict': text_dict  # SimpleTextRetriever 需要
        }


def create_embeddings_and_indices(data, sentence_encoder):
    """创建embeddings和FAISS索引"""
    print("🔄 Creating embeddings and FAISS indices...")
    
    # 创建节点embeddings
    node_texts = []
    for node_id in data['node_list']:
        if node_id in data['KG'].nodes:
            node_data = data['KG'].nodes[node_id]
            text = node_data.get('text', node_data.get('id', str(node_id)))
            node_texts.append(text)
        else:
            node_texts.append(str(node_id))
    
    # 创建边embeddings
    edge_texts = []
    for edge in data['edge_list']:
        if len(edge) >= 2:
            source_data = data['KG'].nodes.get(edge[0], {})
            target_data = data['KG'].nodes.get(edge[1], {})
            edge_data = data['KG'].edges.get(edge, {})
            
            source_text = source_data.get('text', source_data.get('id', str(edge[0])))
            target_text = target_data.get('text', target_data.get('id', str(edge[1])))
            relation = edge_data.get('relation', 'related_to')
            
            edge_text = f"{source_text} {relation} {target_text}"
            edge_texts.append(edge_text)
    
    # 计算embeddings
    if node_texts:
        node_embeddings = sentence_encoder.encode(node_texts)
        if len(node_embeddings.shape) == 1:
            node_embeddings = node_embeddings.reshape(1, -1)
    else:
        node_embeddings = np.zeros((1, 384))
    
    if edge_texts:
        edge_embeddings = sentence_encoder.encode(edge_texts)
        if len(edge_embeddings.shape) == 1:
            edge_embeddings = edge_embeddings.reshape(1, -1)
    else:
        edge_embeddings = np.zeros((1, 384))
    
    # 创建FAISS索引
    if node_embeddings.shape[0] > 0 and node_embeddings.shape[1] > 0:
        node_index = faiss.IndexFlatIP(node_embeddings.shape[1])
        normalized_node_emb = node_embeddings / np.linalg.norm(node_embeddings, axis=1, keepdims=True)
        node_index.add(normalized_node_emb.astype('float32'))
    else:
        node_index = faiss.IndexFlatIP(384)
    
    if edge_embeddings.shape[0] > 0 and edge_embeddings.shape[1] > 0:
        edge_index = faiss.IndexFlatIP(edge_embeddings.shape[1])
        normalized_edge_emb = edge_embeddings / np.linalg.norm(edge_embeddings, axis=1, keepdims=True)
        edge_index.add(normalized_edge_emb.astype('float32'))
    else:
        edge_index = faiss.IndexFlatIP(384)
    
    # 更新数据
    data.update({
        'node_embeddings': node_embeddings,
        'edge_embeddings': edge_embeddings,
        'text_embeddings': node_embeddings,  # 用于某些retriever
        'node_faiss_index': node_index,
        'edge_faiss_index': edge_index,
    })
    
    print(f"✅ Created embeddings: nodes{node_embeddings.shape}, edges{edge_embeddings.shape}")
    return data


def setup_official_benchmark(dataset_name: str, question_file: str, num_samples: int = 50):
    """配置官方benchmark"""
    print(f"\n⚙️ Setting up official AutoSchemaKG benchmark...")
    print(f"📊 Dataset: {dataset_name}")
    print(f"📁 Question file: {question_file}")
    print(f"📊 Samples: {num_samples}")
    
    config_loader = ConfigLoader()
    
    # 检查问题文件是否存在
    if not Path(question_file).exists():
        raise FileNotFoundError(f"Question file not found: {question_file}")
    
    # 创建官方benchmark配置
    benchmark_config = BenchMarkConfig(
        dataset_name=dataset_name,
        question_file=question_file,
        include_concept=True,
        include_events=True,
        reader_model_name=config_loader.config_data.get('models', {}).get('qwen_235b', 'qwen'),
        encoder_model_name="all-MiniLM-L6-v2",
        number_of_samples=num_samples,
        react_max_iterations=3
    )
    
    # 设置logger
    logger = setup_logger(benchmark_config)
    
    return benchmark_config, logger, config_loader


def setup_retrievers(data, sentence_encoder, llm_generator):
    """设置retriever列表"""
    print("🔧 Setting up retrievers for official benchmark...")
    
    retrievers = []
    
    try:
        # 1. SimpleGraphRetriever
        simple_graph = SimpleGraphRetriever(
            llm_generator=llm_generator,
            sentence_encoder=sentence_encoder,
            data=data
        )
        retrievers.append(simple_graph)
        print("✅ SimpleGraphRetriever added")
        
        # 2. SimpleTextRetriever
        passage_dict = data.get('text_dict', {})
        simple_text = SimpleTextRetriever(
            passage_dict=passage_dict,
            sentence_encoder=sentence_encoder,
            data=data
        )
        retrievers.append(simple_text)
        print("✅ SimpleTextRetriever added")
        
        # 3. TogRetriever
        try:
            tog_retriever = TogRetriever(
                llm_generator=llm_generator,
                sentence_encoder=sentence_encoder,
                data=data
            )
            retrievers.append(tog_retriever)
            print("✅ TogRetriever added")
        except Exception as e:
            print(f"⚠️ TogRetriever failed: {e}")
        
        # 4. HippoRAGRetriever
        try:
            # 为HippoRAG准备数据
            hippo_data = data.copy()
            file_id_to_node_id = {}
            
            for node_id in data['node_list']:
                if node_id in hippo_data['KG'].nodes:
                    hippo_data['KG'].nodes[node_id]['type'] = 'passage'
                    file_id = f"file_{node_id}"
                    hippo_data['KG'].nodes[node_id]['file_id'] = file_id
                    file_id_to_node_id[file_id] = [node_id]
            
            # 添加 HippoRAG 需要的映射
            hippo_data['file_id_to_node_id'] = file_id_to_node_id
            
            hipporag = HippoRAGRetriever(
                llm_generator=llm_generator,
                sentence_encoder=sentence_encoder,
                data=hippo_data
            )
            retrievers.append(hipporag)
            print("✅ HippoRAGRetriever added")
        except Exception as e:
            print(f"⚠️ HippoRAGRetriever failed: {e}")
            import traceback
            traceback.print_exc()
        
        # 5. HippoRAG2Retriever
        try:
            hipporag2 = HippoRAG2Retriever(
                llm_generator=llm_generator,
                sentence_encoder=sentence_encoder,
                data=data
            )
            retrievers.append(hipporag2)
            print("✅ HippoRAG2Retriever added")
        except Exception as e:
            print(f"⚠️ HippoRAG2Retriever failed: {e}")
            
    except Exception as e:
        print(f"❌ Setup retrievers failed: {e}")
        import traceback
        traceback.print_exc()
    
    print(f"📋 Total retrievers setup: {len(retrievers)}")
    return retrievers


def run_official_benchmark(dataset_name: str = "hotpotqa", 
                          question_file: str = "../benchmark_data/hotpotqa.json",
                          num_samples: int = 50):
    """运行官方AutoSchemaKG benchmark"""
    print("🚀 Starting Official AutoSchemaKG Benchmark")
    print("=" * 80)
    print("🎯 This will compute standard metrics: EM, F1, Recall@2, Recall@5")
    print("=" * 80)
    
    try:
        # 1. 设置官方benchmark
        benchmark_config, logger, config_loader = setup_official_benchmark(
            dataset_name, question_file, num_samples
        )
        
        # 2. 加载HotpotQA知识图谱
        kg_loader = HotpotKGDataLoader()
        data = kg_loader.load_kg_data()
        
        # 3. 创建sentence encoder
        transformer = SentenceTransformer("all-MiniLM-L6-v2")
        sentence_encoder = SentenceEmbedding(transformer)
        
        # 4. 创建embeddings和索引
        data = create_embeddings_and_indices(data, sentence_encoder)
        
        # 5. 创建LLM生成器
        llm_generator = create_model_client(config_loader)
        
        # 6. 设置retrievers
        retrievers = setup_retrievers(data, sentence_encoder, llm_generator)
        
        if not retrievers:
            print("❌ No retrievers available, cannot run benchmark")
            return
        
        # 7. 运行官方benchmark
        print(f"\n🚀 Running Official AutoSchemaKG Benchmark...")
        print(f"📊 Dataset: {dataset_name}")
        print(f"📊 Question file: {question_file}")
        print(f"📊 Samples: {num_samples}")
        print(f"📊 Retrievers: {[r.__class__.__name__ for r in retrievers]}")
        print("=" * 60)
        
        # 创建并运行官方RAGBenchmark
        benchmark = RAGBenchmark(config=benchmark_config, logger=logger)
        benchmark.run(retrievers, llm_generator)
        
        print(f"\n✅ Official benchmark completed!")
        print(f"📊 Results saved to: ./result/{dataset_name}/")
        print("📋 Standard metrics calculated: EM, F1, Recall@2, Recall@5")
        
        # 8. 打印摘要
        print_benchmark_summary(dataset_name, retrievers, num_samples)
        
    except Exception as e:
        print(f"❌ Official benchmark failed: {e}")
        import traceback
        traceback.print_exc()


def print_benchmark_summary(dataset_name: str, retrievers, num_samples: int):
    """打印测试摘要"""
    print("\n" + "="*80)
    print("📊 OFFICIAL AUTOSCHEMAKG BENCHMARK SUMMARY")
    print("="*80)
    print(f"🗃️  Dataset: {dataset_name}")
    print(f"📊 Samples: {num_samples}")
    print(f"🤖 Knowledge Graph: HotpotQA (1002 concepts)")
    print(f"🔧 Tested Retrievers:")
    for i, retriever in enumerate(retrievers, 1):
        print(f"   {i}. {retriever.__class__.__name__}")
    print(f"📊 Standard Metrics: EM, F1, Recall@2, Recall@5")
    print(f"💾 Results Location: ./result/{dataset_name}/")
    print("="*80)
    print("🎯 This matches the AutoSchemaKG paper evaluation setup!")


def main():
    """主函数"""
    print("🌟 Official AutoSchemaKG Benchmark for HotpotQA Knowledge Graph")
    print("=" * 80)
    print("🎯 Using official benchmark system with standard metrics")
    print("=" * 80)
    
    # 检查可用的测试数据集
    available_datasets = []
    
    # HotpotQA 是首选（与您的KG匹配）
    if Path("../benchmark_data/hotpotqa.json").exists():
        available_datasets.append(("hotpotqa", "../benchmark_data/hotpotqa.json"))
    
    # 其他数据集
    if Path("../benchmark_data/musique_sample.json").exists():
        available_datasets.append(("musique", "../benchmark_data/musique_sample.json"))
    
    if Path("../working_data/2wikimultihopqa.json").exists():
        available_datasets.append(("2wikimultihopqa", "../working_data/2wikimultihopqa.json"))
    
    if not available_datasets:
        print("❌ No official test datasets found!")
        print("💡 Expected locations:")
        print("   - ../benchmark_data/hotpotqa.json (RECOMMENDED)")
        print("   - ../benchmark_data/musique_sample.json")
        print("   - ../working_data/2wikimultihopqa.json")
        return
    
    print(f"\n📋 Available official datasets:")
    for i, (name, path) in enumerate(available_datasets, 1):
        print(f"   {i}. {name} ({path})")
    
    try:
        choice = input(f"\nSelect dataset (1-{len(available_datasets)}) [1]: ").strip() or "1"
        dataset_idx = int(choice) - 1
        
        if 0 <= dataset_idx < len(available_datasets):
            dataset_name, question_file = available_datasets[dataset_idx]
            
            # 获取样本数量（HotpotQA推荐较少样本进行初始测试）
            default_samples = "20" if dataset_name == "hotpotqa" else "50"
            try:
                samples = int(input(f"Number of samples [{default_samples}]: ").strip() or default_samples)
            except ValueError:
                samples = int(default_samples)
            
            # 运行官方benchmark
            run_official_benchmark(dataset_name, question_file, samples)
            
        else:
            print("❌ Invalid choice, using first dataset")
            dataset_name, question_file = available_datasets[0]
            default_samples = 20 if dataset_name == "hotpotqa" else 50
            run_official_benchmark(dataset_name, question_file, default_samples)
            
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()