#!/usr/bin/env python3
"""
NewWork Concept Graph RAG Benchmark
将NewWork生成的概念图谱集成到AutoSchemaKG RAG系统中进行测试
"""

import os
import sys
import json
import pandas as pd
import pickle
import networkx as nx
import numpy as np
from pathlib import Path
from typing import Dict, List, Any

# 添加路径
sys.path.append('..')
sys.path.append('.')

from config_loader import ConfigLoader, create_model_client


class NewWorkToAtlasConverter:
    """将NewWork概念图谱转换为Atlas RAG兼容格式"""
    
    def __init__(self, output_path: str):
        """
        Args:
            output_path: NewWork输出路径，如 "output/simple_test"
        """
        self.output_path = Path(output_path)
        self.concepts_csv = self.output_path / "concept_csv" / "concepts_Dulce_test.csv"
        self.relationships_csv = self.output_path / "concept_csv" / "relationships_Dulce_test.csv"
        self.graph_pkl = self.output_path / "graph" / "dulce_simple.pkl"
        
    def load_concept_graph(self) -> nx.Graph:
        """加载概念图"""
        if self.graph_pkl.exists():
            with open(self.graph_pkl, 'rb') as f:
                return pickle.load(f)
        else:
            raise FileNotFoundError(f"Graph file not found: {self.graph_pkl}")
    
    def load_concepts_and_relations(self) -> tuple:
        """加载概念和关系数据"""
        concepts_df = pd.read_csv(self.concepts_csv)
        relationships_df = pd.read_csv(self.relationships_csv)
        return concepts_df, relationships_df
    
    def convert_to_atlas_format(self) -> Dict[str, Any]:
        """转换为Atlas RAG兼容格式"""
        print("🔄 Converting NewWork graph to Atlas format...")
        
        # 加载数据
        G = self.load_concept_graph()
        concepts_df, relationships_df = self.load_concepts_and_relations()
        
        # 构建节点列表
        node_list = []
        node_dict = {}
        
        for idx, row in concepts_df.iterrows():
            node_id = str(idx)
            node_data = {
                'id': row['name'],
                'text': row['description'] if pd.notna(row['description']) else row['name'],
                'type': row['type'],
                'abstraction_level': row['abstraction_level']
            }
            node_list.append(node_id)
            node_dict[node_id] = node_data
            
            # 添加到NetworkX图中
            if node_id not in G.nodes:
                G.add_node(node_id, **node_data)
        
        # 构建边列表
        edge_list = []
        edge_dict = {}
        
        for idx, row in relationships_df.iterrows():
            source_name = row['source']
            target_name = row['target']
            
            # 查找对应的节点ID
            source_id = None
            target_id = None
            
            for node_idx, concept_row in concepts_df.iterrows():
                if concept_row['name'] == source_name:
                    source_id = str(node_idx)
                if concept_row['name'] == target_name:
                    target_id = str(node_idx)
            
            if source_id and target_id:
                edge_key = (source_id, target_id)
                edge_list.append(edge_key)
                
                edge_data = {
                    'relation': row['relation'],
                    'description': row['description'] if pd.notna(row['description']) else row['relation']
                }
                edge_dict[edge_key] = edge_data
                
                # 添加到NetworkX图中
                G.add_edge(source_id, target_id, **edge_data)
        
        print(f"✅ Converted: {len(node_list)} nodes, {len(edge_list)} edges")
        
        return {
            'KG': G,
            'node_list': node_list,
            'edge_list': edge_list,
            'node_dict': node_dict,
            'edge_dict': edge_dict,
            'concepts_df': concepts_df,
            'relationships_df': relationships_df
        }


class NewWorkRAGTester:
    """NewWork概念图的RAG测试器"""
    
    def __init__(self, config_loader: ConfigLoader, atlas_data: Dict[str, Any]):
        """
        Args:
            config_loader: NewWork配置加载器
            atlas_data: 转换后的Atlas格式数据
        """
        self.config_loader = config_loader
        self.atlas_data = atlas_data
        self.model = create_model_client(config_loader)
        
        # 设置句子编码器
        self._setup_sentence_encoder()
        
        # 创建embeddings
        self._create_embeddings()
    
    def _setup_sentence_encoder(self):
        """设置句子编码器"""
        try:
            from sentence_transformers import SentenceTransformer
            from atlas_rag.vectorstore.embedding_model import SentenceEmbedding
            
            # 正确创建SentenceTransformer对象
            transformer = SentenceTransformer("all-MiniLM-L6-v2")
            self.sentence_encoder = SentenceEmbedding(transformer)
            print("✅ Sentence encoder loaded: all-MiniLM-L6-v2")
        except ImportError:
            print("❌ Failed to load Atlas sentence encoder")
            raise
    
    def _create_embeddings(self):
        """创建节点和边的embeddings"""
        print("🔄 Creating embeddings for nodes and edges...")
        
        # 节点embeddings
        node_texts = []
        for node_id in self.atlas_data['node_list']:
            node_data = self.atlas_data['KG'].nodes[node_id]
            text = f"{node_data.get('id', '')} {node_data.get('text', '')}"
            node_texts.append(text)
        
        # 直接使用SentenceTransformer进行编码
        node_embeddings = self.sentence_encoder.sentence_encoder.encode(node_texts)
        
        # 边embeddings
        edge_texts = []
        for edge in self.atlas_data['edge_list']:
            source_node = self.atlas_data['KG'].nodes[edge[0]]
            target_node = self.atlas_data['KG'].nodes[edge[1]]
            edge_data = self.atlas_data['KG'].edges[edge]
            
            text = f"{source_node.get('id', '')} {edge_data.get('relation', '')} {target_node.get('id', '')}"
            edge_texts.append(text)
        
        # 直接使用SentenceTransformer进行编码
        edge_embeddings = self.sentence_encoder.sentence_encoder.encode(edge_texts)
        
        # 创建FAISS索引
        import faiss
        
        # 节点FAISS索引
        node_faiss_index = faiss.IndexFlatIP(node_embeddings.shape[1])
        node_faiss_index.add(node_embeddings.astype('float32'))
        
        # 边FAISS索引  
        edge_faiss_index = faiss.IndexFlatIP(edge_embeddings.shape[1])
        edge_faiss_index.add(edge_embeddings.astype('float32'))
        
        self.atlas_data.update({
            'node_embeddings': node_embeddings,
            'edge_embeddings': edge_embeddings,
            'node_faiss_index': node_faiss_index,
            'edge_faiss_index': edge_faiss_index
        })
        
        print("✅ Embeddings and FAISS indices created")
    
    def test_simple_graph_retriever(self, query: str, topN: int = 5) -> List[str]:
        """测试简单图检索器（自定义实现）"""
        try:
            print(f"🔍 Simple Graph Retriever results for '{query}':")
            
            # 使用自定义的简单检索方法
            query_embedding = self.sentence_encoder.sentence_encoder.encode([query])
            
            # 搜索最相似的边
            D, I = self.atlas_data['edge_faiss_index'].search(query_embedding, topN)
            
            results = []
            for i, (distance, index) in enumerate(zip(D[0], I[0]), 1):
                if index < len(self.atlas_data['edge_list']):
                    edge = self.atlas_data['edge_list'][index]
                    source_node = self.atlas_data['KG'].nodes[edge[0]]
                    target_node = self.atlas_data['KG'].nodes[edge[1]]
                    edge_data = self.atlas_data['KG'].edges[edge]
                    
                    result = f"{source_node.get('id', edge[0])} {edge_data.get('relation', '')} {target_node.get('id', edge[1])}"
                    results.append(result)
                    print(f"   {i}. {result} (similarity: {1-distance:.3f})")
            
            return results
            
        except Exception as e:
            print(f"❌ Simple Graph Retriever failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def test_node_retriever(self, query: str, topN: int = 5) -> List[str]:
        """测试节点检索器（自定义实现）"""
        try:
            print(f"🔍 Node Retriever results for '{query}':")
            
            # 使用节点embeddings进行检索
            query_embedding = self.sentence_encoder.sentence_encoder.encode([query])
            
            # 搜索最相似的节点
            D, I = self.atlas_data['node_faiss_index'].search(query_embedding, topN)
            
            results = []
            for i, (distance, index) in enumerate(zip(D[0], I[0]), 1):
                if index < len(self.atlas_data['node_list']):
                    node_id = self.atlas_data['node_list'][index]
                    node_data = self.atlas_data['KG'].nodes[node_id]
                    
                    result = f"{node_data.get('id', node_id)} ({node_data.get('type', 'unknown')}): {node_data.get('text', '')}"
                    results.append(result)
                    print(f"   {i}. {result} (similarity: {1-distance:.3f})")
            
            return results
            
        except Exception as e:
            print(f"❌ Node Retriever failed: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def run_benchmark_queries(self, test_queries: List[str]) -> Dict[str, Any]:
        """运行benchmark查询"""
        print("\n🚀 Running RAG Benchmark Tests")
        print("=" * 50)
        
        results = {
            'simple_graph': {},
            'node_retriever': {},
            'queries': test_queries
        }
        
        for query in test_queries:
            print(f"\n📝 Query: {query}")
            print("-" * 30)
            
            # 测试SimpleGraphRetriever
            simple_results = self.test_simple_graph_retriever(query)
            results['simple_graph'][query] = simple_results
            
            print()
            
            # 测试NodeRetriever
            node_results = self.test_node_retriever(query)
            results['node_retriever'][query] = node_results
            
            print()
        
        return results
    
    def save_results(self, results: Dict[str, Any], output_file: str):
        """保存测试结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Results saved to: {output_file}")


def create_test_queries_for_dulce():
    """为Dulce数据创建测试查询"""
    return [
        "Who is Agent Alex Mercer?",
        "What is Operation: Dulce?",
        "What is the Paranormal Military Squad?",
        "Who are the team members involved?",
        "What happens in the briefing room?",
        "What are the protocols mentioned?",
        "Who shows compliance in the team?",
        "What anomalies are being investigated?"
    ]


def main():
    """主函数"""
    print("🌟 NewWork Concept Graph RAG Benchmark")
    print("=" * 60)
    
    try:
        # 1. 加载NewWork配置
        print("📋 Loading NewWork configuration...")
        config_loader = ConfigLoader()
        config_loader.print_config_summary()
        
        # 2. 转换概念图谱
        print("\n🔄 Converting concept graph...")
        converter = NewWorkToAtlasConverter("output/simple_test")
        atlas_data = converter.convert_to_atlas_format()
        
        # 3. 初始化RAG测试器
        print("\n🤖 Initializing RAG tester...")
        rag_tester = NewWorkRAGTester(config_loader, atlas_data)
        
        # 4. 创建测试查询
        test_queries = create_test_queries_for_dulce()
        print(f"\n📝 Created {len(test_queries)} test queries")
        
        # 5. 运行benchmark测试
        results = rag_tester.run_benchmark_queries(test_queries)
        
        # 6. 保存结果
        output_file = "output/simple_test/rag_benchmark_results.json"
        rag_tester.save_results(results, output_file)
        
        print("\n🎉 RAG Benchmark completed successfully!")
        print(f"📊 Results saved to: {output_file}")
        
        # 7. 显示摘要
        print(f"\n📈 Summary:")
        print(f"   📊 Graph: {len(atlas_data['node_list'])} nodes, {len(atlas_data['edge_list'])} edges")
        print(f"   🔍 Queries tested: {len(test_queries)}")
        print(f"   🤖 RAG methods: SimpleGraphRetriever, NodeRetriever")
        
    except KeyboardInterrupt:
        print("\n⚠️ Benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()