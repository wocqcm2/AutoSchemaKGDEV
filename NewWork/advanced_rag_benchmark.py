#!/usr/bin/env python3
"""
Advanced RAG Benchmark for NewWork Concept Graph
集成AutoSchemaKG所有高级RAG方法的完整测试系统
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Tuple

# 添加路径
sys.path.append('..')
sys.path.append('.')

from config_loader import ConfigLoader, create_model_client
<<<<<<< HEAD
from rag_benchmark import NewWorkToAtlasConverter
=======

import pandas as pd
import pickle
import networkx as nx
from pathlib import Path


class HotpotKGToAtlasConverter:
    """将HotpotQA知识图谱转换为Atlas RAG兼容格式"""
    
    def __init__(self, output_path: str):
        """
        Args:
            output_path: HotpotKG输出路径，如 "output/hotpot_kg"
        """
        self.output_path = Path(output_path)
        
        # HotpotQA数据格式
        concept_csv_dir = self.output_path / "concept_csv"
        self.concepts_csv = concept_csv_dir / "concepts_hotpot_kg.csv"
        self.relationships_csv = concept_csv_dir / "relationships_hotpot_kg.csv"
        
        # 图谱文件
        graph_dir = self.output_path / "graph"
        pkl_files = list(graph_dir.glob("*.pkl"))
        if pkl_files:
            self.graph_pkl = pkl_files[0]
        else:
            raise FileNotFoundError(f"未找到PKL图谱文件在: {graph_dir}")
            
    def load_concept_graph(self) -> nx.Graph:
        """加载概念图"""
        print(f"🔄 加载图谱文件: {self.graph_pkl}")
        with open(self.graph_pkl, 'rb') as f:
            return pickle.load(f)
    
    def load_concepts_and_relations(self) -> tuple:
        """加载概念和关系数据"""
        concepts_df = pd.read_csv(self.concepts_csv)
        relationships_df = pd.read_csv(self.relationships_csv)
        return concepts_df, relationships_df
    
    def convert_to_atlas_format(self) -> Dict[str, Any]:
        """转换为Atlas RAG兼容格式"""
        print("🔄 Converting HotpotQA graph to Atlas format...")
        
        # 加载数据
        G = self.load_concept_graph()
        concepts_df, relationships_df = self.load_concepts_and_relations()
        
        # 构建节点列表 - 适配HotpotQA格式
        node_list = []
        node_dict = {}
        
        for idx, row in concepts_df.iterrows():
            node_id = str(idx)
            node_data = {
                'id': row['id'],
                'text': row['text'] if pd.notna(row['text']) else row['id'],
                'type': row['type'],
                'abstraction_level': row['abstraction_level']
            }
            node_list.append(node_id)
            node_dict[node_id] = node_data
            
            # 添加到NetworkX图中
            if node_id not in G.nodes:
                G.add_node(node_id, **node_data)
        
        # 构建边列表 - 适配HotpotQA格式
        edge_list = []
        edge_dict = {}
        
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
                edge_key = (source_id, target_id)
                edge_list.append(edge_key)
                
                edge_data = {
                    'relation': row['relation'],
                    'description': row.get('relation_type', row['relation'])
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
            'original_text_dict': node_dict,
            'original_text_dict_with_node_id': {node_id: data['text'] for node_id, data in node_dict.items()}
        }
>>>>>>> 0ff854f19280eadc04f4289414abf37019510f1e


class CompatibleEmbeddingWrapper:
    """兼容AutoSchemaKG的embedding包装器"""
    
    def __init__(self, sentence_encoder):
        self.sentence_encoder = sentence_encoder
        # 直接获取SentenceTransformer对象
        self.transformer = sentence_encoder.sentence_encoder
    
    def encode(self, query, query_type=None, **kwargs):
        """
        兼容query_type参数的encode方法
        
        Args:
            query: 输入文本或文本列表
            query_type: 查询类型 ('node', 'edge', 'passage', 等)
            **kwargs: 其他参数
        """
        try:
            # 直接使用SentenceTransformer对象，忽略query_type
            return self.transformer.encode(query)
        except Exception as e:
            print(f"⚠️ Direct encoding failed: {e}")
            # 尝试包装成列表
            try:
                if isinstance(query, str):
                    return self.transformer.encode([query])[0]
                else:
                    return self.transformer.encode(query)
            except Exception as e2:
                print(f"❌ All encoding methods failed: {e2}")
                # 最后的回退：创建零向量
                import numpy as np
                if isinstance(query, list):
                    return np.zeros((len(query), 384))
                else:
                    return np.zeros(384)


class AdvancedRAGTester:
    """高级RAG测试器，集成所有AutoSchemaKG RAG方法"""
    
    def __init__(self, config_loader: ConfigLoader, atlas_data: Dict[str, Any]):
        self.config_loader = config_loader
        self.atlas_data = atlas_data
        self.model = create_model_client(config_loader)
        
        # 设置兼容的sentence encoder
        self._setup_compatible_encoder()
        
        # 准备高级RAG所需的数据
        self._prepare_advanced_data()
    
    def _setup_compatible_encoder(self):
        """设置兼容的句子编码器"""
        try:
            from sentence_transformers import SentenceTransformer
            from atlas_rag.vectorstore.embedding_model import SentenceEmbedding
            
            # 正确创建SentenceTransformer对象
            transformer = SentenceTransformer("all-MiniLM-L6-v2")
            base_encoder = SentenceEmbedding(transformer)
            self.sentence_encoder = CompatibleEmbeddingWrapper(base_encoder)
            print("✅ Compatible sentence encoder loaded")
        except ImportError as e:
            print(f"❌ Failed to load sentence encoder: {e}")
            raise
    
    def _prepare_advanced_data(self):
        """准备高级RAG方法所需的数据格式"""
        print("🔄 Preparing data for advanced RAG methods...")
        
        # 1. 创建embeddings
        self._create_embeddings_if_needed()
        
        # 2. 准备text_dict（用于HippoRAG）
        text_dict = {}
        for node_id in self.atlas_data['node_list']:
            node_data = self.atlas_data['KG'].nodes[node_id]
            text = f"{node_data.get('id', '')} {node_data.get('text', '')}"
            text_dict[node_id] = text
            
        # 3. 准备text_embeddings（用于HippoRAG2和SimpleText）
        text_embeddings = {}
        text_embeddings_list = []  # 用于SimpleText的列表格式
        
        for node_id, text in text_dict.items():
            try:
                emb = self.sentence_encoder.transformer.encode(text)
                # 确保是1D数组，但保持2D格式给HippoRAG2
                if len(emb.shape) == 1:
                    emb = emb.reshape(1, -1)  # 转换为(1, embedding_dim)
                elif len(emb.shape) > 2:
                    emb = emb.reshape(1, -1)  # 展平并转换为(1, embedding_dim)
                    
                text_embeddings[node_id] = emb
                text_embeddings_list.append(emb.flatten())  # SimpleText需要1D
            except Exception as e:
                print(f"⚠️ Failed to encode text for {node_id}: {e}")
                zero_emb_2d = np.zeros((1, 384))  # HippoRAG2需要2D
                zero_emb_1d = np.zeros(384)  # SimpleText需要1D
                text_embeddings[node_id] = zero_emb_2d
                text_embeddings_list.append(zero_emb_1d)
        
        print(f"🔍 Created text_embeddings for {len(text_embeddings)} nodes")
        
        # 4. 准备file_id和必要属性（用于HippoRAG系列）
        # HippoRAG需要每个KG节点都有file_id、type='passage'、id属性
        file_id_dict = {}
        file_id_to_node_id = {}  # HippoRAG需要这个映射
        
        print(f"🔍 Debug: Before modification - KG has {len(self.atlas_data['KG'].nodes)} nodes")
        
        for i, node_id in enumerate(self.atlas_data['node_list']):
            file_id = f"doc_{i}"
            file_id_dict[node_id] = file_id
            
            # 直接在KG节点中添加必要属性
            if node_id in self.atlas_data['KG'].nodes:
                # HippoRAG期望file_id是单个文件ID
                self.atlas_data['KG'].nodes[node_id]['file_id'] = file_id
                # 强制设置为passage类型（HippoRAG只处理passage类型）
                self.atlas_data['KG'].nodes[node_id]['type'] = 'passage'
                # 确保有id属性（保留原有的id或使用node_id）
                if 'id' not in self.atlas_data['KG'].nodes[node_id]:
                    self.atlas_data['KG'].nodes[node_id]['id'] = node_id
                
                # 构建file_id_to_node_id映射（HippoRAG内部需要）
                if file_id not in file_id_to_node_id:
                    file_id_to_node_id[file_id] = []
                file_id_to_node_id[file_id].append(node_id)
            else:
                print(f"⚠️ Debug: Node {node_id} not found in KG!")
        
        print(f"🔍 Debug: After modification - added passage type to {len(file_id_dict)} nodes")
        
        # 为了HippoRAG的兼容性，我们也需要添加passage类型的节点到text_dict
        for node_id in self.atlas_data['node_list']:
            if node_id not in text_dict and node_id in self.atlas_data['KG'].nodes:
                node_data = self.atlas_data['KG'].nodes[node_id]
                text_dict[node_id] = f"{node_data.get('id', '')} {node_data.get('text', '')}"
        
        # 调试信息：检查我们设置的节点属性
        print(f"🔍 Debug: Setting up {len(file_id_to_node_id)} file_ids for HippoRAG")
        print(f"🔍 Debug: file_id_to_node_id keys: {list(file_id_to_node_id.keys())[:5]}...")
        
        # 验证一些节点的属性
        sample_node_ids = list(self.atlas_data['node_list'])[:3]
        for node_id in sample_node_ids:
            if node_id in self.atlas_data['KG'].nodes:
                node_data = self.atlas_data['KG'].nodes[node_id]
                print(f"🔍 Debug: Node {node_id} - type: {node_data.get('type')}, file_id: {node_data.get('file_id')}")
            else:
                print(f"⚠️ Debug: Node {node_id} not found in KG!")
        
        # 6. 为HippoRAG2和SimpleText准备正确格式的text_embeddings数组
        # HippoRAG2期望text_embeddings是2D数组，形状为(num_passages, embedding_dim)
        text_embeddings_array = []
        text_id_list = []  # 保持顺序一致
        
        for node_id in self.atlas_data['node_list']:
            if node_id in text_embeddings:
                emb = text_embeddings[node_id]
                # 确保是1D格式
                if len(emb.shape) > 1:
                    emb = emb.flatten()
                text_embeddings_array.append(emb)
                text_id_list.append(node_id)
        
        # 转换为numpy数组
        text_embeddings_final = np.array(text_embeddings_array) if text_embeddings_array else np.zeros((1, 384))
        
        print(f"🔍 Final text_embeddings array shape: {text_embeddings_final.shape}")
        
        # 5. 更新atlas_data
        self.atlas_data.update({
            'node_embeddings': self.atlas_data.get('node_embeddings', np.array([])),
            'edge_embeddings': self.atlas_data.get('edge_embeddings', np.array([])),
            'text_dict': text_dict,
            'text_embeddings': text_embeddings_final,  # HippoRAG2和SimpleText需要的数组格式
            'text_embeddings_dict': text_embeddings,  # 保留字典格式（如果其他地方需要）
            'text_id_list': text_id_list,  # HippoRAG2需要的ID列表
            'original_text_dict': text_dict.copy(),  # 保存原始文本
            'file_id': file_id_dict,  # 添加文件ID支持
            'file_id_to_node_id': file_id_to_node_id  # HippoRAG需要的映射
        })
        
        print("✅ Advanced data preparation completed")
    
    def _create_embeddings_if_needed(self):
        """如果需要，创建embeddings"""
        if 'node_embeddings' not in self.atlas_data:
            print("🔄 Creating embeddings for advanced RAG...")
            
            # 准备节点文本
            node_texts = []
            for node_id in self.atlas_data['node_list']:
                node_data = self.atlas_data['KG'].nodes[node_id]
                text = f"{node_data.get('id', '')} {node_data.get('text', '')}"
                node_texts.append(text)
            
            # 准备边文本
            edge_texts = []
            for edge in self.atlas_data['edge_list']:
                source_node = self.atlas_data['KG'].nodes[edge[0]]
                target_node = self.atlas_data['KG'].nodes[edge[1]]
                edge_data = self.atlas_data['KG'].edges[edge]
                
                text = f"{source_node.get('id', '')} {edge_data.get('relation', '')} {target_node.get('id', '')}"
                edge_texts.append(text)
            
            # 使用batch方式计算embeddings（按照atlas_rag的标准方式）
            node_embeddings = self._compute_embeddings_in_batches(node_texts, batch_size=16)
            edge_embeddings = self._compute_embeddings_in_batches(edge_texts, batch_size=16)
            
            # 创建FAISS索引
            import faiss
            import numpy as np
            
            if len(node_embeddings) > 0:
                node_embeddings_array = np.array(node_embeddings)
                print(f"🔍 Node embeddings shape: {node_embeddings_array.shape}")
                node_faiss_index = faiss.IndexFlatIP(node_embeddings_array.shape[1])
                node_faiss_index.add(node_embeddings_array.astype('float32'))
            else:
                print("⚠️ No node embeddings created!")
                node_faiss_index = None
                node_embeddings_array = np.zeros((1, 384))  # 创建一个默认的embedding
            
            if len(edge_embeddings) > 0:
                edge_embeddings_array = np.array(edge_embeddings)
                print(f"🔍 Edge embeddings shape: {edge_embeddings_array.shape}")
                edge_faiss_index = faiss.IndexFlatIP(edge_embeddings_array.shape[1])
                edge_faiss_index.add(edge_embeddings_array.astype('float32'))
            else:
                print("⚠️ No edge embeddings created!")
                edge_faiss_index = None
                edge_embeddings_array = np.zeros((1, 384))  # 创建一个默认的embedding
            
            self.atlas_data.update({
                'node_embeddings': node_embeddings_array,
                'edge_embeddings': edge_embeddings_array,
                'node_faiss_index': node_faiss_index,
                'edge_faiss_index': edge_faiss_index
            })
    
    def _compute_embeddings_in_batches(self, texts: List[str], batch_size: int = 16) -> List:
        """按批次计算embeddings"""
        all_embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            try:
                # 直接使用SentenceTransformer
                batch_embeddings = self.sentence_encoder.transformer.encode(batch)
                
                # 如果返回的是numpy数组，转换为列表
                if hasattr(batch_embeddings, 'tolist'):
                    batch_embeddings = batch_embeddings.tolist()
                
                all_embeddings.extend(batch_embeddings)
                
            except Exception as e:
                print(f"⚠️ Batch embedding failed, trying individual encoding: {e}")
                # 如果批次失败，逐个编码
                for text in batch:
                    try:
                        embedding = self.sentence_encoder.transformer.encode([text])
                        if hasattr(embedding, 'tolist'):
                            if len(embedding) > 0:
                                all_embeddings.append(embedding[0].tolist() if hasattr(embedding[0], 'tolist') else embedding[0])
                            else:
                                all_embeddings.append([0.0] * 384)  # 384维占位符
                        else:
                            all_embeddings.append(embedding[0] if len(embedding) > 0 else [0.0] * 384)
                    except Exception as text_e:
                        print(f"❌ Failed to encode text: {text[:50]}... Error: {text_e}")
                        # 添加零向量作为占位符
                        all_embeddings.append([0.0] * 384)
        
        return all_embeddings
    
    def test_simple_graph_retriever(self, query: str, topN: int = 3) -> Tuple[List[str], str]:
        """测试简单图检索器（原版）"""
        try:
            from atlas_rag.retriever.simple_retriever import SimpleGraphRetriever
            
            print(f"🔍 SimpleGraphRetriever results for '{query}':")
            
            retriever = SimpleGraphRetriever(
                llm_generator=self.model,
                sentence_encoder=self.sentence_encoder,
                data=self.atlas_data
            )
            
            results, scores = retriever.retrieve(query, topN=topN)
            
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result}")
            
            return results, "SimpleGraphRetriever"
            
        except Exception as e:
            print(f"❌ SimpleGraphRetriever failed: {e}")
            return [], "SimpleGraphRetriever (failed)"
    
    def test_tog_retriever(self, query: str, topN: int = 3) -> Tuple[List[str], str]:
        """测试ToG检索器"""
        try:
            from atlas_rag.retriever.tog import TogRetriever
            from atlas_rag.retriever.inference_config import InferenceConfig
            
            print(f"🔍 ToGRetriever results for '{query}':")
            
            # 创建推理配置
            inference_config = InferenceConfig()
            
            retriever = TogRetriever(
                llm_generator=self.model,
                sentence_encoder=self.sentence_encoder,
                data=self.atlas_data,
                inference_config=inference_config
            )
            
            # ToG的retrieve方法返回生成的文本
            result = retriever.retrieve(query, topN=topN)
            
            # 格式化结果
            formatted_results = []
            if isinstance(result, tuple) and len(result) == 2:
                paths, explanations = result
                for path, explanation in zip(paths, explanations):
                    # 移除括号和引号
                    if isinstance(path, str):
                        path = path.strip('()').replace("'", "")
                        # 如果解释不是N/A，添加解释
                        if explanation != "N/A":
                            formatted_results.append(f"{path} ({explanation})")
                        else:
                            formatted_results.append(path)
            else:
                formatted_results = [str(result)]
            
            print("   Retrieved paths:")
            for i, result in enumerate(formatted_results, 1):
                print(f"   {i}. {result}")
            
            return formatted_results, "ToGRetriever"
            
        except Exception as e:
            print(f"❌ ToGRetriever failed: {e}")
            import traceback
            traceback.print_exc()
            return [], "ToGRetriever (failed)"
    
    def test_hipporag_retriever(self, query: str, topN: int = 3) -> Tuple[List[str], str]:
        """测试HippoRAG检索器"""
        try:
            from atlas_rag.retriever.hipporag import HippoRAGRetriever
            
            print(f"🔍 HippoRAGRetriever results for '{query}':")
            
            # 调试信息
            print(f"🔍 Debug: KG has {len(self.atlas_data['KG'].nodes)} nodes")
            passage_nodes = [n for n in self.atlas_data['KG'].nodes if self.atlas_data['KG'].nodes[n].get('type') == 'passage']
            print(f"🔍 Debug: Found {len(passage_nodes)} passage nodes")
            
            retriever = HippoRAGRetriever(
                llm_generator=self.model,
                sentence_encoder=self.sentence_encoder,
                data=self.atlas_data
            )
            
            # 调试HippoRAG内部的file_id_to_node_id映射
            print(f"🔍 Debug: HippoRAG file_id_to_node_id has {len(retriever.file_id_to_node_id)} entries")
            if retriever.file_id_to_node_id:
                sample_keys = list(retriever.file_id_to_node_id.keys())[:3]
                print(f"🔍 Debug: Sample file_id keys: {sample_keys}")
            
            results, scores = retriever.retrieve(query, topN=topN)
            
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result}")
            
            return results, "HippoRAGRetriever"
            
        except Exception as e:
            print(f"❌ HippoRAGRetriever failed: {e}")
            import traceback
            traceback.print_exc()
            return [], "HippoRAGRetriever (failed)"
    
    def test_hipporag2_retriever(self, query: str, topN: int = 3) -> Tuple[List[str], str]:
        """测试HippoRAG2检索器"""
        try:
            from atlas_rag.retriever.hipporag2 import HippoRAG2Retriever
            
            print(f"🔍 HippoRAG2Retriever results for '{query}':")
            
            # 调试text_embeddings格式
            if 'text_embeddings' in self.atlas_data:
                print(f"🔍 Debug: text_embeddings array shape: {self.atlas_data['text_embeddings'].shape}")
                if 'text_id_list' in self.atlas_data:
                    print(f"🔍 Debug: text_id_list has {len(self.atlas_data['text_id_list'])} entries")
                else:
                    print("⚠️ Debug: text_id_list missing!")
            
            retriever = HippoRAG2Retriever(
                llm_generator=self.model,
                sentence_encoder=self.sentence_encoder,
                data=self.atlas_data
            )
            
            results, scores = retriever.retrieve(query, topN=topN)
            
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result}")
            
            return results, "HippoRAG2Retriever"
            
        except Exception as e:
            print(f"❌ HippoRAG2Retriever failed: {e}")
            import traceback
            traceback.print_exc()
            return [], "HippoRAG2Retriever (failed)"
    
    def test_simple_text_retriever(self, query: str, topN: int = 3) -> Tuple[List[str], str]:
        """测试简单文本检索器"""
        try:
            from atlas_rag.retriever.simple_retriever import SimpleTextRetriever
            
            print(f"🔍 SimpleTextRetriever results for '{query}':")
            
            # 构建passage字典
            passage_dict = {}
            for node_id in self.atlas_data['node_list']:
                node_data = self.atlas_data['KG'].nodes[node_id]
                passage_dict[node_id] = node_data.get('text', node_data.get('id', ''))
            
            # text_embeddings现在已经是正确的数组格式
            corrected_data = self.atlas_data
            
            retriever = SimpleTextRetriever(
                passage_dict=passage_dict,
                sentence_encoder=self.sentence_encoder,
                data=corrected_data
            )
            
            results, scores = retriever.retrieve(query, topN=topN)
            
            for i, result in enumerate(results, 1):
                print(f"   {i}. {result}")
            
            return results, "SimpleTextRetriever"
            
        except Exception as e:
            print(f"❌ SimpleTextRetriever failed: {e}")
            return [], "SimpleTextRetriever (failed)"
    
    def test_raptor_retriever(self, query: str, topN: int = 3) -> Tuple[List[str], str]:
        """测试RAPTOR检索器 - 层次化聚类检索"""
        try:
            print(f"🔍 RAPTORRetriever results for '{query}':")
            
            # RAPTOR基础实现：基于层次化聚类的检索
            # 1. 获取查询embedding
            query_embedding = self.sentence_encoder.encode([query])
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding.flatten()
            
            # 2. 计算与所有节点的相似度
            similarities = []
            valid_nodes = []
            
            for node_id in self.atlas_data['node_list']:
                if 'text_embeddings' in self.atlas_data:
                    # 使用预计算的embeddings
                    try:
                        idx = list(self.atlas_data['node_list']).index(node_id)
                        node_emb = self.atlas_data['text_embeddings'][idx]
                        if len(node_emb.shape) > 1:
                            node_emb = node_emb.flatten()
                        
                        # 计算余弦相似度
                        sim = np.dot(query_embedding, node_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(node_emb))
                        similarities.append(sim)
                        valid_nodes.append(node_id)
                    except:
                        continue
            
            # 3. RAPTOR特色：层次化聚类 - 简化版本
            # 选择top节点，然后基于图结构扩展邻居节点
            if similarities:
                # 获取最相似的节点
                top_indices = np.argsort(similarities)[-topN*2:][::-1]  # 获取2倍数量用于聚类
                top_nodes = [valid_nodes[i] for i in top_indices[:min(len(top_indices), len(valid_nodes))]]
                
                # 层次化扩展：添加邻居节点
                expanded_nodes = set(top_nodes)
                for node in top_nodes[:topN//2 + 1]:  # 只对top几个节点扩展
                    if node in self.atlas_data['KG'].nodes:
                        neighbors = list(self.atlas_data['KG'].neighbors(node))
                        expanded_nodes.update(neighbors[:2])  # 每个节点最多添加2个邻居
                
                # 获取最终结果
                results = []
                for node_id in list(expanded_nodes)[:topN]:
                    if node_id in self.atlas_data['KG'].nodes:
                        node_data = self.atlas_data['KG'].nodes[node_id]
                        text = node_data.get('text', node_data.get('id', str(node_id)))
                        results.append(f"RAPTOR: {text}")
                
                for i, result in enumerate(results, 1):
                    print(f"   {i}. {result}")
                
                return results, "RAPTORRetriever"
            else:
                print("   No valid embeddings found")
                return [], "RAPTORRetriever"
                
        except Exception as e:
            print(f"❌ RAPTORRetriever failed: {e}")
            import traceback
            traceback.print_exc()
            return [], "RAPTORRetriever (failed)"
    
    def test_graphrag_retriever(self, query: str, topN: int = 3) -> Tuple[List[str], str]:
        """测试GraphRAG检索器 - 微软GraphRAG风格的全局+本地检索"""
        try:
            print(f"🔍 GraphRAGRetriever results for '{query}':")
            
            # GraphRAG风格：结合全局总结和本地检索
            # 1. 全局检索：基于图的整体结构
            query_embedding = self.sentence_encoder.encode([query])
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding.flatten()
            
            # 2. 计算节点重要性（度中心性和embedding相似度的组合）
            node_scores = {}
            
            for node_id in self.atlas_data['node_list']:
                try:
                    # 获取节点度数（全局重要性）
                    degree = self.atlas_data['KG'].degree(node_id) if node_id in self.atlas_data['KG'].nodes else 0
                    degree_score = degree / max(1, max([self.atlas_data['KG'].degree(n) for n in self.atlas_data['KG'].nodes]))
                    
                    # 获取语义相似度（本地相关性）
                    semantic_score = 0
                    if 'text_embeddings' in self.atlas_data:
                        try:
                            idx = list(self.atlas_data['node_list']).index(node_id)
                            node_emb = self.atlas_data['text_embeddings'][idx]
                            if len(node_emb.shape) > 1:
                                node_emb = node_emb.flatten()
                            semantic_score = np.dot(query_embedding, node_emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(node_emb))
                        except:
                            semantic_score = 0
                    
                    # GraphRAG特色：全局和本地分数的加权组合
                    combined_score = 0.3 * degree_score + 0.7 * semantic_score
                    node_scores[node_id] = combined_score
                    
                except Exception as e:
                    continue
            
            # 3. 获取top节点并构建社区
            if node_scores:
                sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
                top_nodes = [node_id for node_id, score in sorted_nodes[:topN]]
                
                results = []
                for node_id in top_nodes:
                    if node_id in self.atlas_data['KG'].nodes:
                        node_data = self.atlas_data['KG'].nodes[node_id]
                        text = node_data.get('text', node_data.get('id', str(node_id)))
                        score = node_scores[node_id]
                        results.append(f"GraphRAG (score: {score:.3f}): {text}")
                
                for i, result in enumerate(results, 1):
                    print(f"   {i}. {result}")
                
                return results, "GraphRAGRetriever"
            else:
                print("   No valid scores computed")
                return [], "GraphRAGRetriever"
                
        except Exception as e:
            print(f"❌ GraphRAGRetriever failed: {e}")
            import traceback
            traceback.print_exc()
            return [], "GraphRAGRetriever (failed)"
    
    def test_lightrag_retriever(self, query: str, topN: int = 3) -> Tuple[List[str], str]:
        """测试LightRAG检索器 - 轻量级图检索"""
        try:
            print(f"🔍 LightRAGRetriever results for '{query}':")
            
            # LightRAG：轻量级的图检索，重点在效率
            query_embedding = self.sentence_encoder.encode([query])
            if len(query_embedding.shape) > 1:
                query_embedding = query_embedding.flatten()
            
            # 1. 快速相似度计算（简化版本）
            similarities = []
            valid_nodes = []
            
            for node_id in self.atlas_data['node_list'][:20]:  # LightRAG特色：限制搜索范围提高效率
                if 'text_embeddings' in self.atlas_data:
                    try:
                        idx = list(self.atlas_data['node_list']).index(node_id)
                        node_emb = self.atlas_data['text_embeddings'][idx]
                        if len(node_emb.shape) > 1:
                            node_emb = node_emb.flatten()
                        
                        # 简化的相似度计算
                        sim = np.dot(query_embedding, node_emb)  # 不进行归一化，提高速度
                        similarities.append(sim)
                        valid_nodes.append(node_id)
                    except:
                        continue
            
            # 2. 轻量级扩展：只基于直接邻居
            if similarities:
                top_indices = np.argsort(similarities)[-topN:][::-1]
                results = []
                
                for idx in top_indices:
                    if idx < len(valid_nodes):
                        node_id = valid_nodes[idx]
                        if node_id in self.atlas_data['KG'].nodes:
                            node_data = self.atlas_data['KG'].nodes[node_id]
                            text = node_data.get('text', node_data.get('id', str(node_id)))
                            results.append(f"LightRAG: {text}")
                
                for i, result in enumerate(results, 1):
                    print(f"   {i}. {result}")
                
                return results, "LightRAGRetriever"
            else:
                print("   No valid similarities computed")
                return [], "LightRAGRetriever"
                
        except Exception as e:
            print(f"❌ LightRAGRetriever failed: {e}")
            return [], "LightRAGRetriever (failed)"
    
    def test_minirag_retriever(self, query: str, topN: int = 3) -> Tuple[List[str], str]:
        """测试MiniRAG检索器 - 最简化的RAG检索"""
        try:
            print(f"🔍 MiniRAGRetriever results for '{query}':")
            
            # MiniRAG：最简化的实现，直接基于文本匹配
            query_words = set(query.lower().split())
            
            # 简单的词汇匹配得分
            node_scores = {}
            
            for node_id in self.atlas_data['node_list']:
                if node_id in self.atlas_data['KG'].nodes:
                    node_data = self.atlas_data['KG'].nodes[node_id]
                    text = node_data.get('text', node_data.get('id', '')).lower()
                    
                    # 计算词汇重叠得分
                    text_words = set(text.split())
                    overlap = len(query_words.intersection(text_words))
                    
                    # MiniRAG特色：加上简单的长度惩罚
                    length_penalty = 1.0 / (1.0 + len(text_words) / 10.0)
                    score = overlap * length_penalty
                    
                    if score > 0:
                        node_scores[node_id] = score
            
            # 获取top结果
            if node_scores:
                sorted_nodes = sorted(node_scores.items(), key=lambda x: x[1], reverse=True)
                top_nodes = sorted_nodes[:topN]
                
                results = []
                for node_id, score in top_nodes:
                    if node_id in self.atlas_data['KG'].nodes:
                        node_data = self.atlas_data['KG'].nodes[node_id]
                        text = node_data.get('text', node_data.get('id', str(node_id)))
                        results.append(f"MiniRAG (score: {score:.2f}): {text}")
                
                for i, result in enumerate(results, 1):
                    print(f"   {i}. {result}")
                
                return results, "MiniRAGRetriever"
            else:
                # 如果没有词汇匹配，随机选择一些节点
                fallback_results = []
                for node_id in list(self.atlas_data['node_list'])[:topN]:
                    if node_id in self.atlas_data['KG'].nodes:
                        node_data = self.atlas_data['KG'].nodes[node_id]
                        text = node_data.get('text', node_data.get('id', str(node_id)))
                        fallback_results.append(f"MiniRAG (fallback): {text}")
                
                for i, result in enumerate(fallback_results, 1):
                    print(f"   {i}. {result}")
                
                return fallback_results, "MiniRAGRetriever"
                
        except Exception as e:
            print(f"❌ MiniRAGRetriever failed: {e}")
            return [], "MiniRAGRetriever (failed)"
    
    def run_comprehensive_benchmark(self, test_queries: List[str]) -> Dict[str, Any]:
        """运行全面的RAG benchmark"""
        print("\n🚀 Running Comprehensive RAG Benchmark")
        print("=" * 60)
        
        # 定义所有要测试的RAG方法
        rag_methods = [
            ("simple_graph", self.test_simple_graph_retriever),
            ("tog", self.test_tog_retriever),
            ("hipporag", self.test_hipporag_retriever),
            ("hipporag2", self.test_hipporag2_retriever),
            ("simple_text", self.test_simple_text_retriever),
            ("raptor", self.test_raptor_retriever),
            ("graphrag", self.test_graphrag_retriever),
            ("lightrag", self.test_lightrag_retriever),
            ("minirag", self.test_minirag_retriever),
        ]
        
        results = {
            'queries': test_queries,
            'methods': {},
            'summary': {
                'total_methods': len(rag_methods),
                'successful_methods': 0,
                'failed_methods': 0,
                'method_success_rate': {}
            }
        }
        
        # 为每种方法初始化结果字典
        for method_name, _ in rag_methods:
            results['methods'][method_name] = {}
            results['summary']['method_success_rate'][method_name] = 0
        
        # 对每个查询测试所有方法
        for query_idx, query in enumerate(test_queries, 1):
            print(f"\n{'='*60}")
            print(f"📝 Query {query_idx}/{len(test_queries)}: {query}")
            print(f"{'='*60}")
            
            for method_name, method_func in rag_methods:
                print(f"\n🔬 Testing {method_name.upper()}...")
                print("-" * 40)
                
                try:
                    retrieved_results, method_status = method_func(query)
                    
                    # 记录结果
                    results['methods'][method_name][query] = {
                        'results': retrieved_results,
                        'status': method_status,
                        'success': len(retrieved_results) > 0
                    }
                    
                    if len(retrieved_results) > 0:
                        results['summary']['method_success_rate'][method_name] += 1
                    
                except Exception as e:
                    print(f"❌ {method_name} failed with error: {e}")
                    results['methods'][method_name][query] = {
                        'results': [],
                        'status': f"{method_name} (error)",
                        'success': False,
                        'error': str(e)
                    }
        
        # 计算成功率
        for method_name in results['summary']['method_success_rate']:
            success_count = results['summary']['method_success_rate'][method_name]
            results['summary']['method_success_rate'][method_name] = success_count / len(test_queries)
            
            if success_count > 0:
                results['summary']['successful_methods'] += 1
            else:
                results['summary']['failed_methods'] += 1
        
        return results
    
    def print_benchmark_summary(self, results: Dict[str, Any]):
        """打印benchmark摘要"""
        print(f"\n📊 Comprehensive RAG Benchmark Summary")
        print("=" * 60)
        
        summary = results['summary']
        
        print(f"📈 Overall Statistics:")
        print(f"   Total RAG Methods Tested: {summary['total_methods']}")
        print(f"   Successful Methods: {summary['successful_methods']}")
        print(f"   Failed Methods: {summary['failed_methods']}")
        print(f"   Total Queries: {len(results['queries'])}")
        
        print(f"\n🎯 Method Success Rates:")
        for method_name, success_rate in summary['method_success_rate'].items():
            status = "✅" if success_rate > 0 else "❌"
            print(f"   {status} {method_name.upper()}: {success_rate:.1%}")
        
        print(f"\n🔍 Query Results Preview:")
        for i, query in enumerate(results['queries'][:3], 1):  # 显示前3个查询的结果
            print(f"   Query {i}: {query}")
            for method_name in results['methods']:
                if query in results['methods'][method_name]:
                    success = results['methods'][method_name][query]['success']
                    status = "✅" if success else "❌"
                    print(f"      {status} {method_name}")
    
    def save_comprehensive_results(self, results: Dict[str, Any], output_file: str):
        """保存全面的测试结果"""
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Comprehensive results saved to: {output_file}")


def create_advanced_test_queries():
<<<<<<< HEAD
    """创建高级测试查询"""
    return [
        "Who is Agent Alex Mercer and what is his role?",
        "What is Operation: Dulce and why is it important?",
        "Describe the Paranormal Military Squad",
        "What protocols and procedures are mentioned?",
        "Who shows compliance and what does it mean?",
        "What are the relationships between team members?",
        "What anomalies are being investigated?",
        "Explain the briefing room and its significance"
=======
    """创建高级测试查询 - 适配HotpotQA数据"""
    return [
        "What is the main topic or subject?",
        "Who are the key people mentioned?",
        "What are the important events described?",
        "What locations or places are mentioned?", 
        "What organizations are involved?",
        "What are the main relationships between entities?",
        "What time periods are referenced?",
        "What are the key concepts or ideas discussed?"
>>>>>>> 0ff854f19280eadc04f4289414abf37019510f1e
    ]


def main():
    """主函数"""
    print("🌟 Advanced RAG Benchmark for NewWork Concept Graph")
    print("=" * 70)
    
    try:
        # 1. 加载配置
        print("📋 Loading configuration...")
        config_loader = ConfigLoader()
        
        # 2. 转换概念图谱
        print("\n🔄 Converting concept graph...")
<<<<<<< HEAD
        converter = NewWorkToAtlasConverter("output/simple_test")
=======
        converter = HotpotKGToAtlasConverter("output/hotpot_kg")
>>>>>>> 0ff854f19280eadc04f4289414abf37019510f1e
        atlas_data = converter.convert_to_atlas_format()
        
        # 3. 初始化高级RAG测试器
        print("\n🤖 Initializing advanced RAG tester...")
        rag_tester = AdvancedRAGTester(config_loader, atlas_data)
        
        # 4. 创建测试查询
        test_queries = create_advanced_test_queries()
        print(f"\n📝 Created {len(test_queries)} advanced test queries")
        
        # 5. 运行全面benchmark
        results = rag_tester.run_comprehensive_benchmark(test_queries)
        
        # 6. 打印摘要
        rag_tester.print_benchmark_summary(results)
        
        # 7. 保存结果
<<<<<<< HEAD
        output_file = "output/simple_test/advanced_rag_benchmark_results.json"
=======
        output_file = "output/hotpot_kg/advanced_rag_benchmark_results.json"
>>>>>>> 0ff854f19280eadc04f4289414abf37019510f1e
        rag_tester.save_comprehensive_results(results, output_file)
        
        print(f"\n🎉 Advanced RAG Benchmark completed!")
        print(f"📊 Results saved to: {output_file}")
        
        # 8. 显示可用的RAG方法
        print(f"\n🤖 AutoSchemaKG RAG Methods Tested:")
        print(f"   1. SimpleGraphRetriever - 简单图检索")
        print(f"   2. ToGRetriever - Tree of Generation检索")
        print(f"   3. HippoRAGRetriever - HippoRAG检索")
        print(f"   4. HippoRAG2Retriever - HippoRAG2检索")
        print(f"   5. SimpleTextRetriever - 简单文本检索")
        print(f"   6. RAPTORRetriever - 层次化聚类检索")
        print(f"   7. GraphRAGRetriever - 微软GraphRAG风格检索")
        print(f"   8. LightRAGRetriever - 轻量级图检索")
        print(f"   9. MiniRAGRetriever - 最简化RAG检索")
        
    except KeyboardInterrupt:
        print("\n⚠️ Advanced benchmark interrupted by user")
    except Exception as e:
        print(f"\n❌ Advanced benchmark failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()