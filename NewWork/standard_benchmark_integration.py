#!/usr/bin/env python3
"""
Standard AutoSchemaKG Benchmark Integration for NewWork
将NewWork概念图谱集成到AutoSchemaKG标准benchmark系统中进行科学评估
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Any
import logging
from datetime import datetime

# 添加路径
sys.path.append('..')
sys.path.append('.')

from config_loader import ConfigLoader, create_model_client
from rag_benchmark import NewWorkToAtlasConverter

# 导入AutoSchemaKG标准组件
from atlas_rag.evaluation.benchmark import RAGBenchmark, BenchMarkConfig
from atlas_rag.evaluation.evaluation import QAJudger
from atlas_rag.vectorstore.embedding_model import SentenceEmbedding
from atlas_rag.retriever.simple_retriever import SimpleGraphRetriever, SimpleTextRetriever
from atlas_rag.retriever.hipporag import HippoRAGRetriever
from atlas_rag.retriever.hipporag2 import HippoRAG2Retriever
from atlas_rag.retriever.tog import TogRetriever

from sentence_transformers import SentenceTransformer
import faiss


class NewWorkStandardBenchmark:
    """使用AutoSchemaKG标准benchmark评估NewWork KG"""
    
    def __init__(self, newwork_output_path: str = "output/simple_test"):
        self.newwork_output_path = newwork_output_path
        self.config_loader = ConfigLoader()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """设置日志记录器"""
        logger = logging.getLogger('NewWorkBenchmark')
        logger.setLevel(logging.INFO)
        
        # 避免重复添加handler
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def prepare_newwork_data(self) -> Dict[str, Any]:
        """准备NewWork数据并转换为Atlas格式"""
        print("📊 准备NewWork概念图谱数据...")
        
        # 1. 转换NewWork数据为Atlas格式
        converter = NewWorkToAtlasConverter(self.newwork_output_path)
        atlas_data = converter.convert_to_atlas_format()
        
        print(f"✅ 转换完成: {len(atlas_data.get('node_list', []))} 节点, {len(atlas_data.get('edge_list', []))} 边")
        
        # 2. 设置sentence encoder
        transformer = SentenceTransformer("all-MiniLM-L6-v2")
        sentence_encoder = SentenceEmbedding(transformer)
        
        # 3. 创建embeddings和索引
        atlas_data = self._create_embeddings_and_indices(atlas_data, sentence_encoder)
        
        return atlas_data, sentence_encoder
    
    def _create_embeddings_and_indices(self, atlas_data: Dict[str, Any], sentence_encoder) -> Dict[str, Any]:
        """创建embeddings和FAISS索引"""
        print("🔄 创建embeddings和索引...")
        
        try:
            # 1. 创建node embeddings
            node_texts = []
            for node_id in atlas_data['node_list']:
                if node_id in atlas_data['KG'].nodes:
                    node_data = atlas_data['KG'].nodes[node_id]
                    text = node_data.get('text', node_data.get('id', str(node_id)))
                    node_texts.append(text)
                else:
                    node_texts.append(str(node_id))
            
            if node_texts:
                node_embeddings = sentence_encoder.encode(node_texts)
                if len(node_embeddings.shape) == 1:
                    node_embeddings = node_embeddings.reshape(1, -1)
            else:
                node_embeddings = np.zeros((1, 384))
            
            # 2. 创建edge embeddings
            edge_texts = []
            for edge in atlas_data['edge_list']:
                if len(edge) >= 2:
                    source_data = atlas_data['KG'].nodes.get(edge[0], {})
                    target_data = atlas_data['KG'].nodes.get(edge[1], {})
                    edge_data = atlas_data['KG'].edges.get(edge, {})
                    
                    source_text = source_data.get('text', source_data.get('id', str(edge[0])))
                    target_text = target_data.get('text', target_data.get('id', str(edge[1])))
                    relation = edge_data.get('relation', 'related_to')
                    
                    edge_text = f"{source_text} {relation} {target_text}"
                    edge_texts.append(edge_text)
                else:
                    edge_texts.append("empty_edge")
            
            if edge_texts:
                edge_embeddings = sentence_encoder.encode(edge_texts)
                if len(edge_embeddings.shape) == 1:
                    edge_embeddings = edge_embeddings.reshape(1, -1)
            else:
                edge_embeddings = np.zeros((1, 384))
            
            # 3. 创建FAISS索引
            # Node索引
            if node_embeddings.shape[0] > 0 and node_embeddings.shape[1] > 0:
                node_index = faiss.IndexFlatIP(node_embeddings.shape[1])
                # 标准化embeddings
                normalized_node_emb = node_embeddings / np.linalg.norm(node_embeddings, axis=1, keepdims=True)
                node_index.add(normalized_node_emb.astype('float32'))
            else:
                node_index = faiss.IndexFlatIP(384)
            
            # Edge索引
            if edge_embeddings.shape[0] > 0 and edge_embeddings.shape[1] > 0:
                edge_index = faiss.IndexFlatIP(edge_embeddings.shape[1])
                # 标准化embeddings
                normalized_edge_emb = edge_embeddings / np.linalg.norm(edge_embeddings, axis=1, keepdims=True)
                edge_index.add(normalized_edge_emb.astype('float32'))
            else:
                edge_index = faiss.IndexFlatIP(384)
            
            # 4. 创建text_dict和其他必需数据
            text_dict = {}
            for node_id in atlas_data['node_list']:
                if node_id in atlas_data['KG'].nodes:
                    node_data = atlas_data['KG'].nodes[node_id]
                    text_dict[node_id] = node_data.get('text', node_data.get('id', str(node_id)))
                else:
                    text_dict[node_id] = str(node_id)
            
            # 5. 更新atlas_data
            atlas_data.update({
                'node_embeddings': node_embeddings,
                'edge_embeddings': edge_embeddings,
                'node_faiss_index': node_index,
                'edge_faiss_index': edge_index,
                'text_dict': text_dict,
                'text_embeddings': node_embeddings,  # 用于某些retriever
            })
            
            print(f"✅ Embeddings创建完成: 节点{node_embeddings.shape}, 边{edge_embeddings.shape}")
            return atlas_data
            
        except Exception as e:
            print(f"❌ 创建embeddings失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def setup_retrievers(self, atlas_data: Dict[str, Any], sentence_encoder) -> List:
        """设置要测试的retriever列表"""
        print("🔧 设置Retriever列表...")
        
        retrievers = []
        llm_generator = create_model_client(self.config_loader)
        
        try:
            # 1. SimpleGraphRetriever
            simple_graph = SimpleGraphRetriever(
                llm_generator=llm_generator,
                sentence_encoder=sentence_encoder,
                data=atlas_data
            )
            retrievers.append(simple_graph)
            print("✅ SimpleGraphRetriever 已添加")
            
            # 2. SimpleTextRetriever
            passage_dict = atlas_data.get('text_dict', {})
            simple_text = SimpleTextRetriever(
                passage_dict=passage_dict,
                sentence_encoder=sentence_encoder,
                data=atlas_data
            )
            retrievers.append(simple_text)
            print("✅ SimpleTextRetriever 已添加")
            
            # 3. TogRetriever
            try:
                tog_retriever = TogRetriever(
                    llm_generator=llm_generator,
                    sentence_encoder=sentence_encoder,
                    data=atlas_data
                )
                retrievers.append(tog_retriever)
                print("✅ TogRetriever 已添加")
            except Exception as e:
                print(f"⚠️ TogRetriever 创建失败: {e}")
            
            # 4. HippoRAGRetriever (如果支持)
            try:
                # 为HippoRAG准备特殊数据格式
                hippo_data = self._prepare_hipporag_data(atlas_data)
                hipporag = HippoRAGRetriever(
                    llm_generator=llm_generator,
                    sentence_encoder=sentence_encoder,
                    data=hippo_data,
                    logger=self.logger
                )
                retrievers.append(hipporag)
                print("✅ HippoRAGRetriever 已添加")
            except Exception as e:
                print(f"⚠️ HippoRAGRetriever 创建失败: {e}")
            
            # 5. HippoRAG2Retriever (如果支持)
            try:
                hipporag2 = HippoRAG2Retriever(
                    llm_generator=llm_generator,
                    sentence_encoder=sentence_encoder,
                    data=atlas_data,
                    logger=self.logger
                )
                retrievers.append(hipporag2)
                print("✅ HippoRAG2Retriever 已添加")
            except Exception as e:
                print(f"⚠️ HippoRAG2Retriever 创建失败: {e}")
            
        except Exception as e:
            print(f"❌ 设置Retriever失败: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"📋 总共设置了 {len(retrievers)} 个Retriever")
        return retrievers
    
    def _prepare_hipporag_data(self, atlas_data: Dict[str, Any]) -> Dict[str, Any]:
        """为HippoRAG准备特殊的数据格式"""
        hippo_data = atlas_data.copy()
        
        # 确保每个节点都有passage类型和file_id
        for node_id in atlas_data['node_list']:
            if node_id in hippo_data['KG'].nodes:
                hippo_data['KG'].nodes[node_id]['type'] = 'passage'
                hippo_data['KG'].nodes[node_id]['file_id'] = f"file_{node_id}"
        
        return hippo_data
    
    def run_standard_benchmark(self, dataset_name: str = "musique", num_samples: int = 50):
        """运行标准benchmark测试"""
        print(f"🚀 开始标准benchmark测试 (数据集: {dataset_name}, 样本数: {num_samples})")
        print("=" * 80)
        
        try:
            # 1. 准备NewWork数据
            atlas_data, sentence_encoder = self.prepare_newwork_data()
            
            # 2. 设置retrievers
            retrievers = self.setup_retrievers(atlas_data, sentence_encoder)
            
            if not retrievers:
                print("❌ 没有可用的retriever，终止测试")
                return
            
            # 3. 配置标准benchmark
            benchmark_config = BenchMarkConfig(
                dataset_name=dataset_name,
                question_file=f"../benchmark_data/{dataset_name}.json",
                include_concept=True,
                include_events=True,
                reader_model_name=self.config_loader.config_data.get('models', {}).get('qwen_235b', 'qwen'),
                encoder_model_name="all-MiniLM-L6-v2",
                number_of_samples=num_samples,
                react_max_iterations=3
            )
            
            # 4. 创建LLM生成器
            llm_generator = create_model_client(self.config_loader)
            
            # 5. 运行标准benchmark
            print(f"\n📊 开始运行标准RAG Benchmark...")
            print(f"📋 数据集: {dataset_name}")
            print(f"📋 样本数: {num_samples}")
            print(f"📋 Retriever数量: {len(retrievers)}")
            print(f"📋 Retriever列表: {[r.__class__.__name__ for r in retrievers]}")
            
            benchmark = RAGBenchmark(config=benchmark_config, logger=self.logger)
            benchmark.run(retrievers, llm_generator)
            
            print(f"\n✅ 标准benchmark测试完成!")
            print(f"📊 结果已保存到: ./result/{dataset_name}/")
            
            # 6. 打印摘要信息
            self._print_test_summary(dataset_name, retrievers, num_samples)
            
        except FileNotFoundError as e:
            print(f"❌ 数据文件未找到: {e}")
            print(f"💡 请确保 ../benchmark_data/{dataset_name}.json 文件存在")
            print(f"💡 可用的数据集文件:")
            benchmark_data_path = Path("../benchmark_data")
            if benchmark_data_path.exists():
                for file in benchmark_data_path.glob("*.json"):
                    print(f"   - {file.name}")
        except Exception as e:
            print(f"❌ Benchmark运行失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _print_test_summary(self, dataset_name: str, retrievers: List, num_samples: int):
        """打印测试摘要"""
        print("\n" + "="*60)
        print("📋 测试摘要")
        print("="*60)
        print(f"🗃️  数据集: {dataset_name}")
        print(f"📊 样本数量: {num_samples}")
        print(f"🤖 知识图谱: NewWork概念图谱")
        print(f"📍 图谱路径: {self.newwork_output_path}")
        print(f"🔧 测试的Retriever:")
        for i, retriever in enumerate(retrievers, 1):
            print(f"   {i}. {retriever.__class__.__name__}")
        print(f"📊 评估指标: EM, F1, Recall@2, Recall@5")
        print(f"💾 结果保存位置: ./result/{dataset_name}/")
        print("="*60)
    
    def run_quick_test(self, num_samples: int = 10):
        """运行快速测试（少量样本）"""
        print(f"⚡ 运行快速测试 (样本数: {num_samples})")
        self.run_standard_benchmark(dataset_name="musique", num_samples=num_samples)
    
    def run_full_evaluation(self):
        """运行完整评估"""
        print("🏆 运行完整评估")
        
        # 可以测试多个数据集
        datasets = [
            ("musique", 200),
            # ("hotpotqa", 100),  # 如果有其他数据集
        ]
        
        for dataset_name, num_samples in datasets:
            print(f"\n🔄 测试数据集: {dataset_name}")
            try:
                self.run_standard_benchmark(dataset_name=dataset_name, num_samples=num_samples)
            except Exception as e:
                print(f"❌ 数据集 {dataset_name} 测试失败: {e}")
                continue


def main():
    """主函数"""
    print("🌟 NewWork KG Standard Benchmark Integration")
    print("=" * 80)
    print("🎯 使用AutoSchemaKG标准benchmark评估NewWork概念图谱")
    print("=" * 80)
    
    try:
        # 创建benchmark实例
        benchmark = NewWorkStandardBenchmark("output/simple_test")
        
        # 选择测试模式
        print("\n📋 请选择测试模式:")
        print("1. 快速测试 (10个样本)")
        print("2. 标准测试 (50个样本)") 
        print("3. 完整评估 (200个样本)")
        
        choice = input("请输入选择 (1/2/3) [默认: 1]: ").strip() or "1"
        
        if choice == "1":
            benchmark.run_quick_test(10)
        elif choice == "2":
            benchmark.run_standard_benchmark(num_samples=50)
        elif choice == "3":
            benchmark.run_full_evaluation()
        else:
            print("❌ 无效选择，运行快速测试")
            benchmark.run_quick_test(10)
            
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断了测试")
    except Exception as e:
        print(f"❌ 程序执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()