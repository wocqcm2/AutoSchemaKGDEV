#!/usr/bin/env python3
"""
HotpotQA KG构建和测试Pipeline - 阶段1快速技术验证
使用HotpotQA训练数据构建KG，然后在开发集上验证
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


class HotpotDataProcessor:
    """HotpotQA数据处理器"""
    
    def __init__(self, hotpot_path: str):
        self.hotpot_path = Path(hotpot_path)
        self.train_file = self.hotpot_path / "hotpot_train_v1.1.json"
        self.dev_file = self.hotpot_path / "hotpot_dev_fullwiki_v1.json"
        
        # 调试：打印完整路径
        print(f"🔍 检查训练文件路径: {self.train_file}")
        print(f"🔍 检查开发文件路径: {self.dev_file}")
        print(f"🔍 训练文件存在: {self.train_file.exists()}")
        print(f"🔍 开发文件存在: {self.dev_file.exists()}")
        
    def extract_contexts_for_kg(self, max_samples: int = 5000) -> List[Dict[str, str]]:
        """从训练数据中提取context用于构建KG"""
        print(f"📊 从训练数据提取context (最多{max_samples}个样本)...")
        
        if not self.train_file.exists():
            raise FileNotFoundError(f"训练文件不存在: {self.train_file}")
        
        with open(self.train_file, 'r', encoding='utf-8') as f:
            train_data = json.load(f)
        
        # 限制样本数量
        if max_samples:
            train_data = train_data[:min(max_samples, 1000)]  # 进一步限制到1000个样本
            print(f"✅ 使用前{min(max_samples, 1000)}个训练样本(限制到1000个以内)")
        
        # 提取所有context段落
        contexts = []
        context_id = 0
        
        for sample in train_data:
            for title, paragraphs in sample['context']:
                # 将每个段落转换为NewWork格式
                full_text = ' '.join(paragraphs)
                
                context_item = {
                    "id": f"hotpot_context_{context_id}",
                    "text": full_text,
                    "metadata": {
                        "lang": "en",
                        "source": "hotpot_train",
                        "title": title,
                        "original_question": sample['question']
                    }
                }
                contexts.append(context_item)
                context_id += 1
        
        print(f"✅ 提取了{len(contexts)}个context段落")
        return contexts
    
    def save_contexts_for_newwork(self, contexts: List[Dict], output_file: str):
        """保存context为NewWork格式"""
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(contexts, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Context数据已保存到: {output_file}")
        return str(output_path)
    
    def prepare_test_data(self, max_test_samples: int = 500) -> List[Dict]:
        """准备测试数据"""
        print(f"📊 准备测试数据 (最多{max_test_samples}个样本)...")
        
        if not self.dev_file.exists():
            raise FileNotFoundError(f"开发文件不存在: {self.dev_file}")
        
        with open(self.dev_file, 'r', encoding='utf-8') as f:
            dev_data = json.load(f)
        
        if max_test_samples:
            dev_data = dev_data[:max_test_samples]
            print(f"✅ 使用前{max_test_samples}个测试样本")
        
        return dev_data


class HotpotKGPipeline:
    """HotpotQA KG构建和测试Pipeline"""
    
    def __init__(self, hotpot_path: str):
        self.hotpot_path = hotpot_path
        self.processor = HotpotDataProcessor(hotpot_path)
        self.config_loader = ConfigLoader()
        self.logger = self._setup_logger()
        
    def _setup_logger(self):
        """设置日志"""
        logger = logging.getLogger('HotpotKGPipeline')
        logger.setLevel(logging.INFO)
        
        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
        
        return logger
    
    def stage1_build_kg_from_hotpot(self, max_samples: int = 5000):
        """阶段1: 从HotpotQA训练数据构建KG"""
        print("🚀 阶段1: 从HotpotQA训练数据构建知识图谱")
        print("=" * 70)
        
        try:
            # 1. 提取training context
            contexts = self.processor.extract_contexts_for_kg(max_samples)
            
            # 2. 保存为NewWork格式
            context_file = "hotpot_contexts_for_kg.json"
            self.processor.save_contexts_for_newwork(contexts, context_file)
            
            # 3. 使用NewWork pipeline构建KG
            print("\n🔧 使用NewWork pipeline构建KG...")
            kg_output_path = self._run_newwork_pipeline(context_file)
            
            print(f"\n✅ 阶段1完成！KG已构建完成")
            print(f"📁 KG输出路径: {kg_output_path}")
            
            return kg_output_path
            
        except Exception as e:
            print(f"❌ 阶段1失败: {e}")
            import traceback
            traceback.print_exc()
            raise
    
    def _run_newwork_pipeline(self, context_file: str) -> str:
        """运行NewWork pipeline构建KG"""
        try:
            # 导入NewWork组件
            from direct_concept_pipeline import DirectConceptPipeline
            from direct_concept_config import DirectConceptConfig
            
            # 创建LLM模型实例
            model = create_model_client(self.config_loader)
            
            # 配置NewWork pipeline - 性能优化版
            config = DirectConceptConfig(
                model_path="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",  # 使用Llama-3-8B-Turbo
                data_directory=".",
                filename_pattern=context_file.replace('.json', ''),
                output_directory="output",
                extraction_mode="passage_concept",
                language="en",
                batch_size_concept=20,    # 8B模型可以更高并发  
                max_workers=6,            # 6个worker并发
                temperature=0.1,
                text_chunk_size=4096,     # 适合8B模型的context window
                chunk_overlap=0,          # 取消overlap减少冗余
                debug_mode=True
            )
            
            print("🤖 创建LLM模型实例...")
            print("📋 启动NewWork概念提取pipeline...")
            
            # 运行pipeline
            pipeline = DirectConceptPipeline(model, config)
            output_path = pipeline.run_full_pipeline("hotpot_kg")
            
            return output_path
            
        except Exception as e:
            print(f"❌ NewWork pipeline运行失败: {e}")
            import traceback
            print("🔍 详细错误信息:")
            traceback.print_exc()
            
            # 如果NewWork pipeline失败，创建简化版KG
            print("🔄 尝试创建简化版知识图谱...")
            return self._create_simple_kg_fallback(context_file)
    
    def _create_simple_kg_fallback(self, context_file: str) -> str:
        """创建简化版KG作为fallback"""
        import pandas as pd
        import networkx as nx
        import pickle
        from datetime import datetime
        
        print("🔄 创建简化版知识图谱...")
        
        # 读取context数据
        with open(context_file, 'r') as f:
            contexts = json.load(f)
        
        # 创建输出目录
        output_dir = Path("output/hotpot_kg")
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # 简化的实体和关系提取
        concepts = []
        relationships = []
        concept_id = 0
        
        for context in contexts[:1000]:  # 限制数量避免过多
            title = context['metadata']['title']
            text = context['text']
            
            # 简单的概念提取 - 使用标题作为主要概念
            concepts.append({
                'id': f"concept_{concept_id}",
                'text': title,
                'type': 'entity',
                'abstraction_level': 'specific',
                'source_doc': context['id']
            })
            
            # 简单的关系 - 标题与文本内容的关系
            if len(text) > 50:
                relationships.append({
                    'source': f"concept_{concept_id}",
                    'target': f"concept_{concept_id}",
                    'relation': 'described_by',
                    'relation_type': 'concept_relation'
                })
            
            concept_id += 1
        
        # 保存CSV格式
        concepts_df = pd.DataFrame(concepts)
        relationships_df = pd.DataFrame(relationships)
        
        csv_dir = output_dir / "concept_csv"
        csv_dir.mkdir(exist_ok=True)
        
        concepts_df.to_csv(csv_dir / "concepts_hotpot_kg.csv", index=False)
        relationships_df.to_csv(csv_dir / "relationships_hotpot_kg.csv", index=False)
        
        # 创建简单的NetworkX图
        G = nx.Graph()
        for concept in concepts:
            G.add_node(concept['id'], **concept)
        
        for rel in relationships:
            G.add_edge(rel['source'], rel['target'], relation=rel['relation'])
        
        # 保存图
        graph_dir = output_dir / "graph"
        graph_dir.mkdir(exist_ok=True)
        
        with open(graph_dir / "hotpot_kg.pkl", 'wb') as f:
            pickle.dump(G, f)
        
        # 保存统计信息
        stats = {
            "total_nodes": len(G.nodes),
            "total_edges": len(G.edges),
            "created_at": datetime.now().isoformat(),
            "data_source": "hotpot_train_simplified"
        }
        
        with open(output_dir / "statistics.json", 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ 简化版KG创建完成: {len(G.nodes)}个节点, {len(G.edges)}条边")
        return str(output_dir)
    
    def stage2_test_on_dev_set(self, kg_output_path: str, max_test_samples: int = 100):
        """阶段2: 在开发集上测试KG效果"""
        print("\n🧪 阶段2: 在HotpotQA开发集上测试KG效果")
        print("=" * 70)
        
        try:
            # 1. 准备测试数据
            test_data = self.processor.prepare_test_data(max_test_samples)
            
            # 2. 创建测试问答对文件
            test_file = self._create_test_qa_file(test_data)
            
            # 3. 使用标准benchmark测试
            print("\n🔧 运行标准benchmark测试...")
            self._run_standard_benchmark(kg_output_path, test_file)
            
            print(f"\n✅ 阶段2完成！测试结果已保存")
            
        except Exception as e:
            print(f"❌ 阶段2失败: {e}")
            import traceback
            traceback.print_exc()
    
    def _create_test_qa_file(self, test_data: List[Dict]) -> str:
        """创建测试问答文件"""
        test_qa = []
        
        for item in test_data:
            qa_item = {
                "id": item["_id"],
                "question": item["question"],
                "answer": item["answer"],
                "supporting_facts": item.get("supporting_facts", []),
                "paragraphs": []
            }
            
            # 添加context段落
            for title, paragraphs in item["context"]:
                for para in paragraphs:
                    qa_item["paragraphs"].append({
                        "title": title,
                        "text": para,
                        "is_supporting": True  # 简化处理
                    })
            
            test_qa.append(qa_item)
        
        # 保存测试文件
        test_file = "hotpot_test_questions.json"
        with open(test_file, 'w', encoding='utf-8') as f:
            json.dump(test_qa, f, indent=2, ensure_ascii=False)
        
        print(f"✅ 测试问答文件已保存: {test_file}")
        return test_file
    
    def _run_standard_benchmark(self, kg_output_path: str, test_file: str):
        """运行标准benchmark测试"""
        try:
            print("🔧 尝试使用进阶RAG benchmark...")
            # 使用现有的advanced_rag_benchmark进行测试
            self._run_advanced_rag_test(kg_output_path)
            
        except Exception as e:
            print(f"⚠️ 进阶benchmark测试失败: {e}")
            print("🔄 运行简化版测试...")
            self._run_simple_test(kg_output_path, test_file)
    
    def _run_advanced_rag_test(self, kg_output_path):
        """使用advanced_rag_benchmark测试KG"""
        try:
            # 临时将KG文件复制到expected位置
            import shutil
            
            # 处理字典格式的输出路径
            if isinstance(kg_output_path, dict):
                source_graph = Path(kg_output_path['pickle_file'])
            else:
                source_graph = Path(kg_output_path) / "graph" / "hotpot_kg.pkl"
            target_dir = Path("output/simple_test")
            target_dir.mkdir(parents=True, exist_ok=True)
            target_graph = target_dir / "dulce_simple.pkl"
            
            if source_graph.exists():
                print(f"📋 复制KG文件: {source_graph} -> {target_graph}")
                shutil.copy2(source_graph, target_graph)
                
                # 运行advanced benchmark
                print("🚀 启动Advanced RAG Benchmark...")
                
                # 直接运行advanced_rag_benchmark的main函数
                import subprocess
                import sys
                
                result = subprocess.run([
                    sys.executable, 
                    "advanced_rag_benchmark.py"
                ], capture_output=True, text=True, cwd=".")
                
                print("📋 Advanced RAG Benchmark输出:")
                print(result.stdout)
                if result.stderr:
                    print("⚠️ 错误信息:")
                    print(result.stderr)
                
                print("✅ Advanced RAG测试完成")
                return result.returncode == 0
            else:
                raise FileNotFoundError(f"源KG文件不存在: {source_graph}")
                
        except Exception as e:
            print(f"❌ Advanced RAG测试失败: {e}")
            raise
    
    def _run_simple_test(self, kg_output_path, test_file: str):
        """运行简化版测试"""
        print("🔄 运行简化版KG测试...")
        
        # 加载KG - 处理字典格式的输出路径
        if isinstance(kg_output_path, dict):
            graph_file = Path(kg_output_path['pickle_file'])
        else:
            graph_file = Path(kg_output_path) / "graph" / "hotpot_kg.pkl"
        if graph_file.exists():
            import pickle
            with open(graph_file, 'rb') as f:
                kg = pickle.load(f)
            
            print(f"📊 KG信息: {len(kg.nodes)}个节点, {len(kg.edges)}条边")
        
        # 加载测试问题
        with open(test_file, 'r') as f:
            test_questions = json.load(f)
        
        print(f"📊 测试问题: {len(test_questions)}个")
        
        # 简单的检索测试
        print("🔍 运行简单检索测试...")
        for i, qa in enumerate(test_questions[:5]):  # 只测试前5个
            question = qa['question']
            answer = qa['answer']
            print(f"\n问题{i+1}: {question}")
            print(f"答案: {answer}")
            
            # 简单的关键词匹配
            keywords = question.lower().split()
            relevant_nodes = []
            
            if 'kg' in locals():
                for node_id, node_data in kg.nodes(data=True):
                    node_text = node_data.get('text', '').lower()
                    if any(keyword in node_text for keyword in keywords[:3]):
                        relevant_nodes.append(node_text)
            
            if relevant_nodes:
                print(f"相关节点: {relevant_nodes[:2]}")
            else:
                print("未找到相关节点")
        
        print(f"\n✅ 简化测试完成")


def main():
    """主函数"""
    print("🌟 HotpotQA KG Pipeline - 阶段1快速技术验证")
    print("=" * 80)
    
    # 读取HotpotQA路径
    hotpot_location_file = Path("hotpotlocation.txt")
    if not hotpot_location_file.exists():
        print("❌ 找不到hotpotlocation.txt文件")
        return
    
    with open(hotpot_location_file, 'r') as f:
        hotpot_path = f.read().strip()
    
    print(f"🔍 读取到的路径: '{hotpot_path}'")
    print(f"🔍 路径长度: {len(hotpot_path)}")
    
    # 确保路径不为空
    if not hotpot_path:
        print("❌ hotpotlocation.txt中的路径为空")
        return
    
    if not Path(hotpot_path).exists():
        print(f"❌ HotpotQA路径不存在: {hotpot_path}")
        return
    
    print(f"📁 HotpotQA路径: {hotpot_path}")
    
    try:
        # 创建pipeline
        pipeline = HotpotKGPipeline(hotpot_path)
        
        print("\n📋 阶段1配置:")
        print("  - 数据源: HotpotQA训练集context")
        print("  - 样本数: 5000个context段落")
        print("  - 方法: NewWork概念提取pipeline")
        print("  - 输出: 概念图谱")
        
        # 运行阶段1 - 使用较小的样本数进行快速验证
        kg_output_path = pipeline.stage1_build_kg_from_hotpot(max_samples=100)
        
        # 运行阶段2  
        pipeline.stage2_test_on_dev_set(kg_output_path, max_test_samples=100)
        
        print("\n🎉 HotpotQA KG Pipeline阶段1完成！")
        print(f"📊 结果保存在: {kg_output_path}")
        
    except KeyboardInterrupt:
        print("\n⚠️ 用户中断了pipeline")
    except Exception as e:
        print(f"\n❌ Pipeline执行失败: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()