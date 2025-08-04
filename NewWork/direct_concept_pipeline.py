#!/usr/bin/env python3
"""
Direct Concept Pipeline
直接从文章提取概念构图的完整pipeline
"""

import os
import sys
import json
from datetime import datetime
from typing import Optional

from direct_concept_config import DirectConceptConfig
from direct_concept_extractor import DirectConceptExtractor
from concept_to_graph import ConceptGraphBuilder


class DirectConceptPipeline:
    """直接概念提取pipeline主类"""
    
    def __init__(self, model, config: DirectConceptConfig):
        """
        初始化pipeline
        
        Args:
            model: LLM模型实例
            config: 配置参数
        """
        self.config = config
        self.model = model
        self.extractor = DirectConceptExtractor(model, config)
        self.graph_builder = ConceptGraphBuilder(config)
        
        # 记录执行状态
        self.execution_log = {
            'start_time': datetime.now(),
            'steps_completed': [],
            'errors': [],
            'outputs': {}
        }
    
    def log_step(self, step_name: str, success: bool = True, error: Optional[str] = None):
        """记录执行步骤"""
        step_info = {
            'step': step_name,
            'timestamp': datetime.now(),
            'success': success
        }
        
        if error:
            step_info['error'] = error
            self.execution_log['errors'].append(step_info)
        else:
            self.execution_log['steps_completed'].append(step_info)
        
        if self.config.debug_mode:
            status = "✅" if success else "❌"
            print(f"{status} {step_name}")
            if error:
                print(f"   Error: {error}")
    
    def run_concept_extraction(self) -> str:
        """
        步骤1: 运行概念提取
        
        Returns:
            概念提取结果文件路径
        """
        try:
            print("🚀 Step 1: Direct Concept Extraction")
            concept_file = self.extractor.run_extraction()
            
            self.execution_log['outputs']['concept_extraction'] = concept_file
            self.log_step("Concept Extraction", success=True)
            
            return concept_file
            
        except Exception as e:
            error_msg = f"Concept extraction failed: {str(e)}"
            self.log_step("Concept Extraction", success=False, error=error_msg)
            raise RuntimeError(error_msg) from e
    
    def convert_to_csv(self, concept_file: str) -> tuple:
        """
        步骤2: 转换为CSV格式
        
        Args:
            concept_file: 概念提取结果文件
            
        Returns:
            (concepts_csv, relationships_csv) 文件路径元组
        """
        try:
            print("📊 Step 2: Converting to CSV")
            concepts_csv, relationships_csv = self.extractor.create_concept_csv(concept_file)
            
            self.execution_log['outputs']['concepts_csv'] = concepts_csv
            self.execution_log['outputs']['relationships_csv'] = relationships_csv
            self.log_step("CSV Conversion", success=True)
            
            return concepts_csv, relationships_csv
            
        except Exception as e:
            error_msg = f"CSV conversion failed: {str(e)}"
            self.log_step("CSV Conversion", success=False, error=error_msg)
            raise RuntimeError(error_msg) from e
    
    def build_concept_graph(self, concepts_csv: str, relationships_csv: str):
        """
        步骤3: 构建概念图
        
        Args:
            concepts_csv: 概念CSV文件路径
            relationships_csv: 关系CSV文件路径
            
        Returns:
            NetworkX图对象
        """
        try:
            print("🔧 Step 3: Building Concept Graph")
            
            # 加载概念和关系
            concepts, relationships = self.graph_builder.load_concepts_from_csv(
                concepts_csv, relationships_csv
            )
            
            # 构建基础图
            G = self.graph_builder.build_concept_graph(concepts, relationships)
            
            # 添加抽象级别连接
            if self.config.include_abstraction_levels:
                G = self.graph_builder.add_abstraction_level_edges(G)
            
            # 如果有层次关系数据，添加层次连接
            # 注意：目前的实现主要针对passage_concept模式
            # hierarchical_concept模式的层次关系可以在此处添加
            
            self.log_step("Graph Construction", success=True)
            return G
            
        except Exception as e:
            error_msg = f"Graph construction failed: {str(e)}"
            self.log_step("Graph Construction", success=False, error=error_msg)
            raise RuntimeError(error_msg) from e
    
    def save_graph(self, G, output_name: Optional[str] = None):
        """
        步骤4: 保存图文件
        
        Args:
            G: NetworkX图对象
            output_name: 自定义输出文件名（可选）
        """
        try:
            print("💾 Step 4: Saving Graph")
            
            if output_name is None:
                timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
                output_name = f"{self.config.filename_pattern}_concept_graph_{timestamp}"
            
            # 保存GraphML格式（用于可视化和分析）
            graphml_path = f"{self.config.output_directory}/graph/{output_name}.graphml"
            self.graph_builder.save_graph_to_graphml(G, graphml_path)
            
            # 保存Pickle格式（用于后续处理）
            pickle_path = f"{self.config.output_directory}/graph/{output_name}.pkl"
            self.graph_builder.save_graph_to_pickle(G, pickle_path)
            
            self.execution_log['outputs']['graphml_file'] = graphml_path
            self.execution_log['outputs']['pickle_file'] = pickle_path
            self.log_step("Graph Saving", success=True)
            
            return graphml_path, pickle_path
            
        except Exception as e:
            error_msg = f"Graph saving failed: {str(e)}"
            self.log_step("Graph Saving", success=False, error=error_msg)
            raise RuntimeError(error_msg) from e
    
    def generate_statistics(self, G):
        """
        步骤5: 生成图统计信息
        
        Args:
            G: NetworkX图对象
        """
        try:
            print("📈 Step 5: Generating Statistics")
            
            # 打印统计信息
            self.graph_builder.print_graph_statistics(G)
            
            # 保存统计信息到文件
            stats = self.graph_builder.generate_graph_statistics(G)
            stats_file = f"{self.config.output_directory}/statistics.json"
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            
            self.execution_log['outputs']['statistics_file'] = stats_file
            self.log_step("Statistics Generation", success=True)
            
            return stats
            
        except Exception as e:
            error_msg = f"Statistics generation failed: {str(e)}"
            self.log_step("Statistics Generation", success=False, error=error_msg)
            print(f"Warning: {error_msg}")
            return {}
    
    def save_execution_log(self):
        """保存执行日志"""
        try:
            self.execution_log['end_time'] = datetime.now()
            self.execution_log['total_duration'] = (
                self.execution_log['end_time'] - self.execution_log['start_time']
            ).total_seconds()
            
            log_file = f"{self.config.output_directory}/execution_log.json"
            
            # 转换datetime对象为字符串
            log_to_save = self.execution_log.copy()
            log_to_save['start_time'] = self.execution_log['start_time'].isoformat()
            log_to_save['end_time'] = self.execution_log['end_time'].isoformat()
            
            for step in log_to_save['steps_completed']:
                step['timestamp'] = step['timestamp'].isoformat()
            
            for error in log_to_save['errors']:
                error['timestamp'] = error['timestamp'].isoformat()
            
            with open(log_file, 'w', encoding='utf-8') as f:
                json.dump(log_to_save, f, indent=2, ensure_ascii=False)
            
            print(f"📄 Execution log saved: {log_file}")
            
        except Exception as e:
            print(f"Warning: Failed to save execution log: {e}")
    
    def run_full_pipeline(self, output_name: Optional[str] = None):
        """
        运行完整的概念提取pipeline
        
        Args:
            output_name: 自定义输出文件名（可选）
            
        Returns:
            包含所有输出文件路径的字典
        """
        print("🎯 Starting Direct Concept Extraction Pipeline")
        print("=" * 60)
        
        try:
            # 步骤1: 概念提取
            concept_file = self.run_concept_extraction()
            
            # 步骤2: 转换为CSV
            concepts_csv, relationships_csv = self.convert_to_csv(concept_file)
            
            # 步骤3: 构建图
            G = self.build_concept_graph(concepts_csv, relationships_csv)
            
            # 步骤4: 保存图
            graphml_path, pickle_path = self.save_graph(G, output_name)
            
            # 步骤5: 生成统计信息
            stats = self.generate_statistics(G)
            
            # 保存执行日志
            self.save_execution_log()
            
            print("\n🎉 Pipeline completed successfully!")
            print("=" * 60)
            print("📁 Output files:")
            for key, value in self.execution_log['outputs'].items():
                print(f"   {key}: {value}")
            
            return self.execution_log['outputs']
            
        except Exception as e:
            print(f"\n❌ Pipeline failed: {e}")
            self.save_execution_log()
            raise
    
    def run_extraction_only(self):
        """仅运行概念提取，不构建图"""
        print("🚀 Running Concept Extraction Only")
        print("=" * 40)
        
        try:
            concept_file = self.run_concept_extraction()
            concepts_csv, relationships_csv = self.convert_to_csv(concept_file)
            
            self.save_execution_log()
            
            print("\n✅ Concept extraction completed!")
            print(f"📄 Concepts CSV: {concepts_csv}")
            print(f"📄 Relationships CSV: {relationships_csv}")
            
            return {
                'concept_file': concept_file,
                'concepts_csv': concepts_csv,
                'relationships_csv': relationships_csv
            }
            
        except Exception as e:
            print(f"\n❌ Extraction failed: {e}")
            self.save_execution_log()
            raise
    
    def run_graph_only(self, concepts_csv: str, relationships_csv: str, output_name: Optional[str] = None):
        """仅从已有CSV构建图"""
        print("🔧 Running Graph Construction Only")
        print("=" * 40)
        
        try:
            G = self.build_concept_graph(concepts_csv, relationships_csv)
            graphml_path, pickle_path = self.save_graph(G, output_name)
            stats = self.generate_statistics(G)
            
            self.save_execution_log()
            
            print("\n✅ Graph construction completed!")
            print(f"📄 GraphML: {graphml_path}")
            print(f"📄 Pickle: {pickle_path}")
            
            return {
                'graph': G,
                'graphml_file': graphml_path,
                'pickle_file': pickle_path,
                'statistics': stats
            }
            
        except Exception as e:
            print(f"\n❌ Graph construction failed: {e}")
            self.save_execution_log()
            raise


def create_default_config(
    model_path: str = "gpt-4o",
    data_directory: str = "example_data",
    filename_pattern: str = "sample",
    output_directory: str = "NewWork/output",
    extraction_mode: str = "passage_concept",
    language: str = "en"
) -> DirectConceptConfig:
    """创建默认配置"""
    
    return DirectConceptConfig(
        model_path=model_path,
        data_directory=data_directory,
        filename_pattern=filename_pattern,
        output_directory=output_directory,
        extraction_mode=extraction_mode,
        language=language,
        batch_size_concept=8,
        text_chunk_size=1024,
        chunk_overlap=100,
        include_abstraction_levels=True,
        include_hierarchical_relations=True,
        min_concept_frequency=1,
        debug_mode=True,
        record_usage=True
    ) 