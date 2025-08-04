#!/usr/bin/env python3
"""
Direct Concept Extraction Configuration
直接概念提取的配置类
"""

from dataclasses import dataclass
from typing import Optional

@dataclass
class DirectConceptConfig:
    """直接概念提取的配置参数"""
    
    # 模型相关配置
    model_path: str = "gpt-4o"
    max_new_tokens: int = 2048
    temperature: float = 0.7
    max_workers: int = 3
    
    # 数据相关配置
    data_directory: str = "example_data"
    filename_pattern: str = "sample"
    output_directory: str = "output"
    
    # 批处理配置
    batch_size_concept: int = 16
    text_chunk_size: int = 1024
    chunk_overlap: int = 100
    
    # 概念提取模式
    extraction_mode: str = "passage_concept"  # "passage_concept" 或 "hierarchical_concept"
    language: str = "en"  # "en" 或 "zh"
    
    # 图构建配置
    include_abstraction_levels: bool = True
    include_hierarchical_relations: bool = True
    min_concept_frequency: int = 1
    
    # 输出配置
    save_intermediate_results: bool = True
    debug_mode: bool = False
    record_usage: bool = False
    
    # 处理配置
    remove_doc_spaces: bool = True
    normalize_concept_names: bool = True
    filter_low_quality_concepts: bool = True
    
    # 分片配置（用于大数据集）
    current_shard_concept: int = 0
    total_shards_concept: int = 1

    def __post_init__(self):
        """配置后处理和验证"""
        if self.extraction_mode not in ["passage_concept", "hierarchical_concept"]:
            raise ValueError("extraction_mode must be 'passage_concept' or 'hierarchical_concept'")
        
        if self.language not in ["en", "zh"]:
            raise ValueError("language must be 'en' or 'zh'")
        
        if self.batch_size_concept <= 0:
            raise ValueError("batch_size_concept must be positive")
        
        if self.text_chunk_size <= 0:
            raise ValueError("text_chunk_size must be positive") 