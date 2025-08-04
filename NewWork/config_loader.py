#!/usr/bin/env python3
"""
Configuration Loader for NewWork Pipeline
从config.json加载配置的工具类
"""

import json
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass

from direct_concept_config import DirectConceptConfig


@dataclass
class ApiConfig:
    """API配置类"""
    base_url: str
    api_key: str
    model: str
    timeout: int = 120
    max_retries: int = 3


class ConfigLoader:
    """配置加载器"""
    
    def __init__(self, config_path: str = "config.json"):
        """
        初始化配置加载器
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = Path(config_path)
        self.config_data = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not self.config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {self.config_path}")
        
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = json.load(f)
            return config
        except json.JSONDecodeError as e:
            raise ValueError(f"配置文件格式错误: {e}")
        except Exception as e:
            raise RuntimeError(f"加载配置文件失败: {e}")
    
    def get_api_config(self, model_name: Optional[str] = None) -> ApiConfig:
        """
        获取API配置
        
        Args:
            model_name: 模型名称，如果不指定则使用默认模型
            
        Returns:
            ApiConfig: API配置对象
        """
        api_section = self.config_data.get("api", {})
        
        # 如果指定了模型名称，从models部分获取
        if model_name and model_name in self.config_data.get("models", {}):
            model = self.config_data["models"][model_name]
        else:
            model = api_section.get("model", "gpt-4o")
        
        return ApiConfig(
            base_url=api_section.get("base_url", "https://api.openai.com/v1"),
            api_key=api_section.get("api_key", os.getenv("OPENAI_API_KEY", "")),
            model=model,
            timeout=api_section.get("timeout", 120),
            max_retries=api_section.get("max_retries", 3)
        )
    
    def get_direct_concept_config(
        self, 
        model_name: Optional[str] = None,
        filename_pattern: Optional[str] = None,
        output_name: Optional[str] = None,
        **overrides
    ) -> DirectConceptConfig:
        """
        获取DirectConceptConfig配置
        
        Args:
            model_name: 模型名称
            filename_pattern: 文件名模式
            output_name: 输出名称
            **overrides: 覆盖的配置参数
            
        Returns:
            DirectConceptConfig: 直接概念提取配置
        """
        # 获取模型路径
        if model_name and model_name in self.config_data.get("models", {}):
            model_path = self.config_data["models"][model_name]
        else:
            model_path = self.config_data.get("api", {}).get("model", "gpt-4o")
        
        # 合并各部分配置
        extraction_config = self.config_data.get("extraction", {})
        processing_config = self.config_data.get("processing", {})
        paths_config = self.config_data.get("paths", {})
        logging_config = self.config_data.get("logging", {})
        
        # 构建配置参数
        config_params = {
            # 模型配置
            "model_path": model_path,
            "max_new_tokens": extraction_config.get("max_new_tokens", 2048),
            "temperature": extraction_config.get("temperature", 0.1),
            "max_workers": extraction_config.get("max_workers", 3),
            
            # 数据配置
            "data_directory": paths_config.get("data_directory", "../example_data"),
            "filename_pattern": filename_pattern or paths_config.get("default_filename_pattern", "sample"),
            "output_directory": os.path.join(
                paths_config.get("output_directory", "output"),
                output_name or "default"
            ),
            
            # 处理配置
            "batch_size_concept": extraction_config.get("batch_size_concept", 8),
            "text_chunk_size": extraction_config.get("text_chunk_size", 1024),
            "chunk_overlap": extraction_config.get("chunk_overlap", 100),
            
            # 提取配置
            "extraction_mode": processing_config.get("extraction_mode", "passage_concept"),
            "language": processing_config.get("language", "en"),
            
            # 图构建配置
            "include_abstraction_levels": processing_config.get("include_abstraction_levels", True),
            "include_hierarchical_relations": processing_config.get("include_hierarchical_relations", True),
            "min_concept_frequency": processing_config.get("min_concept_frequency", 1),
            
            # 质量控制
            "normalize_concept_names": processing_config.get("normalize_concept_names", True),
            "filter_low_quality_concepts": processing_config.get("filter_low_quality_concepts", True),
            "remove_doc_spaces": processing_config.get("remove_doc_spaces", True),
            
            # 调试配置
            "debug_mode": logging_config.get("debug_mode", True),
            "record_usage": logging_config.get("record_usage", True),
            "save_intermediate_results": logging_config.get("save_intermediate_results", True)
        }
        
        # 应用覆盖参数
        config_params.update(overrides)
        
        return DirectConceptConfig(**config_params)
    
    def list_available_models(self) -> Dict[str, str]:
        """列出可用的模型"""
        return self.config_data.get("models", {})
    
    def get_model_by_name(self, model_name: str) -> str:
        """根据简短名称获取完整模型名"""
        models = self.config_data.get("models", {})
        if model_name in models:
            return models[model_name]
        else:
            raise ValueError(f"模型 '{model_name}' 不存在。可用模型: {list(models.keys())}")
    
    def update_config(self, section: str, key: str, value: Any):
        """更新配置值"""
        if section not in self.config_data:
            self.config_data[section] = {}
        self.config_data[section][key] = value
    
    def save_config(self):
        """保存配置到文件"""
        with open(self.config_path, 'w', encoding='utf-8') as f:
            json.dump(self.config_data, f, ensure_ascii=False, indent=2)
    
    def print_config_summary(self):
        """打印配置摘要"""
        print("📋 Configuration Summary:")
        print("=" * 50)
        
        # API配置
        api_config = self.get_api_config()
        print(f"🔗 API URL: {api_config.base_url}")
        print(f"🤖 Default Model: {api_config.model}")
        print(f"🔑 API Key: {api_config.api_key[:10]}...{api_config.api_key[-4:]}")
        
        # 可用模型
        models = self.list_available_models()
        print(f"\n🎯 Available Models ({len(models)}):")
        for short_name, full_name in models.items():
            print(f"   {short_name}: {full_name}")
        
        # 处理配置
        extraction = self.config_data.get("extraction", {})
        processing = self.config_data.get("processing", {})
        print(f"\n⚙️ Processing Settings:")
        print(f"   Batch Size: {extraction.get('batch_size_concept', 8)}")
        print(f"   Chunk Size: {extraction.get('text_chunk_size', 1024)}")
        print(f"   Temperature: {extraction.get('temperature', 0.1)}")
        print(f"   Mode: {processing.get('extraction_mode', 'passage_concept')}")
        print(f"   Language: {processing.get('language', 'en')}")


def create_model_client(config_loader: ConfigLoader, model_name: Optional[str] = None):
    """
    根据配置创建模型客户端
    
    Args:
        config_loader: 配置加载器
        model_name: 模型名称
        
    Returns:
        LLMGenerator: 模型生成器
    """
    import sys
    sys.path.append('..')
    
    api_config = config_loader.get_api_config(model_name)
    
    # 判断是否使用OpenAI API格式
    if "openai" in api_config.base_url.lower() or api_config.base_url.endswith("/v1/openai"):
        from openai import OpenAI
        
        client = OpenAI(
            api_key=api_config.api_key,
            base_url=api_config.base_url,
            timeout=api_config.timeout,
            max_retries=api_config.max_retries
        )
        
        from atlas_rag.llm_generator import LLMGenerator
        return LLMGenerator(client, model_name=api_config.model)
    
    else:
        # 本地模型或其他API
        from transformers import pipeline
        from atlas_rag.llm_generator import LLMGenerator
        
        # 本地模型配置
        performance_config = config_loader.config_data.get("performance", {})
        
        client = pipeline(
            "text-generation",
            model=api_config.model,
            device_map=performance_config.get("device_map", "auto"),
            load_in_8bit=performance_config.get("load_in_8bit", False),
            load_in_4bit=performance_config.get("load_in_4bit", False)
        )
        
        return LLMGenerator(client, model_name=api_config.model)


# 便捷函数
def load_config(config_path: str = "config.json") -> ConfigLoader:
    """加载配置文件"""
    return ConfigLoader(config_path)


def quick_setup(model_name: Optional[str] = None, config_path: str = "config.json"):
    """
    快速设置：加载配置并创建模型
    
    Args:
        model_name: 模型名称
        config_path: 配置文件路径
        
    Returns:
        tuple: (config_loader, model, model_name)
    """
    config_loader = load_config(config_path)
    model = create_model_client(config_loader, model_name)
    
    api_config = config_loader.get_api_config(model_name)
    
    return config_loader, model, api_config.model