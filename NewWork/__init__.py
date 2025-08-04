"""
Direct Concept Extraction Pipeline
直接从文章提取概念构图的新pipeline

主要模块:
- DirectConceptPipeline: 完整pipeline类
- DirectConceptConfig: 配置参数类
- DirectConceptExtractor: 核心提取器
- ConceptGraphBuilder: 图构建器
"""

from .direct_concept_pipeline import DirectConceptPipeline, create_default_config
from .direct_concept_config import DirectConceptConfig
from .direct_concept_extractor import DirectConceptExtractor
from .concept_to_graph import ConceptGraphBuilder

__version__ = "1.0.0"
__author__ = "AutoSchemaKG Team"

__all__ = [
    "DirectConceptPipeline",
    "DirectConceptConfig", 
    "DirectConceptExtractor",
    "ConceptGraphBuilder",
    "create_default_config"
] 