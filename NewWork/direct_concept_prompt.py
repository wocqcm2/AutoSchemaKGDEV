#!/usr/bin/env python3
"""
Direct Concept Extraction Prompts
从文章直接提取概念的prompt模板
"""

DIRECT_CONCEPT_INSTRUCTIONS = {
    "en": {
        "system": "You are a helpful AI assistant specialized in concept extraction and knowledge graph construction.",
        
        "passage_concept": """Given a passage, extract important concepts and their relationships to construct a knowledge graph. 
        Focus on identifying:
        1. **Key Entities**: Important nouns, people, places, organizations, objects
        2. **Abstract Concepts**: Ideas, theories, principles, categories, types
        3. **Events/Actions**: Significant activities, processes, occurrences
<<<<<<< HEAD
        4. **Relationships**: How these concepts connect to each other
        
        For each concept, also provide:
        - **Type**: entity, abstract_concept, event, or process
        - **Abstraction Level**: specific, general, or abstract
        - **Related Concepts**: other concepts it connects to
=======
        4. **Cross-Concept Relationships**: ONLY meaningful connections between DIFFERENT concepts
        
        **CRITICAL RULES for relationships**:
        - Source and target must be DIFFERENT concepts (NO self-referential relationships)
        - Each relationship must represent a clear, meaningful connection
        - Use specific relationship types: contains, belongs_to, causes, leads_to, is_part_of, relates_to, influences
>>>>>>> 0ff854f19280eadc04f4289414abf37019510f1e
        
        You must **strictly output in the following JSON format**:
        {
            "concepts": [
                {
                    "name": "concept name",
                    "type": "entity|abstract_concept|event|process", 
                    "abstraction_level": "specific|general|abstract",
<<<<<<< HEAD
                    "description": "brief description",
                    "related_concepts": ["related concept 1", "related concept 2"]
=======
                    "description": "brief description"
>>>>>>> 0ff854f19280eadc04f4289414abf37019510f1e
                }
            ],
            "relationships": [
                {
                    "source": "concept A",
                    "target": "concept B", 
<<<<<<< HEAD
                    "relation": "relationship type",
=======
                    "relation": "contains|belongs_to|causes|leads_to|is_part_of|relates_to|influences",
>>>>>>> 0ff854f19280eadc04f4289414abf37019510f1e
                    "description": "relationship description"
                }
            ]
        }""",
        
        "hierarchical_concept": """Given a passage, extract concepts at different abstraction levels and organize them hierarchically.
        
        Extract concepts in three levels:
        1. **Specific Level**: Concrete entities, specific events, particular objects
        2. **General Level**: Categories, types, general processes  
        3. **Abstract Level**: High-level ideas, principles, theoretical concepts
        
        Also identify hierarchical relationships: is_a, part_of, instance_of, category_of
        
        You must **strictly output in the following JSON format**:
        {
            "specific_concepts": [
                {
                    "name": "specific concept",
                    "type": "entity|event|process",
                    "description": "description"
                }
            ],
            "general_concepts": [
                {
                    "name": "general concept", 
                    "type": "category|type|class",
                    "description": "description"
                }
            ],
            "abstract_concepts": [
                {
                    "name": "abstract concept",
                    "type": "principle|idea|theory", 
                    "description": "description"
                }
            ],
            "hierarchical_relations": [
                {
                    "child": "more specific concept",
                    "parent": "more general concept",
                    "relation_type": "is_a|part_of|instance_of|category_of"
                }
            ]
        }""",
        
        "passage_start": "Given the following passage:"
    },
    
    "zh": {
        "system": "你是一个专门从事概念提取和知识图谱构建的AI助手。",
        
        "passage_concept": """给定一段文章，提取重要的概念及其关系来构建知识图谱。
        重点识别：
        1. **关键实体**：重要的名词、人物、地点、组织、物体
        2. **抽象概念**：思想、理论、原理、分类、类型
        3. **事件/行为**：重要的活动、过程、事件
<<<<<<< HEAD
        4. **关系**：这些概念之间的连接方式
=======
        4. **跨概念关系**：仅提取不同概念之间的有意义连接
        
        **关系提取关键规则**：
        - 来源和目标必须是不同的概念（禁止自引用关系）
        - 每个关系必须表示清晰、有意义的连接
        - 使用具体的关系类型：包含、属于、导致、引发、是_部分、相关于、影响
>>>>>>> 0ff854f19280eadc04f4289414abf37019510f1e
        
        对每个概念，还需提供：
        - **类型**：实体、抽象概念、事件或过程
        - **抽象级别**：具体、一般或抽象
        - **相关概念**：与之相连的其他概念
        
        你必须**严格按照以下JSON格式输出**：
        {
            "concepts": [
                {
                    "name": "概念名称",
                    "type": "entity|abstract_concept|event|process", 
                    "abstraction_level": "specific|general|abstract",
                    "description": "简要描述",
                    "related_concepts": ["相关概念1", "相关概念2"]
                }
            ],
            "relationships": [
                {
                    "source": "概念A",
                    "target": "概念B", 
                    "relation": "关系类型",
                    "description": "关系描述"
                }
            ]
        }""",
        
        "hierarchical_concept": """给定一段文章，提取不同抽象层次的概念并按层次组织。
        
        在三个层次提取概念：
        1. **具体层次**：具体实体、特定事件、特定对象
        2. **一般层次**：分类、类型、一般过程
        3. **抽象层次**：高层思想、原理、理论概念
        
        同时识别层次关系：是一种、部分、实例、分类
        
        你必须**严格按照以下JSON格式输出**：
        {
            "specific_concepts": [
                {
                    "name": "具体概念",
                    "type": "entity|event|process",
                    "description": "描述"
                }
            ],
            "general_concepts": [
                {
                    "name": "一般概念", 
                    "type": "category|type|class",
                    "description": "描述"
                }
            ],
            "abstract_concepts": [
                {
                    "name": "抽象概念",
                    "type": "principle|idea|theory", 
                    "description": "描述"
                }
            ],
            "hierarchical_relations": [
                {
                    "child": "更具体的概念",
                    "parent": "更一般的概念",
                    "relation_type": "is_a|part_of|instance_of|category_of"
                }
            ]
        }""",
        
        "passage_start": "给定以下段落："
    }
} 