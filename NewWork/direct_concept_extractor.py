#!/usr/bin/env python3
"""
Direct Concept Extractor
直接从文章提取概念的核心类
"""

import os
import json
import csv
import re
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import json_repair
from datasets import load_dataset
from concurrent.futures import ThreadPoolExecutor

from direct_concept_prompt import DIRECT_CONCEPT_INSTRUCTIONS
from direct_concept_config import DirectConceptConfig


class TextChunker:
    """文本分块处理器"""
    
    def __init__(self, chunk_size: int = 1024, overlap: int = 100):
        self.chunk_size = chunk_size
        self.overlap = overlap
    
    def chunk_text(self, text: str) -> List[str]:
        """将长文本分块"""
        if len(text) <= self.chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            # 尝试在句号处分割
            if end < len(text):
                # 向后查找句号
                sentence_end = text.rfind('.', start, end)
                if sentence_end > start + self.chunk_size // 2:
                    end = sentence_end + 1
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            # 计算下一个块的起始位置，考虑重叠
            start = max(end - self.overlap, start + 1)
            
            if start >= len(text):
                break
        
        return chunks


class ConceptDataProcessor:
    """概念数据处理器"""
    
    def __init__(self, config: DirectConceptConfig):
        self.config = config
        self.chunker = TextChunker(config.text_chunk_size, config.chunk_overlap)
    
    def prepare_dataset(self, raw_dataset) -> List[Dict[str, Any]]:
        """准备数据集用于概念提取"""
        processed_samples = []
        
        for item in tqdm(raw_dataset, desc="Processing dataset"):
            text = item.get('text', '')
            metadata = item.get('metadata', {})
            
            # 清理文本
            if self.config.remove_doc_spaces:
                text = re.sub(r'\s+', ' ', text).strip()
            
            # 分块处理长文本
            text_chunks = self.chunker.chunk_text(text)
            
            for i, chunk in enumerate(text_chunks):
                chunk_metadata = metadata.copy()
                chunk_metadata['chunk_id'] = i
                chunk_metadata['total_chunks'] = len(text_chunks)
                chunk_metadata['original_length'] = len(text)
                
                processed_samples.append({
                    'text': chunk,
                    'metadata': chunk_metadata,
                    'chunk_id': f"{metadata.get('file_id', 'unknown')}_{i}"
                })
        
        return processed_samples


class ConceptOutputParser:
    """概念输出解析器"""
    
    def __init__(self, config: DirectConceptConfig):
        self.config = config
    
    def parse_concept_output(self, output: str) -> Dict[str, Any]:
        """解析LLM输出的概念数据"""
        try:
            # 使用json_repair来修复可能损坏的JSON
            parsed_data = json_repair.loads(output)
            
            # 验证数据格式
            if self.config.extraction_mode == "passage_concept":
                return self._validate_passage_concept(parsed_data)
            elif self.config.extraction_mode == "hierarchical_concept":
                return self._validate_hierarchical_concept(parsed_data)
            
        except Exception as e:
            print(f"Error parsing concept output: {e}")
            return {"concepts": [], "relationships": []}
    
    def _validate_passage_concept(self, data: Dict) -> Dict:
        """验证passage_concept格式的数据"""
        validated = {"concepts": [], "relationships": []}
        
        # 验证concepts
        if "concepts" in data and isinstance(data["concepts"], list):
            for concept in data["concepts"]:
                if self._is_valid_concept(concept):
                    validated["concepts"].append(concept)
        
        # 验证relationships
        if "relationships" in data and isinstance(data["relationships"], list):
            for rel in data["relationships"]:
                if self._is_valid_relationship(rel):
                    validated["relationships"].append(rel)
        
        return validated
    
    def _validate_hierarchical_concept(self, data: Dict) -> Dict:
        """验证hierarchical_concept格式的数据"""
        validated = {
            "specific_concepts": [],
            "general_concepts": [],
            "abstract_concepts": [],
            "hierarchical_relations": []
        }
        
        # 验证各级别概念
        for level in ["specific_concepts", "general_concepts", "abstract_concepts"]:
            if level in data and isinstance(data[level], list):
                for concept in data[level]:
                    if self._is_valid_simple_concept(concept):
                        validated[level].append(concept)
        
        # 验证层次关系
        if "hierarchical_relations" in data and isinstance(data["hierarchical_relations"], list):
            for rel in data["hierarchical_relations"]:
                if self._is_valid_hierarchical_relation(rel):
                    validated["hierarchical_relations"].append(rel)
        
        return validated
    
    def _is_valid_concept(self, concept: Dict) -> bool:
        """验证concept是否有效"""
        required_fields = ["name", "type", "abstraction_level"]
        return all(field in concept for field in required_fields)
    
    def _is_valid_simple_concept(self, concept: Dict) -> bool:
        """验证简单concept是否有效"""
        required_fields = ["name", "type"]
        return all(field in concept for field in required_fields)
    
    def _is_valid_relationship(self, relationship: Dict) -> bool:
<<<<<<< HEAD
        """验证relationship是否有效"""
        required_fields = ["source", "target", "relation"]
        return all(field in relationship for field in required_fields)
=======
        """验证relationship是否有效，严格禁止自引用"""
        required_fields = ["source", "target", "relation"]
        
        # 基础字段验证
        if not all(field in relationship for field in required_fields):
            return False
        
        # 🔥 关键：严格禁止自引用关系
        source = str(relationship["source"]).strip().lower()
        target = str(relationship["target"]).strip().lower()
        
        if source == target:
            if self.config.debug_mode:
                print(f"❌ Rejected self-referential relationship: {source} -> {target}")
            return False
        
        # 验证关系类型有效性
        valid_relations = {
            "contains", "belongs_to", "causes", "leads_to", "is_part_of", 
            "relates_to", "influences", "located_in", "is_a", "has", "uses"
        }
        
        relation = str(relationship["relation"]).strip().lower()
        if relation not in valid_relations:
            if self.config.debug_mode:
                print(f"⚠️ Unknown relation type: {relation}")
        
        return True
    
    def _filter_and_deduplicate_relationships(self, relationships: List[Dict]) -> List[Dict]:
        """过滤和去重关系，确保高质量"""
        if not relationships:
            return []
        
        filtered = []
        seen_relations = set()
        
        for rel in relationships:
            # 先验证有效性（包括反自引检查）
            if not self._is_valid_relationship(rel):
                continue
            
            # 标准化关系表示用于去重
            source = str(rel["source"]).strip().lower()
            target = str(rel["target"]).strip().lower()
            relation = str(rel["relation"]).strip().lower()
            
            # 创建关系签名（考虑双向关系）
            rel_signature = tuple(sorted([source, target]) + [relation])
            
            if rel_signature not in seen_relations:
                seen_relations.add(rel_signature)
                filtered.append(rel)
            elif self.config.debug_mode:
                print(f"🔄 Deduplicated relationship: {source} -{relation}-> {target}")
        
        return filtered
>>>>>>> 0ff854f19280eadc04f4289414abf37019510f1e
    
    def _is_valid_hierarchical_relation(self, relation: Dict) -> bool:
        """验证hierarchical_relation是否有效"""
        required_fields = ["child", "parent", "relation_type"]
        return all(field in relation for field in required_fields)


class DirectConceptExtractor:
    """直接概念提取器主类"""
    
    def __init__(self, model, config: DirectConceptConfig):
        self.config = config
        self.model = model
        self.processor = ConceptDataProcessor(config)
        self.parser = ConceptOutputParser(config)
        
        # 创建输出目录
        os.makedirs(self.config.output_directory, exist_ok=True)
        os.makedirs(f"{self.config.output_directory}/concepts", exist_ok=True)
    
    def load_dataset(self):
        """加载数据集"""
        data_path = Path(self.config.data_directory)
        all_files = os.listdir(data_path)
        
        valid_files = [
            filename for filename in all_files
            if filename.startswith(self.config.filename_pattern) and
            (filename.endswith(".json.gz") or filename.endswith(".json") or 
             filename.endswith(".jsonl") or filename.endswith(".jsonl.gz"))
        ]
        
        print(f"Found data files: {valid_files}")
        dataset_config = {"train": valid_files}
<<<<<<< HEAD
        return load_dataset("json", data_files=dataset_config["train"])
=======
        return load_dataset(self.config.data_directory, data_files=dataset_config["train"])
>>>>>>> 0ff854f19280eadc04f4289414abf37019510f1e
    
    def create_concept_extraction_messages(self, text_batch: List[str]) -> List[List[Dict]]:
        """创建概念提取的消息"""
        messages = []
        
        # 获取语言对应的指令
        instructions = DIRECT_CONCEPT_INSTRUCTIONS.get(
            self.config.language, 
            DIRECT_CONCEPT_INSTRUCTIONS["en"]
        )
        
        system_msg = instructions["system"]
        
        if self.config.extraction_mode == "passage_concept":
            concept_prompt = instructions["passage_concept"]
        else:
            concept_prompt = instructions["hierarchical_concept"]
        
        for text in text_batch:
            user_msg = f"{concept_prompt}\n\n{instructions['passage_start']}\n{text}"
            
            message = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg}
            ]
            messages.append(message)
        
        return messages
    
    def extract_concepts_batch(self, text_batch: List[str]) -> List[Dict]:
        """批量提取概念"""
        messages = self.create_concept_extraction_messages(text_batch)
        
        try:
            # 调用模型进行概念提取
            outputs = self.model.generate_response(
                messages, 
                max_new_tokens=self.config.max_new_tokens,
                temperature=self.config.temperature,
                return_text_only=not self.config.record_usage,
                max_workers=self.config.max_workers
            )
            
            if self.config.record_usage:
                text_outputs = [output[0] for output in outputs]
                usages = [output[1] for output in outputs]
            else:
                text_outputs = outputs
                usages = None
            
            # 解析输出
            parsed_results = []
            for i, output in enumerate(text_outputs):
                parsed_data = self.parser.parse_concept_output(output)
                
                result = {
                    'concepts': parsed_data.get('concepts', []),
                    'relationships': parsed_data.get('relationships', []),
                    'raw_output': output
                }
                
                if self.config.extraction_mode == "hierarchical_concept":
                    result.update({
                        'specific_concepts': parsed_data.get('specific_concepts', []),
                        'general_concepts': parsed_data.get('general_concepts', []),
                        'abstract_concepts': parsed_data.get('abstract_concepts', []),
                        'hierarchical_relations': parsed_data.get('hierarchical_relations', [])
                    })
                
                if usages:
                    result['usage'] = usages[i]
                
                parsed_results.append(result)
            
            return parsed_results
            
        except Exception as e:
            print(f"Error in concept extraction: {e}")
            return [{"concepts": [], "relationships": []} for _ in text_batch]
    
    def run_extraction(self):
        """运行完整的概念提取流程"""
        print("🚀 Starting direct concept extraction...")
        
        # 加载数据集
        dataset = self.load_dataset()
        processed_data = self.processor.prepare_dataset(dataset["train"])
        
        print(f"📊 Processing {len(processed_data)} text chunks")
        
        # 创建输出文件
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        output_file = f"{self.config.output_directory}/concepts/direct_concepts_{timestamp}.json"
        
        all_concepts = []
        all_relationships = []
        
        # 批处理提取概念
        batch_size = self.config.batch_size_concept
        
        with open(output_file, "w", encoding="utf-8") as f:
            for i in tqdm(range(0, len(processed_data), batch_size), desc="Extracting concepts"):
                batch = processed_data[i:i + batch_size]
                text_batch = [item['text'] for item in batch]
                
                # 提取概念
                concept_results = self.extract_concepts_batch(text_batch)
                
                # 保存结果
                for j, result in enumerate(concept_results):
                    output_item = {
                        'chunk_id': batch[j]['chunk_id'],
                        'metadata': batch[j]['metadata'],
                        'text': batch[j]['text'],
                        'extracted_concepts': result
                    }
                    
<<<<<<< HEAD
                    # 收集所有概念和关系
                    all_concepts.extend(result.get('concepts', []))
                    all_relationships.extend(result.get('relationships', []))
=======
                    # 收集所有概念和关系，应用质量控制
                    concepts = result.get('concepts', [])
                    relationships = result.get('relationships', [])
                    
                    # 🔥 关系质量控制：去重和验证
                    filtered_relationships = self._filter_and_deduplicate_relationships(relationships)
                    
                    all_concepts.extend(concepts)
                    all_relationships.extend(filtered_relationships)
>>>>>>> 0ff854f19280eadc04f4289414abf37019510f1e
                    
                    if self.config.debug_mode:
                        print(f"Chunk {batch[j]['chunk_id']}: "
                              f"{len(result.get('concepts', []))} concepts, "
                              f"{len(result.get('relationships', []))} relationships")
                    
                    f.write(json.dumps(output_item, ensure_ascii=False) + "\n")
                    f.flush()
        
        print(f"✅ Concept extraction completed!")
        print(f"📄 Results saved to: {output_file}")
        print(f"📊 Total extracted: {len(all_concepts)} concepts, {len(all_relationships)} relationships")
        
        return output_file
    
    def create_concept_csv(self, concept_file: str):
        """将概念数据转换为CSV格式"""
        print("📊 Converting concepts to CSV...")
        
        csv_output_dir = f"{self.config.output_directory}/concept_csv"
        os.makedirs(csv_output_dir, exist_ok=True)
        
        concepts_csv = f"{csv_output_dir}/concepts_{self.config.filename_pattern}.csv"
        relationships_csv = f"{csv_output_dir}/relationships_{self.config.filename_pattern}.csv"
        
        all_concepts = []
        all_relationships = []
        
        # 读取概念文件
        with open(concept_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    extracted = data['extracted_concepts']
                    
                    # 收集概念
                    for concept in extracted.get('concepts', []):
                        concept['source_chunk'] = data['chunk_id']
                        all_concepts.append(concept)
                    
                    # 收集关系
                    for rel in extracted.get('relationships', []):
                        rel['source_chunk'] = data['chunk_id']
                        all_relationships.append(rel)
        
        # 写入概念CSV
        with open(concepts_csv, 'w', newline='', encoding='utf-8') as f:
            if all_concepts:
                fieldnames = ['name', 'type', 'abstraction_level', 'description', 'source_chunk']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for concept in all_concepts:
                    row = {field: concept.get(field, '') for field in fieldnames}
                    writer.writerow(row)
        
        # 写入关系CSV
        with open(relationships_csv, 'w', newline='', encoding='utf-8') as f:
            if all_relationships:
                fieldnames = ['source', 'target', 'relation', 'description', 'source_chunk']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for rel in all_relationships:
                    row = {field: rel.get(field, '') for field in fieldnames}
                    writer.writerow(row)
        
        print(f"✅ CSV files created:")
        print(f"   📄 Concepts: {concepts_csv}")
        print(f"   📄 Relationships: {relationships_csv}")
        
        return concepts_csv, relationships_csv 