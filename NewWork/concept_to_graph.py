#!/usr/bin/env python3
"""
Concept to Graph Converter
将提取的概念转换为图结构
"""

import os
import csv
import json
import hashlib
import networkx as nx
import pickle
from typing import Dict, List, Set, Tuple
from collections import defaultdict, Counter
from tqdm import tqdm

from direct_concept_config import DirectConceptConfig


class ConceptGraphBuilder:
    """概念图构建器"""
    
    def __init__(self, config: DirectConceptConfig):
        self.config = config
        self.concept_counter = Counter()
        self.relationship_counter = Counter()
        self.concept_details = {}
        
    def normalize_concept_name(self, name: str) -> str:
        """标准化概念名称"""
        if not self.config.normalize_concept_names:
            return name
        
        # 转换为小写，去除多余空格
        normalized = ' '.join(name.lower().strip().split())
        return normalized
    
    def compute_concept_id(self, concept_name: str) -> str:
        """为概念生成唯一ID"""
        normalized_name = self.normalize_concept_name(concept_name)
        return hashlib.sha256(normalized_name.encode('utf-8')).hexdigest()[:16]
    
    def load_concepts_from_csv(self, concepts_csv: str, relationships_csv: str) -> Tuple[List[Dict], List[Dict]]:
        """从CSV文件加载概念和关系"""
        concepts = []
        relationships = []
        
        # 加载概念
        if os.path.exists(concepts_csv):
            with open(concepts_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['name'].strip():  # 过滤空概念
                        concepts.append(row)
        
        # 加载关系
        if os.path.exists(relationships_csv):
            with open(relationships_csv, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row['source'].strip() and row['target'].strip():  # 过滤空关系
                        relationships.append(row)
        
        return concepts, relationships
    
    def filter_concepts_by_frequency(self, concepts: List[Dict]) -> List[Dict]:
        """根据频率过滤概念"""
        if self.config.min_concept_frequency <= 1:
            return concepts
        
        # 统计概念频率
        concept_freq = Counter()
        for concept in concepts:
            normalized_name = self.normalize_concept_name(concept['name'])
            concept_freq[normalized_name] += 1
        
        # 过滤低频概念
        filtered_concepts = []
        for concept in concepts:
            normalized_name = self.normalize_concept_name(concept['name'])
            if concept_freq[normalized_name] >= self.config.min_concept_frequency:
                filtered_concepts.append(concept)
        
        print(f"📊 Filtered concepts: {len(concepts)} → {len(filtered_concepts)} "
              f"(min_frequency={self.config.min_concept_frequency})")
        
        return filtered_concepts
    
    def build_concept_graph(self, concepts: List[Dict], relationships: List[Dict]) -> nx.DiGraph:
        """构建概念图"""
        print("🔧 Building concept graph...")
        
        G = nx.DiGraph()
        
        # 过滤概念
        concepts = self.filter_concepts_by_frequency(concepts)
        
        # 添加概念节点
        concept_id_map = {}
        for concept in tqdm(concepts, desc="Adding concept nodes"):
            name = concept['name'].strip()
            if not name:
                continue
            
            normalized_name = self.normalize_concept_name(name)
            concept_id = self.compute_concept_id(name)
            concept_id_map[normalized_name] = concept_id
            
            # 添加节点属性
            node_attrs = {
                'id': name,
                'normalized_name': normalized_name,
                'type': concept.get('type', 'unknown'),
                'abstraction_level': concept.get('abstraction_level', 'unknown'),
                'description': concept.get('description', ''),
                'source_chunk': concept.get('source_chunk', ''),
                'node_type': 'concept'
            }
            
            G.add_node(concept_id, **node_attrs)
            
            # 记录概念详情
            if normalized_name not in self.concept_details:
                self.concept_details[normalized_name] = {
                    'names': set([name]),
                    'types': set([concept.get('type', 'unknown')]),
                    'levels': set([concept.get('abstraction_level', 'unknown')]),
                    'descriptions': set([concept.get('description', '')]),
                    'chunks': set([concept.get('source_chunk', '')])
                }
            else:
                self.concept_details[normalized_name]['names'].add(name)
                self.concept_details[normalized_name]['types'].add(concept.get('type', 'unknown'))
                self.concept_details[normalized_name]['levels'].add(concept.get('abstraction_level', 'unknown'))
                self.concept_details[normalized_name]['descriptions'].add(concept.get('description', ''))
                self.concept_details[normalized_name]['chunks'].add(concept.get('source_chunk', ''))
        
        # 添加关系边
        valid_relationships = 0
        for relationship in tqdm(relationships, desc="Adding relationship edges"):
            source_name = self.normalize_concept_name(relationship['source'].strip())
            target_name = self.normalize_concept_name(relationship['target'].strip())
            
            if source_name in concept_id_map and target_name in concept_id_map:
                source_id = concept_id_map[source_name]
                target_id = concept_id_map[target_name]
                
                # 避免自环
                if source_id != target_id:
                    edge_attrs = {
                        'relation': relationship.get('relation', 'related_to'),
                        'description': relationship.get('description', ''),
                        'source_chunk': relationship.get('source_chunk', ''),
                        'edge_type': 'concept_relation'
                    }
                    
                    G.add_edge(source_id, target_id, **edge_attrs)
                    valid_relationships += 1
        
        print(f"✅ Graph built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")
        print(f"📊 Valid relationships: {valid_relationships}/{len(relationships)}")
        
        return G
    
    def add_hierarchical_relations(self, G: nx.DiGraph, hierarchical_data: List[Dict]) -> nx.DiGraph:
        """添加层次关系"""
        if not self.config.include_hierarchical_relations:
            return G
        
        print("🔗 Adding hierarchical relations...")
        
        hierarchical_edges = 0
        for relation in tqdm(hierarchical_data, desc="Processing hierarchical relations"):
            child_name = self.normalize_concept_name(relation.get('child', '').strip())
            parent_name = self.normalize_concept_name(relation.get('parent', '').strip())
            relation_type = relation.get('relation_type', 'is_a')
            
            # 查找对应的节点ID
            child_id = None
            parent_id = None
            
            for node_id, node_attrs in G.nodes(data=True):
                if node_attrs.get('normalized_name') == child_name:
                    child_id = node_id
                if node_attrs.get('normalized_name') == parent_name:
                    parent_id = node_id
            
            if child_id and parent_id and child_id != parent_id:
                edge_attrs = {
                    'relation': relation_type,
                    'description': f'Hierarchical relation: {relation_type}',
                    'edge_type': 'hierarchical_relation'
                }
                
                G.add_edge(child_id, parent_id, **edge_attrs)
                hierarchical_edges += 1
        
        print(f"✅ Added {hierarchical_edges} hierarchical relations")
        return G
    
    def add_abstraction_level_edges(self, G: nx.DiGraph) -> nx.DiGraph:
        """根据抽象级别添加边"""
        if not self.config.include_abstraction_levels:
            return G
        
        print("🔗 Adding abstraction level edges...")
        
        # 按抽象级别分组节点
        level_groups = defaultdict(list)
        for node_id, node_attrs in G.nodes(data=True):
            level = node_attrs.get('abstraction_level', 'unknown')
            level_groups[level].append(node_id)
        
        abstraction_edges = 0
        
        # 在specific和general之间添加边
        if 'specific' in level_groups and 'general' in level_groups:
            for specific_node in level_groups['specific']:
                for general_node in level_groups['general']:
                    # 检查是否有相似的概念名称或类型
                    specific_attrs = G.nodes[specific_node]
                    general_attrs = G.nodes[general_node]
                    
                    if self._should_connect_by_abstraction(specific_attrs, general_attrs):
                        edge_attrs = {
                            'relation': 'abstraction_of',
                            'description': 'Abstraction level connection',
                            'edge_type': 'abstraction_relation'
                        }
                        G.add_edge(specific_node, general_node, **edge_attrs)
                        abstraction_edges += 1
        
        # 在general和abstract之间添加边
        if 'general' in level_groups and 'abstract' in level_groups:
            for general_node in level_groups['general']:
                for abstract_node in level_groups['abstract']:
                    general_attrs = G.nodes[general_node]
                    abstract_attrs = G.nodes[abstract_node]
                    
                    if self._should_connect_by_abstraction(general_attrs, abstract_attrs):
                        edge_attrs = {
                            'relation': 'abstraction_of',
                            'description': 'Abstraction level connection',
                            'edge_type': 'abstraction_relation'
                        }
                        G.add_edge(general_node, abstract_node, **edge_attrs)
                        abstraction_edges += 1
        
        print(f"✅ Added {abstraction_edges} abstraction level edges")
        return G
    
    def _should_connect_by_abstraction(self, lower_attrs: Dict, higher_attrs: Dict) -> bool:
        """判断两个不同抽象级别的概念是否应该连接"""
        # 简单的启发式规则：如果类型相同或名称相似
        lower_type = lower_attrs.get('type', '')
        higher_type = higher_attrs.get('type', '')
        
        lower_name = lower_attrs.get('normalized_name', '')
        higher_name = higher_attrs.get('normalized_name', '')
        
        # 类型匹配
        if lower_type == higher_type:
            return True
        
        # 名称包含关系
        if lower_name in higher_name or higher_name in lower_name:
            return True
        
        # 可以添加更复杂的语义相似性判断
        return False
    
    def save_graph_to_graphml(self, G: nx.DiGraph, output_path: str):
        """保存图为GraphML格式"""
        print(f"💾 Saving graph to GraphML: {output_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 清理属性，确保GraphML兼容
        cleaned_G = G.copy()
        for node_id, node_attrs in cleaned_G.nodes(data=True):
            for key, value in node_attrs.items():
                if isinstance(value, set):
                    node_attrs[key] = list(value)
                elif not isinstance(value, (str, int, float, bool)):
                    node_attrs[key] = str(value)
        
        for source, target, edge_attrs in cleaned_G.edges(data=True):
            for key, value in edge_attrs.items():
                if isinstance(value, set):
                    edge_attrs[key] = list(value)
                elif not isinstance(value, (str, int, float, bool)):
                    edge_attrs[key] = str(value)
        
        nx.write_graphml(cleaned_G, output_path)
        print(f"✅ Graph saved successfully!")
    
    def save_graph_to_pickle(self, G: nx.DiGraph, output_path: str):
        """保存图为pickle格式（用于后续处理）"""
        print(f"💾 Saving graph to pickle: {output_path}")
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(G, f)
        
        print(f"✅ Pickle saved successfully!")
    
    def generate_graph_statistics(self, G: nx.DiGraph) -> Dict:
        """生成图统计信息"""
        stats = {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'density': nx.density(G),
            'is_connected': nx.is_weakly_connected(G),
            'number_of_components': nx.number_weakly_connected_components(G)
        }
        
        # 节点类型统计
        node_types = defaultdict(int)
        abstraction_levels = defaultdict(int)
        
        for node_id, node_attrs in G.nodes(data=True):
            node_types[node_attrs.get('type', 'unknown')] += 1
            abstraction_levels[node_attrs.get('abstraction_level', 'unknown')] += 1
        
        stats['node_types'] = dict(node_types)
        stats['abstraction_levels'] = dict(abstraction_levels)
        
        # 边类型统计
        edge_types = defaultdict(int)
        for source, target, edge_attrs in G.edges(data=True):
            edge_types[edge_attrs.get('edge_type', 'unknown')] += 1
        
        stats['edge_types'] = dict(edge_types)
        
        # 度分布
        in_degrees = [G.in_degree(n) for n in G.nodes()]
        out_degrees = [G.out_degree(n) for n in G.nodes()]
        
        stats['avg_in_degree'] = sum(in_degrees) / len(in_degrees) if in_degrees else 0
        stats['avg_out_degree'] = sum(out_degrees) / len(out_degrees) if out_degrees else 0
        stats['max_in_degree'] = max(in_degrees) if in_degrees else 0
        stats['max_out_degree'] = max(out_degrees) if out_degrees else 0
        
        return stats
    
    def print_graph_statistics(self, G: nx.DiGraph):
        """打印图统计信息"""
        stats = self.generate_graph_statistics(G)
        
        print("\n" + "="*50)
        print("📊 CONCEPT GRAPH STATISTICS")
        print("="*50)
        print(f"Total Nodes: {stats['total_nodes']}")
        print(f"Total Edges: {stats['total_edges']}")
        print(f"Graph Density: {stats['density']:.4f}")
        print(f"Weakly Connected: {stats['is_connected']}")
        print(f"Number of Components: {stats['number_of_components']}")
        
        print(f"\n📋 Node Types:")
        for node_type, count in stats['node_types'].items():
            print(f"  {node_type}: {count}")
        
        print(f"\n🔍 Abstraction Levels:")
        for level, count in stats['abstraction_levels'].items():
            print(f"  {level}: {count}")
        
        print(f"\n🔗 Edge Types:")
        for edge_type, count in stats['edge_types'].items():
            print(f"  {edge_type}: {count}")
        
        print(f"\n📈 Degree Statistics:")
        print(f"  Average In-Degree: {stats['avg_in_degree']:.2f}")
        print(f"  Average Out-Degree: {stats['avg_out_degree']:.2f}")
        print(f"  Max In-Degree: {stats['max_in_degree']}")
        print(f"  Max Out-Degree: {stats['max_out_degree']}")
        print("="*50) 