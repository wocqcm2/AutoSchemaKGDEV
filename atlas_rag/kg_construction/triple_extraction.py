#!/usr/bin/env python3
"""
Knowledge Graph Extraction Pipeline
Extracts entities, relations, and events from text data using transformer models.
"""
import hashlib
import networkx as nx
import json
import os
import argparse
from datetime import datetime
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from pathlib import Path
import torch
from datasets import load_dataset
from tqdm import tqdm
import json_repair
from atlas_rag.utils.triple_generator import TripleGenerator
from atlas_rag.utils.json_2_csv import json2csv
from atlas_rag.kg_construction.concept_generation import generate_concept
from atlas_rag.utils.merge_csv import merge_csv_files
from atlas_rag.utils.csv_to_graphml import csvs_to_graphml
from atlas_rag.utils.concept_to_csv import all_concept_triples_csv_to_csv
from atlas_rag.utils.csv_add_column import add_csv_columns
from atlas_rag.utils.convert_csv2npy import convert_csv_to_npy
from atlas_rag.utils.compute_embedding import compute_embedding
from atlas_rag.utils.create_index import build_faiss_from_npy
from atlas_rag.retrieval.embedding_model import BaseEmbeddingModel
from atlas_rag.kg_construction.prompt import TRIPLE_INSTRUCTIONS



# Constants
TOKEN_LIMIT = 1024
INSTRUCTION_TOKEN_ESTIMATE = 200
CHAR_TO_TOKEN_RATIO = 3.5


@dataclass
class ProcessingConfig:
    """Configuration for text processing pipeline."""
    model_path: str
    data_directory: str
    filename_pattern: str
    batch_size: int = 16
    output_directory: str = "./generation_result_debug"
    slice_total: int = 1
    slice_current: int = 0
    use_8bit: bool = False
    debug_mode: bool = False
    resume_from: int = 0


class TextChunker:
    """Handles text chunking based on token limits."""
    
    def __init__(self, max_tokens: int = TOKEN_LIMIT, instruction_tokens: int = INSTRUCTION_TOKEN_ESTIMATE):
        self.max_tokens = max_tokens
        self.instruction_tokens = instruction_tokens
        self.char_ratio = CHAR_TO_TOKEN_RATIO
        
    def calculate_max_chars(self) -> int:
        """Calculate maximum characters per chunk."""
        available_tokens = self.max_tokens - self.instruction_tokens
        return int(available_tokens * self.char_ratio)
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks that fit within token limits."""
        max_chars = self.calculate_max_chars()
        chunks = []
        
        while len(text) > max_chars:
            chunks.append(text[:max_chars])
            text = text[max_chars:]
        
        if text:  # Add remaining text
            chunks.append(text)
            
        return chunks


class DatasetProcessor:
    """Processes and prepares dataset for knowledge graph extraction."""
    
    def __init__(self, config: ProcessingConfig):
        self.config = config
        self.chunker = TextChunker()
        
    def filter_language_content(self, sample: Dict[str, Any]) -> bool:
        """Check if content is in English."""
        metadata = sample.get("metadata", {})
        language = metadata.get("lang", "en")  # Default to English if not specified
        supported_languages = list(TRIPLE_INSTRUCTIONS.keys())
        return language in supported_languages
    
    
    def create_sample_chunks(self, sample: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create chunks from a single sample."""
        text_chunks = self.chunker.split_text(sample["text"])
        chunks = []
        
        for chunk_idx, chunk_text in enumerate(text_chunks):
            chunk_data = {
                "id": sample["id"],
                "text": chunk_text,
                "chunk_id": chunk_idx,
                "metadata": sample["metadata"]
            }
            chunks.append(chunk_data)
            
        return chunks
    
    def prepare_dataset(self, raw_dataset) -> List[Dict[str, Any]]:
        """Process raw dataset into chunks suitable for processing with generalized slicing."""
        
        processed_samples = []
        total_texts = len(raw_dataset)
        
        # Handle edge cases
        if total_texts == 0:
            print(f"No texts found for slice {self.config.slice_current+1}/{self.config.slice_total}")
            return processed_samples
        
        # Calculate base and remainder for fair distribution
        base_texts_per_slice = total_texts // self.config.slice_total
        remainder = total_texts % self.config.slice_total
        
        # Calculate start index
        if self.config.slice_current < remainder:
            start_idx = self.config.slice_current * (base_texts_per_slice + 1)
        else:
            start_idx = remainder * (base_texts_per_slice + 1) + (self.config.slice_current - remainder) * base_texts_per_slice
        
        # Calculate end index
        if self.config.slice_current < remainder:
            end_idx = start_idx + (base_texts_per_slice + 1)
        else:
            end_idx = start_idx + base_texts_per_slice
        
        # Ensure indices are within bounds
        start_idx = min(start_idx, total_texts)
        end_idx = min(end_idx, total_texts)
        
        print(f"Processing slice {self.config.slice_current+1}/{self.config.slice_total} "
            f"(texts {start_idx}-{end_idx-1} of {total_texts}, {end_idx - start_idx} documents)")
        
        # Process documents in assigned slice
        for idx in range(start_idx, end_idx):
            sample = raw_dataset[idx]
            
            # Filter by language
            if not self.filter_language_content(sample):
                print(f"Unsupported language in sample {idx}, skipping.")
                continue
                
            # Create chunks
            chunks = self.create_sample_chunks(sample)
            processed_samples.extend(chunks)
            
            # Debug mode early termination
            if self.config.debug_mode and len(processed_samples) >= 20:
                print("Debug mode: Stopping at 20 chunks")
                break
        
        print(f"Generated {len(processed_samples)} chunks for slice {self.config.slice_current+1}/{self.config.slice_total}")
        return processed_samples


class CustomDataLoader:
    """Custom data loader for knowledge graph extraction."""
    
    def __init__(self, dataset, processor: DatasetProcessor):
        self.raw_dataset = dataset
        self.processor = processor
        self.processed_data = processor.prepare_dataset(dataset)
        self.stage_to_prompt_dict = {
            "stage_1": "entity_relation",
            "stage_2": "event_entity",
            "stage_3": "event_relation"
        }
        
    def __len__(self) -> int:
        return len(self.processed_data)
    
    def create_batch_instructions(self, batch_data: List[Dict[str, Any]]) -> List[str]:
        messages_dict = {
            'stage_1': [],
            'stage_2': [],
            'stage_3': []
        }
        for item in batch_data:
            # get language
            language = item.get("metadata",{}).get("lang", "en")
            system_msg = TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['system'] 
            stage_1_msg = TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['entity_relation'] + TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['passage_start'] + '\n' + item["text"]
            stage_2_msg = TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['event_entity'] + TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['passage_start'] + '\n'+ item["text"]
            stage_3_msg = TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['event_relation'] + TRIPLE_INSTRUCTIONS.get(language, TRIPLE_INSTRUCTIONS["en"])['passage_start'] + '\n'+ item["text"]
            stage_one_message = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": stage_1_msg}
            ]
            stage_two_message = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": stage_2_msg}
            ]
            stage_three_message = [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": stage_3_msg}
            ]
            messages_dict['stage_1'].append(stage_one_message)
            messages_dict['stage_2'].append(stage_two_message)
            messages_dict['stage_3'].append(stage_three_message)
        
        return messages_dict
    
    def __iter__(self):
        """Iterate through batches."""
        batch_size = self.processor.config.batch_size
        start_idx = self.processor.config.resume_from * batch_size
        
        for i in tqdm(range(start_idx, len(self.processed_data), batch_size)):
            batch_data = self.processed_data[i:i + batch_size]
            
            # Prepare instructions
            instructions = self.create_batch_instructions(batch_data)
            
            # Extract batch information
            batch_ids = [item["id"] for item in batch_data]
            batch_metadata = [item["metadata"] for item in batch_data]
            batch_texts = [item["text"] for item in batch_data]
            
            yield instructions, batch_ids, batch_texts, batch_metadata


class OutputParser:
    """Parses model outputs and extracts structured data."""
    def __init__(self):
        pass

    def extract_structured_data(self, outputs: List[str]) -> List[List[Dict[str, Any]]]:
        """Extract structured data from model outputs."""
        results = []
        
        for output in outputs:
            parsed_data = json_repair.loads(output)
            results.append(parsed_data)
            
        return results


class KnowledgeGraphExtractor:
    """Main class for knowledge graph extraction pipeline."""
    
    def __init__(self, model:TripleGenerator, config: ProcessingConfig):
        self.config = config
        self.model = None
        self.parser = None
        self.model = model
        self.model_name = model.model_name
        self.parser = OutputParser()
    
    def load_dataset(self) -> Any:
        """Load and prepare dataset."""
        data_files = self.get_data_files()
        dataset_config = {"train": data_files}
        return load_dataset(self.config.data_directory, data_files=dataset_config["train"])
    
    def get_data_files(self) -> List[str]:
        """Get list of data files to process."""
        data_path = Path(self.config.data_directory)
        all_files = os.listdir(data_path)
        
        valid_files = [
            filename for filename in all_files
            if filename.startswith(self.config.filename_pattern) and
            (filename.endswith(".json.gz") or filename.endswith(".json"))
        ]
        
        print(f"Found data files: {valid_files}")
        return valid_files
    
    def generate_with_model(self, model_inputs: Dict[str, str], max_tokens: int =8192, stage = 1) -> List[str]:
        """Generate text using the model."""
        return self.model.generate(messages=model_inputs, max_tokens=max_tokens, stage=stage)
    
    def process_stage(self, instructions: Dict[str, str], stage = 1) -> Tuple[List[str], List[List[Dict[str, Any]]]]:
        """Process first stage: entity-relation extraction."""
        outputs = self.generate_with_model(instructions, stage = stage)
        structured_data = self.parser.extract_structured_data(outputs)
        return outputs, structured_data
    
    def create_output_filename(self) -> str:
        """Create output filename with timestamp and slice info."""
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        model_name_safe = self.config.model_path.replace("/", "_")
        
        filename = (f"{model_name_safe}_{self.config.filename_pattern}_output_"
                   f"{timestamp}_{self.config.slice_current + 1}_in_{self.config.slice_total}.json")
        
        extraction_dir = os.path.join(self.config.output_directory, "kg_extraction")
        os.makedirs(extraction_dir, exist_ok=True)
        
        return os.path.join(extraction_dir, filename)
    
    def prepare_result_dict(self, batch_data: Tuple, stage_outputs: Tuple, index: int) -> Dict[str, Any]:
        """Prepare result dictionary for a single sample."""
        ids, original_texts, metadata = batch_data
        (stage1_outputs, entity_relations), (stage2_outputs, event_entities), (stage3_outputs, event_relations) = stage_outputs
        
        result = {
            "id": ids[index],
            "metadata": metadata[index],
            "original_text": original_texts[index],
            "entity_relation_dict": entity_relations[index],
            "event_entity_relation_dict": event_entities[index],
            "event_relation_dict": event_relations[index],
            "output_stage_one": stage1_outputs[index],
            "output_stage_two": stage2_outputs[index],
            "output_stage_three": stage3_outputs[index]
        }
        
        # Handle date serialization
        if 'date_download' in result['metadata']:
            result['metadata']['date_download'] = str(result['metadata']['date_download'])
            
        return result
    
    def debug_print_result(self, result: Dict[str, Any]):
        """Print result for debugging."""
        for key, value in result.items():
            print(f"{key}: {value}")
            print("-" * 100)
    
    def run_extraction(self):
        """Run the complete knowledge graph extraction pipeline."""
        # Setup
        os.makedirs(self.config.output_directory+'/kg_extraction', exist_ok=True)
        dataset = self.load_dataset()
        
        if self.config.debug_mode:
            print("Debug mode: Processing only 20 samples")
        
        # Create data processor and loader
        processor = DatasetProcessor(self.config)
        data_loader = CustomDataLoader(dataset["train"], processor)
        
        output_file = self.create_output_filename()
        print(f"Model: {self.config.model_path}")
        
        batch_counter = 0
        
        with torch.no_grad():
            with open(output_file, "w") as output_stream:
                for batch in data_loader:
                    batch_counter += 1
                    messages_dict, batch_ids, batch_texts, batch_metadata = batch
                    
                    # Process all three stages
                    stage1_results = self.process_stage(messages_dict['stage_1'],1)
                    stage2_results = self.process_stage(messages_dict['stage_2'],2)
                    stage3_results = self.process_stage(messages_dict['stage_3'],3)
                    
                    # Combine results
                    batch_data = (batch_ids, batch_texts, batch_metadata)
                    stage_outputs = (stage1_results, stage2_results, stage3_results)
                    
                    # Write results
                    print(f"Processed {batch_counter} batches ({batch_counter * self.config.batch_size} chunks)")
                    for i in range(len(batch_ids)):
                        result = self.prepare_result_dict(batch_data, stage_outputs, i)
                        
                        if self.config.debug_mode:
                            self.debug_print_result(result)
                        
                        output_stream.write(json.dumps(result, ensure_ascii=False) + "\n")
                        output_stream.flush()

    def convert_json_to_csv(self):
        json2csv(dataset = self.config.filename_pattern, 
                 output_dir=f"{self.config.output_directory}/triples_csv",
                 data_dir=f"{self.config.output_directory}/kg_extraction"
                 )
    
    def generate_concept_csv_temp(self, batch_size: int = 64, **kwargs):
        generate_concept(
            model=self.model,
            input_file=f"{self.config.output_directory}/triples_csv/missing_concepts_{self.config.filename_pattern}_from_json.csv",
            input_triple_nodes_file=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            input_triple_edges_file=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb.csv",
            output_folder=f"{self.config.output_directory}/concepts",
            output_file="concept.json",
            logging_file=f"{self.config.output_directory}/concepts/logging.txt",
            batch_size=batch_size,
            **kwargs
        )
    
    def create_concept_csv(self):
        merge_csv_files(
            output_file=f"{self.config.output_directory}/triples_csv/{self.config.filename_pattern}_from_json_with_concept.csv",
            input_dir=f"{self.config.output_directory}/concepts",
        )
        all_concept_triples_csv_to_csv(
            node_file=f'{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv',
            edge_file=f'{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb.csv',
            concepts_file=f'{self.config.output_directory}/triples_csv/{self.config.filename_pattern}_from_json_with_concept.csv',
            output_node_file=f'{self.config.output_directory}/concept_csv/concept_nodes_{self.config.filename_pattern}_from_json_with_concept.csv',
            output_edge_file=f'{self.config.output_directory}/concept_csv/concept_edges_{self.config.filename_pattern}_from_json_with_concept.csv',
            output_full_concept_triple_edges=f'{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv',
        )
        
    def convert_to_graphml(self):
        csvs_to_graphml(
            triple_node_file=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            text_node_file=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json.csv",
            concept_node_file=f"{self.config.output_directory}/concept_csv/concept_nodes_{self.config.filename_pattern}_from_json_with_concept.csv",
            triple_edge_file=f"{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv",
            text_edge_file=f"{self.config.output_directory}/triples_csv/text_edges_{self.config.filename_pattern}_from_json.csv",
            concept_edge_file=f"{self.config.output_directory}/concept_csv/concept_edges_{self.config.filename_pattern}_from_json_with_concept.csv",
            output_file=f"{self.config.output_directory}/kg_graphml/{self.config.filename_pattern}_graph.graphml",
        )
    
    def add_numeric_id(self):
        add_csv_columns(
            node_csv=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            edge_csv=f"{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv",
            text_csv=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json.csv",
            node_with_numeric_id=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb_with_numeric_id.csv",
            edge_with_numeric_id=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_without_emb_with_numeric_id.csv",
            text_with_numeric_id=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json_with_numeric_id.csv",
        )

    def compute_embedding(self, encoder_model:BaseEmbeddingModel):

        compute_embedding(
            model=encoder_model,
            node_csv_without_emb=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_without_emb.csv",
            node_csv_file=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_with_emb.csv",
            edge_csv_without_emb=f"{self.config.output_directory}/concept_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept.csv",
            edge_csv_file=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept_with_emb.csv",
            text_node_csv_without_emb=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json.csv",
            text_node_csv=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json_with_emb.csv",
        )

    def create_faiss_index(self, index_type="HNSW,Flat"):
        """
        Create faiss index for the graph, for index type, see https://github.com/facebookresearch/faiss/wiki/Faiss-indexes

        "IVF65536_HNSW32,Flat" for 1M to 10M nodes

        "HNSW,Flat" for toy dataset

        """
        # Convert csv to npy
        convert_csv_to_npy(
            csv_path=f"{self.config.output_directory}/triples_csv/triple_nodes_{self.config.filename_pattern}_from_json_with_emb.csv",
            npy_path=f"{self.config.output_directory}/vector_index/triple_nodes_{self.config.filename_pattern}_from_json_with_emb.npy",
        )

        convert_csv_to_npy(
            csv_path=f"{self.config.output_directory}/triples_csv/text_nodes_{self.config.filename_pattern}_from_json_with_emb.csv",
            npy_path=f"{self.config.output_directory}/vector_index/text_nodes_{self.config.filename_pattern}_from_json_with_emb.npy",
        )

        convert_csv_to_npy(
            csv_path=f"{self.config.output_directory}/triples_csv/triple_edges_{self.config.filename_pattern}_from_json_with_concept_with_emb.csv",
            npy_path=f"{self.config.output_directory}/vector_index/triple_edges_{self.config.filename_pattern}_from_json_with_concept_with_emb.npy",
        )

        build_faiss_from_npy(
            index_type=index_type,
            index_path=f"{self.config.output_directory}/vector_index/triple_nodes_{self.config.filename_pattern}_from_json_with_emb_non_norm.index",
            npy_path=f"{self.config.output_directory}/vector_index/triple_nodes_{self.config.filename_pattern}_from_json_with_emb.npy",
        )

        build_faiss_from_npy(
            index_type=index_type,
            index_path=f"{self.config.output_directory}/vector_index/text_nodes_{self.config.filename_pattern}_from_json_with_emb_non_norm.index",
            npy_path=f"{self.config.output_directory}/vector_index/text_nodes_{self.config.filename_pattern}_from_json_with_emb.npy",
        )

        build_faiss_from_npy(
            index_type=index_type,
            index_path=f"{self.config.output_directory}/vector_index/triple_edges_{self.config.filename_pattern}_from_json_with_concept_with_emb_non_norm.index",
            npy_path=f"{self.config.output_directory}/vector_index/triple_edges_{self.config.filename_pattern}_from_json_with_concept_with_emb.npy",
        )

def parse_command_line_arguments() -> ProcessingConfig:
    """Parse command line arguments and return configuration."""
    parser = argparse.ArgumentParser(description="Knowledge Graph Extraction Pipeline")
    
    parser.add_argument("-m", "--model", type=str, required=True,
                       default="meta-llama/Meta-Llama-3-8B-Instruct",
                       help="Model path for knowledge extraction")
    parser.add_argument("--data_dir", type=str, default="your_data_dir",
                       help="Directory containing input data")
    parser.add_argument("--file_name", type=str, default="en_simple_wiki_v0",
                       help="Filename pattern to match")
    parser.add_argument("-b", "--batch_size", type=int, default=16,
                       help="Batch size for processing")
    parser.add_argument("--output_dir", type=str, default="./generation_result_debug",
                       help="Output directory for results")
    parser.add_argument("--total_slices", type=int, default=1,
                       help="Total number of data slices")
    parser.add_argument("--slice", type=int, default=0,
                       help="Current slice index")
    parser.add_argument("--bit8", action="store_true",
                       help="Use 8-bit quantization")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    parser.add_argument("--resume", type=int, default=0,
                       help="Resume from specific batch")
    
    args = parser.parse_args()
    
    return ProcessingConfig(
        model_path=args.model,
        data_directory=args.data_dir,
        filename_pattern=args.file_name,
        batch_size=args.batch_size,
        output_directory=args.output_dir,
        slice_total=args.total_slices,
        slice_current=args.slice,
        use_8bit=args.bit8,
        debug_mode=args.debug,
        resume_from=args.resume
    )


def main():
    """Main entry point for the knowledge graph extraction pipeline."""
    config = parse_command_line_arguments()
    extractor = KnowledgeGraphExtractor(config)
    extractor.run_extraction()


if __name__ == "__main__":
    main() 