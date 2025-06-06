import pandas as pd
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)

class IndexCreator:
    def __init__(self, model_name='all-MiniLM-L6-v2'):
        self.model = SentenceTransformer(model_name, truncate_dim=32)
        
    def create_text_index(self, text_file, output_path):
        """Create FAISS index for text nodes"""
        logging.info(f"Creating text index from {text_file}")
        df = pd.read_csv(text_file)
        
        # Generate embeddings for text content
        texts = df['original_text'].tolist()
        embeddings = self.model.encode(texts)
        
        # Create and save FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save index
        faiss.write_index(index, str(output_path))
        logging.info(f"Text index saved to {output_path}")
        
    def create_node_index(self, node_file, output_path):
        """Create FAISS index for nodes"""
        logging.info(f"Creating node index from {node_file}")
        df = pd.read_csv(node_file)
        
        # Generate embeddings for node names
        names = df['name:ID'].tolist()
        embeddings = self.model.encode(names)
        
        # Create and save FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save index
        faiss.write_index(index, str(output_path))
        logging.info(f"Node index saved to {output_path}")
        
    def create_edge_index(self, edge_file, output_path):
        """Create FAISS index for edges"""
        logging.info(f"Creating edge index from {edge_file}")
        df = pd.read_csv(edge_file)
        
        # Generate embeddings for edge types
        edge_types = df['relation'].tolist()
        embeddings = self.model.encode(edge_types)
        
        # Create and save FAISS index
        dimension = embeddings.shape[1]
        index = faiss.IndexFlatL2(dimension)
        index.add(embeddings.astype('float32'))
        
        # Save index
        faiss.write_index(index, str(output_path))
        logging.info(f"Edge index saved to {output_path}")

def main():
    # Initialize index creator
    creator = IndexCreator()
    
    # Define input and output paths
    current_dir = Path.cwd()
    import_dir = current_dir.parent.parent / "import" / "Dulce" / "triples_csv"
    
    # Create indices for Dulce dataset
    creator.create_text_index(
        import_dir / "text_nodes_Dulce_from_json.csv",
        import_dir / "text_nodes_Dulce_from_json.index"
    )
    
    creator.create_node_index(
        import_dir / "triple_nodes_Dulce_from_json_without_emb.csv",
        import_dir / "triple_nodes_Dulce_from_json.index"
    )
    
    creator.create_edge_index(
        import_dir / "triple_edges_Dulce_from_json_without_emb.csv",
        import_dir / "triple_edges_Dulce_from_json.index"
    )

if __name__ == "__main__":
    main() 