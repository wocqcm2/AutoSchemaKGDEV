import csv


from sentence_transformers import SentenceTransformer
from transformers import AutoModel

def compute_embedding(model, node_csv_without_emb, node_csv_file, edge_csv_without_emb, edge_csv_file, text_node_csv_without_emb, text_node_csv):


    with open(node_csv_without_emb, "r") as csvfile_node:
        with open(node_csv_file, "w") as csvfile_node_emb:
            reader_node = csv.reader(csvfile_node)

            # the reader has [name:ID,type,concepts,synsets,:LABEL]
            writer_node = csv.writer(csvfile_node_emb)
            writer_node.writerow(["name:ID", "type", "file_id", "concepts", "synsets", "embedding:STRING", ":LABEL"])

            # the encoding will be processed in batch of 1024
            batch_size = 2048
            batch_nodes = []
            batch_rows = []
            for row in reader_node:
                if row[0] == "name:ID":
                    continue
                batch_nodes.append(row[0])
                batch_rows.append(row)
            
                if len(batch_nodes) == batch_size:
                    node_embeddings = model.encode(batch_nodes, batch_size=batch_size, show_progress_bar=False)
                    node_embedding_dict = dict(zip(batch_nodes, node_embeddings))
                    for row in batch_rows:
                       
                        new_row = [row[0], row[1], "", row[2], row[3], node_embedding_dict[row[0]].tolist(), row[4]]
                        writer_node.writerow(new_row)
                        
                    
                    
                    batch_nodes = []
                    batch_rows = []

            if len(batch_nodes) > 0:
                node_embeddings = model.encode(batch_nodes, batch_size=batch_size, show_progress_bar=False)
                node_embedding_dict = dict(zip(batch_nodes, node_embeddings))
                for row in batch_rows:
                    new_row = [row[0], row[1], "", row[2], row[3], node_embedding_dict[row[0]].tolist(), row[4]]
                    writer_node.writerow(new_row)
                batch_nodes = []
                batch_rows = []
    

    with open(edge_csv_without_emb, "r") as csvfile_edge:
        with open(edge_csv_file, "w") as csvfile_edge_emb:
            reader_edge = csv.reader(csvfile_edge)
            # [":START_ID",":END_ID","relation","concepts","synsets",":TYPE"]
            writer_edge = csv.writer(csvfile_edge_emb)
            writer_edge.writerow([":START_ID", ":END_ID", "relation", "file_id", "concepts", "synsets", "embedding:STRING", ":TYPE"])

            # the encoding will be processed in batch of 4096
            batch_size = 2048
            batch_edges = []
            batch_rows = []
            for row in reader_edge:
                if row[0] == ":START_ID":
                    continue
                batch_edges.append(" ".join([row[0], row[2], row[1]]))
                batch_rows.append(row)
            
                if len(batch_edges) == batch_size:
                    edge_embeddings = model.encode(batch_edges, batch_size=batch_size, show_progress_bar=False)
                    edge_embedding_dict = dict(zip(batch_edges, edge_embeddings))
                    for row in batch_rows:
                        new_row = [row[0], row[1], row[2], "", row[3], row[4], edge_embedding_dict[" ".join([row[0], row[2], row[1]])].tolist(), row[5]]
                        writer_edge.writerow(new_row)
                    batch_edges = []
                    batch_rows = []

            if len(batch_edges) > 0:
                edge_embeddings = model.encode(batch_edges, batch_size=batch_size, show_progress_bar=False)
                edge_embedding_dict = dict(zip(batch_edges, edge_embeddings))
                for row in batch_rows:
                    new_row = [row[0], row[1], row[2], "", row[3], row[4], edge_embedding_dict[" ".join([row[0], row[2], row[1]])].tolist(), row[5]]    
                    writer_edge.writerow(new_row)
                batch_edges = []
                batch_rows = []
    

    with open(text_node_csv_without_emb, "r") as csvfile_text_node:
        with open(text_node_csv, "w") as csvfile_text_node_emb:
            reader_text_node = csv.reader(csvfile_text_node)
            # [text_id:ID,original_text,:LABEL]
            writer_text_node = csv.writer(csvfile_text_node_emb)

            writer_text_node.writerow(["text_id:ID", "original_text", ":LABEL", "embedding:STRING"])

            # the encoding will be processed in batch of 2048
            batch_size = 2048
            batch_text_nodes = []
            batch_rows = []
            for row in reader_text_node:
                if row[0] == "text_id:ID":
                    continue
                
                batch_text_nodes.append(row[0])
                batch_rows.append(row)
            
                if len(batch_text_nodes) == batch_size:
                    text_node_embeddings = model.encode(batch_text_nodes, batch_size=batch_size, show_progress_bar=False)
                    text_node_embedding_dict = dict(zip(batch_text_nodes, text_node_embeddings))
                    for row in batch_rows:
                        embedding  = text_node_embedding_dict[row[0]].tolist()
                        new_row = [row[0], row[1], row[2], embedding]
                        writer_text_node.writerow(new_row)

                    batch_text_nodes = []
                    batch_rows = []

            if len(batch_text_nodes) > 0:
                text_node_embeddings = model.encode(batch_text_nodes, batch_size=batch_size, show_progress_bar=False)
                text_node_embedding_dict = dict(zip(batch_text_nodes, text_node_embeddings))
                for row in batch_rows:
                    embedding  = text_node_embedding_dict[row[0]].tolist()
                    new_row = [row[0], row[1], row[2], embedding]
                    
                    writer_text_node.writerow(new_row)
                batch_text_nodes = []
                batch_rows = []


if __name__ == "__main__":
    model = SentenceTransformer("all-MiniLM-L12-v2")


    output_directory = "/home/jbai/AutoSchemaKG/import/Dulce"
    filename_pattern = "Dulce"

    compute_embedding(
            model=model,
            node_csv_without_emb=f"{output_directory}/triples_csv/triple_nodes_{filename_pattern}_from_json_without_emb.csv",
            node_csv_file=f"{output_directory}/triples_csv/triple_nodes_{filename_pattern}_from_json_with_emb.csv",
            edge_csv_without_emb=f"{output_directory}/concept_csv/triple_edges_{filename_pattern}_from_json_with_concept.csv",
            edge_csv_file=f"{output_directory}/triples_csv/triple_edges_{filename_pattern}_from_json_with_concept_with_emb.csv",
            text_node_csv_without_emb=f"{output_directory}/triples_csv/text_nodes_{filename_pattern}_from_json.csv",
            text_node_csv=f"{output_directory}/triples_csv/text_nodes_{filename_pattern}_from_json_with_emb.csv",
        )
   