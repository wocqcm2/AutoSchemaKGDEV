import pickle
import faiss
import os


def load_all_data(keyword, precompute_input_dir, include_events, include_concept, encoder_model_name):
    # Define paths for loading data
    node_index_path = f"{precompute_input_dir}/precompute/{keyword}_event{include_events}_concept{include_concept}_{encoder_model_name}_node_faiss.index"
    node_list_path = f"{precompute_input_dir}/precompute/{keyword}_event{include_events}_concept{include_concept}_node_list.pkl"
    edge_index_path = f"{precompute_input_dir}/precompute/{keyword}_event{include_events}_concept{include_concept}_{encoder_model_name}_edge_faiss.index"
    edge_list_path = f"{precompute_input_dir}/precompute/{keyword}_event{include_events}_concept{include_concept}_edge_list.pkl"
    node_embeddings_path = f"{precompute_input_dir}/precompute/{keyword}_event{include_events}_concept{include_concept}_{encoder_model_name}_node_embeddings.pkl"
    edge_embeddings_path = f"{precompute_input_dir}/precompute/{keyword}_event{include_events}_concept{include_concept}_{encoder_model_name}_edge_embeddings.pkl"
    text_embeddings_path = f"{precompute_input_dir}/precompute/{keyword}_{encoder_model_name}_text_embeddings.pkl"
    text_index_path = f"{precompute_input_dir}/precompute/{keyword}_text_faiss.index"
    
    text_list_path = f"{precompute_input_dir}/precompute/{keyword}_text_list.pkl"
    original_text_dict_path = f"{precompute_input_dir}/precompute/{keyword}_original_text_dict_with_node_id.pkl"
    
    
    
    # Check if all required files exist
    required_paths = [
    text_index_path,
    node_list_path, edge_list_path,
    text_list_path, original_text_dict_path,
    node_index_path, edge_index_path,
    text_embeddings_path, node_embeddings_path, edge_embeddings_path,
    
    ]

    # Check if all required files exist
    missing_paths = [path for path in required_paths if not os.path.exists(path)]

    if missing_paths:
        print("Missing files:")
        for path in missing_paths:
            print(path)
        raise FileNotFoundError("One or more required files are missing.")
    
    # Load data
    text_faiss_index = faiss.read_index(text_index_path)

    with open(node_list_path, "rb") as f:
        node_list = pickle.load(f)

    with open(edge_list_path, "rb") as f:
        edge_list = pickle.load(f)

    with open(text_list_path, "rb") as f:
        original_text_list = pickle.load(f)

    with open(original_text_dict_path, "rb") as f:
        original_text_dict_with_node_id = pickle.load(f)

    node_faiss_index = faiss.read_index(node_index_path)
    edge_faiss_index = faiss.read_index(edge_index_path)

    # Load text embeddings
    with open(text_embeddings_path, "rb") as f:
        text_embeddings = pickle.load(f)
    with open(node_embeddings_path, "rb") as f:
        node_embeddings = pickle.load(f)
    with open(edge_embeddings_path, "rb") as f:
        edge_embeddings = pickle.load(f)
    return {
        "text_faiss_index": text_faiss_index,
        "node_list": node_list,
        "edge_list": edge_list,
        "original_text_list": original_text_list,
        "original_text_dict_with_node_id": original_text_dict_with_node_id,
        "node_faiss_index": node_faiss_index,
        "edge_faiss_index": edge_faiss_index,
        "text_embeddings": text_embeddings,
        "node_embeddings": node_embeddings,
        "edge_embeddings": edge_embeddings,
    }