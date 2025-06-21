from tqdm import tqdm
import random
import logging
import csv
import os
import hashlib
import re
from atlas_rag.utils.triple_generator import TripleGenerator
from atlas_rag.utils.csv_to_graphml import csvs_to_temp_graphml, get_node_id
from atlas_rag.kg_construction.prompt import CONCEPT_INSTRUCTIONS
# Increase the field size limit
csv.field_size_limit(10 * 1024 * 1024)  # 10 MB limit



def build_batch_data(sessions, batch_size):
    batched_sessions = []
    for i in range(0, len(sessions), batch_size):
        batched_sessions.append(sessions[i:i+batch_size])
    return batched_sessions

# Function to compute a hash ID from text
def compute_hash_id(text):
    # Use SHA-256 to generate a hash
    hash_object = hashlib.sha256(text.encode('utf-8'))
    return hash_object.hexdigest()  # Return hash as a hex string

def convert_attribute(value):
    """ Convert attributes to GDS-compatible types. """
    if isinstance(value, list):
        return [str(v) for v in value]
    elif isinstance(value, (int, float)):
        return value
    else:
        return str(value)

def clean_text(text):
    # remove NUL as well
    
    new_text = text.replace("\n", " ").replace("\r", " ").replace("\t", " ").replace("\v", " ").replace("\f", " ").replace("\b", " ").replace("\a", " ").replace("\e", " ").replace(";", ",")
    new_text = new_text.replace("\x00", "")
    new_text = re.sub(r'\s+', ' ', new_text).strip()

    return new_text

def remove_NUL(text):
    return text.replace("\x00", "")

def build_batched_events(all_node_list, batch_size):
    "The types are in Entity Event Relation"
    event_nodes = [node[0] for node in all_node_list if node[1].lower() == "event"]
    batched_events = []

    for i in range(0, len(event_nodes), batch_size):
        batched_events.append(event_nodes[i:i+batch_size])
    
    return batched_events

def build_batched_entities(all_node_list, batch_size):
    
    entity_nodes = [node[0] for node in all_node_list if node[1].lower() == "entity"]
    batched_entities = []

    for i in range(0, len(entity_nodes), batch_size):
        batched_entities.append(entity_nodes[i:i+batch_size])
    
    return batched_entities

def build_batched_relations(all_node_list, batch_size):

    relations = [node[0] for node in all_node_list if node[1].lower() == "relation"]
    # relations = list(set(relations))
    batched_relations = []

    for i in range(0, len(relations), batch_size):
        batched_relations.append(relations[i:i+batch_size])
    
    return batched_relations

def batched_inference(model:TripleGenerator, inputs):
    responses = model.generate(inputs)
    answers = []
    for i in range(len(responses)):
        answer = responses[i]
        answers.append([x.strip().lower() for x in answer.split(",")])
    
    return answers

def load_data_with_shard(input_file, shard_idx, num_shards):

    with open(input_file, "r") as f:
        csv_reader = list(csv.reader(f))
    
    # data = csv_reader  
    data = csv_reader[1:]
    # Random shuffle the data before splitting into shards
    random.shuffle(data)
    
    total_lines = len(data)
    lines_per_shard = (total_lines + num_shards - 1) // num_shards
    start_idx = shard_idx * lines_per_shard
    end_idx = min((shard_idx + 1) * lines_per_shard, total_lines)
    
    return data[start_idx:end_idx]

def conceptualize(model: TripleGenerator,
                  input_file = 'processed_data/triples_csv', 
                  output_folder = 'processed_data/triples_conceptualized', 
                  output_file = 'output.json', 
                  logging_file = 'processed_data/logging.txt', 
                  sample_num=None, 
                  batch_size=32, 
                  shard=0, 
                  num_shards=1,
                  **kwargs):
    """
    Encapsulates the logic for parsing arguments, setting up the environment, and calling the generate function.

    Args:
        model (TripleGenerator): The model to use for generating concepts.
        input_file (str): Path to the input file.
        output_folder (str): Path to the output folder.
        output_file (str): Path to the output file.
        logging_file (str): Path to the logging file.
        sample_num (int): Sample number of sessions.
        batch_size (int): Number of sessions processed at the same time.
        shard (int): Shard id.
        num_shards (int): Total number of shards.
    """
    # Set random seed for reproducibility

    # Print environment information
    # print("\n>>>Shard id:", shard)
    # print("CUDA Device Count:", torch.cuda.device_count())
    # print("VISIBLE DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES", None))
    # print("Current CUDA Device:", torch.cuda.current_device())

    # Call the generate function with the provided arguments
    generate_concept(model,input_file=input_file,
             output_folder=output_folder,
             output_file=output_file,
             logging_file=logging_file,
             sample_num=sample_num,
             batch_size=batch_size,
             shard=shard,
             num_shards=num_shards,
             **kwargs)

def generate_concept(model: TripleGenerator,
            input_file = 'processed_data/triples_csv', 
            input_triple_nodes_file = 'processed_data/triple_nodes.csv',
            input_triple_edges_file = 'processed_data/triple_edges.csv',
            output_folder = 'processed_data/triples_conceptualized', 
            output_file = 'output.json', 
            logging_file = 'processed_data/logging.txt', 
            sample_num=None, 
            batch_size=32, 
            shard=0, 
            num_shards=1,
            **kwargs):
    log_dir = os.path.dirname(logging_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir)

    # Create the log file if it doesn't exist
    if not os.path.exists(logging_file):
        open(logging_file, 'w').close()

    language = kwargs.get('language', 'en')

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logging_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logging.getLogger().addHandler(file_handler)
    
    temp_kg = csvs_to_temp_graphml(input_triple_nodes_file, input_triple_edges_file)

    # read data
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    all_missing_nodes = load_data_with_shard(
        input_file,
        shard_idx=shard,
        num_shards=num_shards
    )
        
    batched_events = build_batched_events(all_missing_nodes, batch_size)
    batched_entities = build_batched_entities(all_missing_nodes, batch_size)
    batched_relations = build_batched_relations(all_missing_nodes, batch_size)
    
    all_batches = []
    all_batches.extend(('event', batch) for batch in batched_events)
    all_batches.extend(('entity', batch) for batch in batched_entities)
    all_batches.extend(('relation', batch) for batch in batched_relations)
    
    print("all_batches", len(all_batches))



    
    output_file = output_folder + f"/{output_file.rsplit('.', 1)[0]}_shard_{shard}.csv"
    with open(output_file, "w") as file:
        csv_writer = csv.writer(file)
        csv_writer.writerow(["node", "conceptualized_node", "node_type"])

        # for batch_type, batch in tqdm(all_batches, total=total_batches, desc="Generating concepts"):
        # don't use tqdm for now
        for batch_type, batch in tqdm(all_batches, desc="Shard_{}".format(shard)):
            # print("batch_type", batch_type)
            # print("batch", batch)
            replace_context_token = None
            if batch_type == 'event':
                template = CONCEPT_INSTRUCTIONS[language]['event']
                node_type = 'event'
                replace_token = '[EVENT]'
            elif batch_type == 'entity':
                template = CONCEPT_INSTRUCTIONS[language]['entity']
                node_type = 'entity'
                replace_token = '[ENTITY]'
                replace_context_token = '[CONTEXT]'
            elif batch_type == 'relation':
                template = CONCEPT_INSTRUCTIONS[language]['relation']
                node_type = 'relation'
                replace_token = '[RELATION]'

            inputs = []
            for node in batch:
                # sample node from given node and replace context token.
                if replace_context_token:
                    node_id = get_node_id(node)
                    entity_predecessors = list(temp_kg.predecessors(node_id))
                    entity_successors = list(temp_kg.successors(node_id))

                    context = ""

                    if len(entity_predecessors) > 0:
                        random_two_neighbors = random.sample(entity_predecessors, min(1, len(entity_predecessors)))
                        context += ", ".join([f"{temp_kg.nodes[neighbor]['id']} {temp_kg[neighbor][node_id]['relation']}" for neighbor in random_two_neighbors])
                    
                    if len(entity_successors) > 0:
                        random_two_neighbors = random.sample(entity_successors, min(1, len(entity_successors)))
                        context += ", ".join([f"{temp_kg[node_id][neighbor]['relation']} {temp_kg.nodes[neighbor]['id']}" for neighbor in random_two_neighbors])
                    
                    prompt = template.replace(replace_token, node).replace(replace_context_token, context)
                else:
                    prompt = template.replace(replace_token, node)
                constructed_input = [
                    {"role": "system", "content": "You are a helpful AI assistant."},
                    {"role": "user", "content": f"{prompt}"},
                ]
                inputs.append(constructed_input)

            try:
                # print("inputs", inputs)
                answers = batched_inference(model, inputs)
                # print("answers", answers)
            except Exception as e:
                logging.error(f"Error processing {batch_type} batch: {e}")
                raise e
            # try:
            #     answers = batched_inference(llm, sampling_params, inputs)
            # except Exception as e:
            #     logging.error(f"Error processing {batch_type} batch: {e}")
            #     continue

            for node, answer in zip(batch, answers):
                # print(node, answer, node_type)
                csv_writer.writerow([node, ", ".join(answer), node_type])
                file.flush()
    # count unique conceptualized nodes
    conceptualized_nodes = []

    conceptualized_events = []
    conceptualized_entities = []
    conceptualized_relations = []

    with open(output_file, "r") as file:
        reader = csv.reader(file)
        next(reader)
        for row in reader:
            conceptualized_nodes.extend(row[1].split(","))
            if row[2] == "event":
                conceptualized_events.extend(row[1].split(","))
            elif row[2] == "entity":
                conceptualized_entities.extend(row[1].split(","))
            elif row[2] == "relation":
                conceptualized_relations.extend(row[1].split(","))
    
    conceptualized_nodes = [x.strip() for x in conceptualized_nodes]
    conceptualized_events = [x.strip() for x in conceptualized_events]
    conceptualized_entities = [x.strip() for x in conceptualized_entities]
    conceptualized_relations = [x.strip() for x in conceptualized_relations]

    unique_conceptualized_nodes = list(set(conceptualized_nodes))
    unique_conceptualized_events = list(set(conceptualized_events))
    unique_conceptualized_entities = list(set(conceptualized_entities))
    unique_conceptualized_relations = list(set(conceptualized_relations))

    print(f"Number of unique conceptualized nodes: {len(unique_conceptualized_nodes)}")
    print(f"Number of unique conceptualized events: {len(unique_conceptualized_events)}")
    print(f"Number of unique conceptualized entities: {len(unique_conceptualized_entities)}")
    print(f"Number of unique conceptualized relations: {len(unique_conceptualized_relations)}")

    return 
    


