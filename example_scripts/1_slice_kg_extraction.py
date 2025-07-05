from azure.ai.projects import AIProjectClient
from atlas_rag.kg_construction.triple_extraction import KnowledgeGraphExtractor
from atlas_rag.kg_construction.triple_config import ProcessingConfig
from azure.identity import DefaultAzureCredential
from atlas_rag.llm_generator import LLMGenerator
from configparser import ConfigParser
from argparse import ArgumentParser
from openai import OpenAI

parser = ArgumentParser(description="Generate knowledge graph slices from text data.")
parser.add_argument("--shard", type=int, help="Shard number to process.", default=0)
parser.add_argument("--total_shards", type=int, help="Total number of slices to process.", default=1)
args = parser.parse_args()
if __name__ == "__main__":
    keyword = 'hotpotqa_corpus_kg_input'
    config = ConfigParser()
    config.read('config.ini')
    # Added support for Azure Foundry. To use it, please do az-login in cmd first.
    model_name = "DeepSeek-V3-0324"

    connection = AIProjectClient(
        endpoint=config["urls"]["AZURE_URL"],
        credential=DefaultAzureCredential(),
    )
    client = connection.inference.get_azure_openai_client(api_version="2024-12-01-preview")
    # model_name = "Qwen/Qwen2.5-72B-Instruct"
    # client = OpenAI(
    # base_url="https://api.deepinfra.com/v1/openai",
    # api_key=config['settings']['DEEPINFRA_API_KEY'],
    # )
    # check if model name has / if yes then split and use -1
    triple_generator = LLMGenerator(client, model_name=model_name,)
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    output_directory = f'import/{keyword}/{model_name}'
    kg_extraction_config = ProcessingConfig(
        model_path=model_name,
        data_directory="benchmark_data",
        filename_pattern=keyword,
        batch_size_triple=4,
        batch_size_concept=64,
        output_directory=f"{output_directory}",
        current_shard_triple=args.slice,
        total_shards_triple=args.total_slices,
        record=True,
        max_new_tokens=4096
    )
    kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)
    kg_extractor.run_extraction()