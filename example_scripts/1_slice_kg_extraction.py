from azure.ai.projects import AIProjectClient
from atlas_rag import TripleGenerator, KnowledgeGraphExtractor, ProcessingConfig
from azure.identity import DefaultAzureCredential
from atlas_rag.reader import LLMGenerator
from configparser import ConfigParser
from argparse import ArgumentParser
from openai import OpenAI

parser = ArgumentParser(description="Generate knowledge graph slices from text data.")
parser.add_argument("--slice", type=int, help="Slice number to process.", default=0)
parser.add_argument("--total_slices", type=int, help="Total number of slices to process.", default=1)
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
    llm_generator = LLMGenerator(client=client, model_name=model_name)
    # check if model name has / if yes then split and use -1
    triple_generator = TripleGenerator(client, model_name=model_name,
    max_new_tokens = 4096,
    temperature = 0.1,
    frequency_penalty = 1.1)
    if '/' in model_name:
        model_name = model_name.split('/')[-1]
    output_directory = f'import/{keyword}/{model_name}'
    kg_extraction_config = ProcessingConfig(
        model_path=model_name,
        data_directory="benchmark_data",
        filename_pattern=keyword,
        batch_size=4,
        output_directory=f"{output_directory}",
        slice_current=args.slice,
        slice_total=args.total_slices,
        record=True
    )
    kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)
    kg_extractor.run_extraction()