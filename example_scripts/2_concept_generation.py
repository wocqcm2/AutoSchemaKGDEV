from azure.ai.projects import AIProjectClient
from atlas_rag import TripleGenerator, KnowledgeGraphExtractor, ProcessingConfig
from azure.identity import DefaultAzureCredential
from atlas_rag.reader import LLMGenerator
from configparser import ConfigParser
from argparse import ArgumentParser

parser = ArgumentParser(description="Generate knowledge graph slices from text data.")
parser.add_argument("--slice", type=int, help="Slice number to process.", default=0)
parser.add_argument("--total_slices", type=int, help="Total number of slices to process.", default=1)
args = parser.parse_args()

if __name__ == "__main__":
    keyword = 'RomanceOfTheThreeKingdom-zh-CN'
    config = ConfigParser()
    config.read('config.ini')
    # Added support for Azure Foundry. To use it, please do az-login in cmd first.
    model_name = "DeepSeek-V3"
    connection = AIProjectClient(
        endpoint=config["urls"]["AFMR_AZURE_URL"],
        credential=DefaultAzureCredential(),
    )
    client = connection.inference.get_azure_openai_client(api_version="2024-12-01-preview")
    llm_generator = LLMGenerator(client=client, model_name=model_name)
    output_directory = f'import/{keyword}'
    triple_generator = TripleGenerator(client, model_name=model_name,
    max_new_tokens = 4096,
    temperature = 0.1,
    frequency_penalty = 1.1)
    kg_extraction_config = ProcessingConfig(
        model_path=model_name,
        data_directory="example_data",
        filename_pattern=keyword,
        batch_size=4,
        output_directory=f"{output_directory}",
    )
    kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)
    kg_extractor.convert_json_to_csv()
    kg_extractor.generate_concept_csv_temp(batch_size=64,language='zh-CN')
    kg_extractor.create_concept_csv()