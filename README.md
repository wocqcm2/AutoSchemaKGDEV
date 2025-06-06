# AutoSchemaKG: A Knowledge Graph Construction Framework with Schema Generation and Knowledge Graph Completion

This repository contains the implementation of AutoSchemaKG, a novel framework for automatic knowledge graph construction that combines schema generation and knowledge graph completion. The framework is designed to address the challenges of constructing high-quality knowledge graphs from unstructured text.

## Overview

AutoSchemaKG introduces a two-stage approach:
1. **Knowledge Graph Triple Extraction**: Extract triples comprising entities and events from text by using LLMs 
2. **Schema Induction**: Automatically generates schema for the knowledge graph by using conceptualization


The framework achieves state-of-the-art performance on multiple benchmarks and demonstrates strong generalization capabilities across different domains.



## Project Structure

```
AutoSchemaKG/
├── src/
│   ├── LKGConstruction/       # Knowledge Graph Construction components
│   ├── ATLASMultiHopQA/       # Multi-hop Question Answering implementation
│   └── ATLASRetriever/        # Retrieval components
├── EvaluateFactuality/     # Factuality evaluation metrics
├── EvaluateKGC/            # Knowledge Graph evaluation metrics
│   ├── InfoPreservation/    # Information preservation metrics
│   ├── SchemaQuality/       # Schema quality evaluation
│   └── TripleAccuracy/      # Triple accuracy metrics
└── EvaluationGeneral/       # General evaluation metrics
```

## Installation with Pip (Recommended)

```bash
pip install atlas-rag
```

### KG Construction with ATLAS

```python
from atlas_rag import TripleGenerator, KnowledgeGraphExtractor, ProcessingConfig
from openai import OpenAI
from transformers import pipeline
# client = OpenAI(api_key='<your_api_key>',base_url="<your_api_base_url>") 
# model_name = "meta-llama/llama-3.1-8b-instruct"

model_name = "meta-llama/Llama-3.1-8B-Instruct"
client = pipeline(
    "text-generation",
    model=model_name,
    device_map="auto",
)
keyword = 'Dulce'
output_directory = f'import/{keyword}'
triple_generator = TripleGenerator(client, model_name=model_name)
kg_extraction_config = ProcessingConfig(
      model_path=model_name,
      data_directory="tests",
      filename_pattern=keyword,
      batch_size=2,
      output_directory=f"{output_directory}",
)
kg_extractor = KnowledgeGraphExtractor(model=triple_generator, config=kg_extraction_config)

# Construct entity&event graph
kg_extractor.run_extraction() # Involved LLM Generation
# Convert Triples Json to CSV
kg_extractor.convert_json_to_csv()
# Concept Generation
kg_extractor.generate_concept_csv(batch_size=64) # Involved LLM Generation
# Create Concept CSV
kg_extractor.create_concept_csv()
# Convert csv to graphml for networkx
kg_extractor.convert_to_graphml()
```

### Retrieval Augumented Generation with ATLAS

## Installation from source

1. Clone the repository:
```bash
git clone https://github.com/yourusername/AutoSchemaKG.git
cd AutoSchemaKG
```

2. Create and activate the conda environment:
```bash
conda env create -f src/environment.yaml
conda activate autoschemakg
```

## Usage (for running source file directly)

### Auto Schema Knowledge Graph Construction
Change the working directory to the `src/LKGConstruction` folder and run the triple extraction script for your dataset. The dataset can be `wiki`, `pes2o`, or `cc`.

```shell
# dataset can be wiki, pes2o, or cc
cd script

sh triple_extraction_{dataset}.sh
```

JSON to CSV

Convert the extracted triples from JSON to CSV format. Ensure you are in the script directory and run the appropriate script for your dataset.

```shell
cd script

sh json2csv_{dataset}_with_text.sh
```

Conceptualization

Generate concepts for the extracted triples and merge them into the dataset. Run the following scripts in the script directory.

```shell
cd script

sh concept_generation_{dataset}.sh

sh merge_{dataset}_concept.sh
```

Load Concept to CSV

load the conceptualized data into CSV format. Run the script for your dataset in the script directory.

```shell
cd script

sh concept_to_csv_{dataset}.sh
```

Convert CSV to graphml

```shell
python csv_to_graphml.py 
    --triple_node_file ../import/triple_nodes_en_simple_wiki_v0_from_json_without_emb.csv 
    --text_node_file ../import/text_nodes_en_simple_wiki_v0_from_json.csv 
    --concept_node_file ../import/concept_nodes_en_simple_wiki_v0_from_json_without_emb.csv 
    --triple_edge_file ../import/triple_edges_en_simple_wiki_v0_from_json_without_emb_full_concept.csv 
    --text_edge_file ../import/text_edges_en_simple_wiki_v0_from_json.csv 
    --concept_edge_file ../import/concept_edges_en_simple_wiki_v0_from_json_without_emb.csv 
    --output_file ../import/output_graph.graphml
```

### Retrieval Augumented Generation for Multi-hop QA 

Change working directory to AutoSchemaKG/src/ATLASMultiHopQA
```
cd src/ATLASMultiHopQA
```

Download the constructed knowledge graph files if you prefer not to build them from scratch. Due to the inherent randomness in LLM decoding, there may be minor variations in the extracted triples each time. However, our experiments show these variations don't significantly impact performance.

Download these files:
- `2wikimultihopqa_concept.graphml`
- `hotpotqa_concept.graphml`
- `musique_concept.graphml`

Get them from [this link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jbai_connect_ust_hk/EgJCqoU91KpAlSSOi6dzgccB6SCL4YBpsCyEtGiRBV4WNg?e=88SYt0) and place them in your project directory.
```
AutoSchemaKG/src/ATLASMultiHopQA/processed_data/kg_graphml
```
Meanwhile, you need to explicitly create an output directory, otherwise will cause the following script to stop.

```
mkdir src/ATLASMultiHopQA/processed_data/precompute
```


For running the data processing from the graphs, use the following script for computing the embeddings of using ```all-MiniLM-L12-v2```. 
```
bash create_preload_data_for_each_config.bash
```

You can also uncomment the ```nvidia/NV-Embed-v2``` line in ```create_preload_data_for_each_config.bash``` for computing the nv-embed-v2 vectors, however this takes a long time. We provide the pre-computed in [this link](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jbai_connect_ust_hk/EgJCqoU91KpAlSSOi6dzgccB6SCL4YBpsCyEtGiRBV4WNg?e=88SYt0), with the file name ```nvembed.zip```. Unzip it to ```src/ATLASMultiHopQA/processed_data/precompute```.


Create your own config.ini with path ```src/ATLASMultiHopQA/config.ini``` and include the necessary api key

(you can replace with any other api key that is compatable with OpenAI package and please remember to change the base url in the code.)
You should put the following contents in your ```config.ini``` file
```conf
[settings]
DEEPINFRA_API_KEY = <your api key>
```

Then you can run the below command 
```shell

python ./opendomainqa/open_domain_qa.py --keyword hotpotqa --include_events --include_concept --encoder_model_name all-MiniLM-L12-v2 --method hipporag2

python ./opendomainqa/open_domain_qa.py --keyword hotpotqa --include_events --include_concept --encoder_model_name all-MiniLM-L12-v2 --method hipporag

python ./opendomainqa/open_domain_qa.py --keyword hotpotqa --include_events --include_concept --encoder_model_name all-MiniLM-L12-v2 --method tog
```
by altering parameters to choose between different graph config.

The keyword can be selected from ```hotpotqa``` ```2wikimultihopqa```, and ```musique``` for different dataset. 

For the method we support ```hipporag```, ```hipporag2```, and ```tog```. As this implementation include the source text nodes, the TOG method here is actually think-on-graph 2.0 instead of 1.0. 

If you have also download or processed with NV-Embedd-V2, you can replace ```--encoder_model_name``` of ```all-MiniLM-L12-v2``` with ```nvidia/NV-Embed-v2```, which can be used to replicate the results from our paper. 


### Large Knowledge Graph Extraction for ATLAS-wiki, ATLAS-pes2o, and ATLAS-cc


### Retrieval Augumented Generation for ATLAS-wiki, ATLAS-pes2o, and ATLAS-cc







## Evaluation

The framework includes comprehensive evaluation metrics across three dimensions:
- Knowledge Graph Quality 
- Factual Consistency on FELM
- General Performance on MMLU

Detailed evaluation procedures can be found in the respective evaluation directories.

## Citation

If you use this code in your research, please cite our paper:

```
@misc{bai2025autoschemakgautonomousknowledgegraph,
      title={AutoSchemaKG: Autonomous Knowledge Graph Construction through Dynamic Schema Induction from Web-Scale Corpora}, 
      author={Jiaxin Bai and Wei Fan and Qi Hu and Qing Zong and Chunyang Li and Hong Ting Tsang and Hongyu Luo and Yauwai Yim and Haoyu Huang and Xiao Zhou and Feng Qin and Tianshi Zheng and Xi Peng and Xin Yao and Huiwen Yang and Leijie Wu and Yi Ji and Gong Zhang and Renhai Chen and Yangqiu Song},
      year={2025},
      eprint={2505.23628},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.23628}, 
}
```





## Contact

Jiaxin Bai: jbai@connect.ust.hk 

Dennis: httsangaj@connect.ust.hk

Haoyu: haoyuhuang@link.cuhk.edu.hk


## Todo

- [ ] We are working on the link of our full KG data.
- [ ] We are working on restructuring the code for easier replication of results
- [ ] We are working on demo and tutorials for easier end-to-end RAG with our framework

Billion Level Graph Workflow:
- Triples Json Generation **Dulce Done**
- Json to csv **Dulce Done**
- CSV + concept -> all csv **Dulce Done**
- Calculate embedding -> to get both (without emb, with emb csv) **Dulce Done**
- With emb, faiss index  **Dulce Done**
- without emb, numeric id for neo4j import  **Dulce Done**




