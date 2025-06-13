# AutoSchemaKG: A Knowledge Graph Construction Framework with Schema Generation and Knowledge Graph Completion

This repository contains the implementation of AutoSchemaKG, a novel framework for automatic knowledge graph construction that combines schema generation via conceptualization. The framework is designed to address the challenges of constructing high-quality knowledge graphs from unstructured text.

This project uses the following paper and data:

*   **Paper:** [Read the paper](https://arxiv.org/abs/2505.23628)
*   **Full Data:** [Download the dataset](https://hkustconnect-my.sharepoint.com/:f:/g/personal/jbai_connect_ust_hk/EgJCqoU91KpAlSSOi6dzgccB6SCL4YBpsCyEtGiRBV4WNg) (onedrive)
*   **Neo4j CSV Dumps:** [Download the dataset](https://huggingface.co/datasets/AlexFanWei/AutoSchemaKG) (huggingface dataset)


## AutoSchemaKG Overview

AutoSchemaKG introduces a two-stage approach:
1. **Knowledge Graph Triple Extraction**: Extract triples comprising entities and events from text by using LLMs 
2. **Schema Induction**: Automatically generate schema for the knowledge graph by using conceptualization and create semantic bridges between seemingly disparate information to enable zero-shot inferencing across domains


The framework achieves state-of-the-art performance on multiple benchmarks and demonstrates strong generalization capabilities across different domains.

## ATLAS Knowledge Graphs

ATLAS (Automated Triple Linking And Schema induction) is a family of knowledge graphs created through the AutoSchemaKG framework, which enables fully autonomous knowledge graph construction without predefined schemas. Here's a summary of what ATLAS is and how it works:

### Key Features of ATLAS Knowledge Graphs

- **Scale**: Consists of 900+ million nodes connected by 5.9 billion edges
- **Autonomous Construction**: Built without predefined schemas or manual intervention
- **Three Variants**: ATLAS-Wiki (from Wikipedia), ATLAS-Pes2o (from academic papers), and ATLAS-CC (from Common Crawl)




## Project Structure

```
AutoSchemaKG/
├── atlas_rag/                  # Main package directory
│   ├── kg_construction/        # Knowledge graph construction modules
│   ├── retriever/              # Retrieval components
│   ├── reader/                 # Reading and processing components
│   ├── utils/                  # Utility functions
│   ├── evaluation/             # Evaluation metrics and tools
│   └── billion/                # Large-scale KG processing
├── EvaluateKGC/                # Knowledge Graph Construction evaluation
├── EvaluateFactuality/         # Factual consistency evaluation
├── EvaluateGeneralTask/        # General task performance evaluation
├── neo4j_scripts/              # Neo4j database scripts
├── neo4j_api_host/             # Neo4j API hosting
├── import/                     # Data import directory
├── dist/                       # Distribution files
├── atlas_full_pipeline.ipynb   # Example for construct KG on new text data and doing RAG on it
├── atlas_multihopqa.ipynb      # Example for benchmarking the multi-hop QA datasets
└── atlas_billion_kg_usage.ipynb # Example for hosting and doing RAG with the constructed ATLAS-cc/ATLAS-wiki/ATLAS-pes2o
```

The project is organized into several key components:
- `atlas_rag/`: Core package containing the main functionality
- Evaluation directories for different aspects of the system
- Database-related scripts and API hosting
- Example notebooks demonstrating usage
- Import and distribution directories for data management

## TODO
- check Optimal Environment Version for running both RL and NV-Embed.

## Environment set up of cudnn 9.10.2
First change directory to your desired storing directory.
1. Install CUDA locally (for users without sudo access)
```bash
wget https://developer.download.nvidia.com/compute/cuda/12.4.1/local_installers/cuda_12.4.1_550.54.15_linux.run
chmod +x cuda_12.4.1_550.54.15_linux.run
./cuda_12.4.1_550.54.15_linux.run
```
During installation:
- Uncheck all driver-related options
- Change toolkit install path to your desired directory (e.g., `/data/httsangaj/drivers/cuda-12.4`)
- Uncheck "Create symbolic link from /usr/local/cuda"
- Uncheck "Create desktop menu shortcuts"
- Uncheck "Install manpage documents"

2. Install cudnn
```bash
wget https://developer.download.nvidia.com/compute/cudnn/redist/cudnn/linux-x86_64/cudnn-linux-x86_64-9.10.2.21_cuda12-archive.tar.xz
tar -xvf cudnn-linux-x86_64-9.10.2.21_cuda12-archive.tar.xz
mkdir cudnn-9.10.2
mv cudnn-linux-x86_64-9.10.2.21_cuda12-archive cudnn-9.10.2/
```

3. Set Environment Variables Permanent:
```bash
echo 'export CUDA_HOME=/data/httsangaj/drivers/cuda-12.4' >> ~/.bashrc
echo 'export PATH=$CUDA_HOME/bin:$PATH' >> ~/.bashrc
echo 'export CUDNN_HOME=/data/httsangaj/drivers/cudnn-9.10.2/cudnn-linux-x86_64-9.10.2.21_cuda12-archive' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=$CUDNN_HOME/lib:$CUDA_HOME/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
echo 'export CPATH=$CUDNN_HOME/include:$CPATH' >> ~/.bashrc
echo 'export LIBRARY_PATH=$CUDNN_HOME/lib:$LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

Verify installation:
```bash
nvcc --version  # Should show CUDA 12.4
```

## RL-setup
In order to install atlas-rag with gpu, it is recommended for you to first install pytorch-gpu with cuda and faiss-gpu, then you can run pip insatll atlas-rag to install necessary packages.

As faiss-gpu only support CUDA 11.4 and 12.1 for now. so,
1. Set up clean environment
```bash
conda remove -n atlas-rag-gpu-test --all
conda create -n atlas-rag-gpu-test python=3.10 pip
conda activate atlas-rag-gpu-test
```
2. Install pytorch 
(cuda 12.1)
```bash
conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia
```
(cuda 12.4)
``` bash
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
```
https://pytorch.org/get-started/locally/

3. For faiss-gpu (For cuda 11.8 and 12.1)
```bash
conda install -c pytorch -c nvidia faiss-gpu
```
For faiss-gpu (For cuda <= 12.5)
```bash
conda install -c pytorch -c rapidsai -c rapidsai-nightly -c conda-forge -c nvidia pytorch/label/nightly::faiss-gpu-cuvs 'cuda-version>=12.0,<=12.5'
```

Pip install dev_requirement (TODO to include smooth installation)
```bash
pip install -r dev_requirement.txt
```

4. Set up for verl
```bash
git clone https://github.com/NVIDIA/apex.git && \
cd apex && \
MAX_JOB=32 pip install -v --disable-pip-version-check --no-cache-dir --no-build-isolation --config-settings "--build-option=--cpp_ext" --config-settings "--build-option=--cuda_ext" ./
```
git clone verl for set up of verl
```
git clone https://github.com/volcengine/verl.git
```
cd to verl, then (for fsdp)
``` shell
USE_MEGATRON=0 bash scripts/install_vllm_sglang_mcore.sh
```

finally
```
cd verl
pip install --no-deps -e .
```
## Install atlas-rag through pip
```bash
pip install atlas-rag
```

If there exist error

replace manually 
```python
from apex import amp 
```
to
```python
from torch.cuda import amp
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




## Large Knowledge Graph Hosting and Retrieval Augmented Generation

This repository provides support for hosting and implementing Retrieval Augmented Generation (RAG) over our constructed knowledge graphs: `ATLAS-wiki`, `ATLAS-pes2o`, and `ATLAS-cc`. For detailed instructions on hosting and running these knowledge graphs, please refer to the `atlas_billion_kg_usage.ipynb` notebook. 

## Building New Knowledge Graphs and Implementing RAG

The `atlas_full_pipeline.ipynb` notebook demonstrates how to:
- Build new knowledge graphs using AutoschemaKG
- Implement Retrieval Augmented Generation on your custom knowledge graphs


## Multi-hop Question Answering Evaluation

To replicate our multi-hop question answering evaluation results on benchmark datasets:
- `MuSiQue`
- `HotpotQA` 
- `2WikiMultiHopQA`

Please follow the instructions in the `atlas_multihopqa.ipynb` notebook, which contains all necessary code and configuration details.

## General Evaluation


The framework includes comprehensive evaluation metrics across three dimensions:
- Knowledge Graph Quality  (`EvaluateKGC`)
- Factual Consistency on FELM (`EvaluateFactuality`)
- General Performance on MMLU (`EValuateGeneralTask`)

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

Dennis Hong Ting TSANG : httsangaj@connect.ust.hk

Haoyu Huang: haoyuhuang@link.cuhk.edu.hk

