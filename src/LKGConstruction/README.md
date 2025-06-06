# Large-scale Knowledge Graph Construction

## Step 1: Triple Extraction
```shell
# dataset can be wiki, pes2o, or cc
cd script

sh triple_extraction_{dataset}.sh
```

## Step 2: Json to CSV
```shell
# dataset can be wiki, pes2o, or cc
cd script

sh json2csv_{dataset}_with_text.sh
```

## Step 3: Conceptualization
```shell
cd script

sh concept_generation_{dataset}.sh

sh merge_{dataset}_concept.sh
```
## Step 4: Load Concept to CSV
```shell
# dataset can be wiki, pes2o, or cc
cd script
sh concept_to_csv_{dataset}.sh
```

## Step 5: Neo4j Server Installation
```shell
# dataset can be wiki, pes2o, or cc

# Download and install Neo4j server
cd script
sh get_neo4j_{dataset}.sh

# Start Neo4j server
sh start_neo4j_{dataset}.sh

```
!!! Before starting Neo4j, copy the `script/neo4j.conf` file to the conf directory of the Neo4j server. Then, update the following settings as needed: 1.Set dbms.default_database to the desired dataset name, such as wiki-csv-json-text, pes2o-csv-json-text, or cc-csv-json-text. 2.Configure the Bolt, HTTP, and HTTPS connectors according to your requirements.

## Step 6 Create Faiss Index for nodes in graph.

### Step 6.1 Add numeric id for each node
If you already have numeric_id included in the CSV files you can skip this step.

Otherwise, change the line to specify your desired KG name and the directory where your CSV is stored.
``` python
keyword = "en_simple_wiki_v0"
csv_dir = "./import"
```
and please run:
``` shell
python csv_add_column.py
```

### Step 6.2 Convert CSV to npy
Here we extract the embedding from CSV and convert it to npy file.

Change the line to specify your desired KG name and the directory where your CSV is stored.
``` python
keyword = "en_simple_wiki_v0"
csv_dir = "./import"
```
The npy will be stored at the same csv_dir.

and please run:
``` shell
python convert_csv2npy.py
```

### Step 6.3 Build Faiss Index from npy files
After the creation of .npy files, you can proceed to create faiss index for nodes in graph.

Please run:
``` shell
python create_index.py --keyword cc_en --npy_dir {your npy directory location}
```
The created index will be stored in the npy_dir.

## Step 7: Load CSV to Neo4j

For the wiki

```

../neo4j-server-wiki/bin/neo4j stop
../neo4j-server-wiki/bin/neo4j-admin database import full wiki-csv-json-text \
    --nodes=../import/text_nodes_en_simple_wiki_v0_from_json_with_numeric_id.csv \
    ../import/triple_nodes_en_simple_wiki_v0_from_json_without_emb_with_numeric_id.csv \
    ../import/concept_nodes_en_simple_wiki_v0_from_json_without_emb.csv \
    --relationships=../import/text_edges_en_simple_wiki_v0_from_json.csv \
    ../import/triple_edges_en_simple_wiki_v0_from_json_without_emb_full_concept_with_numeric_id.csv \
    ../import/concept_edges_en_simple_wiki_v0_from_json_without_emb.csv \
    --overwrite-destination \
    --multiline-fields=true \
    --id-type=string \
    --verbose --skip-bad-relationships=true
sh start_neo4j_wiki.sh
../neo4j-server-wiki/bin/cypher-shell -u neo4j -p admin2024 -a bolt://localhost:8011
```

For the pes2o

```

../neo4j-server-pes2o/bin/neo4j stop
../neo4j-server-pes2o/bin/neo4j-admin database import full pes2o-csv-json-text \
    --nodes=../import/text_nodes_pes2o_abstract_from_json_with_numeric_id.csv  ../import/triple_nodes_pes2o_abstract_from_json_without_emb_with_numeric_id.csv  ../import/concept_nodes_pes2o_abstract_from_json_without_emb.csv \
    --relationships=../import/text_edges_pes2o_abstract_from_json.csv  ../import/triple_edges_pes2o_abstract_from_json_without_emb_full_concept_with_numeric_id.csv  ../import/concept_edges_pes2o_abstract_from_json_without_emb.csv  \
    --overwrite-destination \
    --multiline-fields=true \
    --verbose --skip-bad-relationships=true --bad-tolerance=100000
sh start_neo4j_pes2o.sh
../neo4j-server-pes2o/bin/cypher-shell -u neo4j -p admin2024 -a bolt://localhost:8012
```

For the cc

```

../neo4j-server-cc/bin/neo4j stop
../neo4j-server-cc/bin/neo4j-admin database import full cc-csv-json-text \
    --nodes=../import/text_nodes_cc_en_from_json_with_numeric_id.csv  ../import/triple_nodes_cc_en_from_json_without_emb_with_numeric_id.csv  ../import/concept_nodes_cc_en_from_json_without_emb.csv\
    --relationships=../import/text_edges_cc_en_from_json.csv ../import/triple_edges_cc_en_from_json_without_emb_full_concept.csv  ../import/concept_edges_cc_en_from_json_without_emb.csv\
    --overwrite-destination \
    --multiline-fields=true \
    --verbose --skip-bad-relationships=true
sh start_neo4j_cc.sh
../neo4j-server-cc/bin/cypher-shell -u neo4j -p admin2024 -a bolt://localhost:8013
```

### Wiki

```shell
cd script
# Stop Neo4j if running
../neo4j-server-wiki/bin/neo4j stop

# Load the CSV files into Neo4j
../neo4j-server-wiki/bin/neo4j-admin database import full wiki-csv-json-text \
    --nodes=../import/text_nodes_en_simple_wiki_v0_from_json.csv \
    ../import/triple_nodes_en_simple_wiki_v0_from_json_without_emb.csv \
    ../import/concept_nodes_en_simple_wiki_v0_from_json_without_emb.csv \
    --relationships=../import/text_edges_en_simple_wiki_v0_from_json.csv \
    ../import/triple_edges_en_simple_wiki_v0_from_json_without_emb_full_concept.csv \
    ../import/concept_edges_en_simple_wiki_v0_from_json_without_emb.csv \
    --overwrite-destination \
    --multiline-fields=true \
    --id-type=string \
    --verbose --skip-bad-relationships=true

# Start Neo4j
sh start_neo4j_wiki.sh
```

### Pes2o

```shell
cd script
# Stop Neo4j if running
../neo4j-server-pes2o/bin/neo4j stop

# Load the CSV files into Neo4j
../neo4j-server-pes2o/bin/neo4j-admin database import full pes2o-csv-json-text \
    --nodes=../import/text_nodes_pes2o_abstract_from_json.csv  ../import/triple_nodes_pes2o_abstract_from_json_without_emb.csv  ../import/concept_nodes_pes2o_abstract_from_json_without_emb.csv \
    --relationships=../import/text_edges_pes2o_abstract_from_json.csv  ../import/triple_edges_pes2o_abstract_from_json_without_emb_full_concept.csv  ../import/concept_edges_pes2o_abstract_from_json_without_emb.csv  \
    --overwrite-destination \
    --multiline-fields=true \
    --verbose --skip-bad-relationships=true --bad-tolerance=100000

# Start Neo4j
sh start_neo4j_pes2o.sh
```

### CC

```shell
cd script
# Stop Neo4j if running
../neo4j-server-cc/bin/neo4j stop

# Load the CSV files into Neo4j
../neo4j-server-cc/bin/neo4j-admin database import full cc-csv-json-text \
    --nodes=../import/text_nodes_cc_en_from_json.csv  ../import/triple_nodes_cc_en_from_json_without_emb.csv  ../import/concept_nodes_cc_en_from_json_without_emb.csv\
    --relationships=../import/text_edges_cc_en_from_json.csv ../import/triple_edges_cc_en_from_json_without_emb_full_concept.csv  ../import/concept_edges_cc_en_from_json_without_emb.csv\
    --overwrite-destination \
    --multiline-fields=true \
    --verbose --skip-bad-relationships=true

# Start Neo4j
sh start_neo4j_cc.sh
```

