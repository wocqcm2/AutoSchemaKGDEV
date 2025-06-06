# Guide for hosting LKG Retriever
## ATLASRetriever
To host the server for the 3 Large KG mentioned in the paper, you need to

Create your own api_key.ini with path src/ATLASRetriever/config.ini and include the necessary api key

(you can replace with any other api key that is compatable with OpenAI package and please remember to change the base url in the code.)
```conf
[settings]
DEEPINFRA_API_KEY = <your api key>
```

And create faiss index with:
```shell
python create_faiss_index.py
```

Then you can run:
### For cc:
``` python
python app_neo4j_retriever_cc.py
```
### For pes2o:
``` python
python app_neo4j_retriever_pes2o.py
```
### For Wiki:
``` python
python app_neo4j_retriever_wiki.py
```
### For demo:
``` python
python app_neo4j_retriever_demo.py
```

### Accessing the API
After running the script you can access the api through localhost with port:

| KG name    | port |
| -------- | ------- |
| wiki  | 10087    |
| cc | 10088    |
| pes2o    | 10089   |
| demo    | 10090   |