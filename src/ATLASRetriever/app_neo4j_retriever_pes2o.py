import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import time
import uvicorn
from fastapi import FastAPI, HTTPException, Response
from typing import List, Literal, Optional, Union
from pydantic import BaseModel, Field
import json
import os
import json
import torch
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import configparser
from api_neo4j_retriever_model import load_api_start_up, LLMGenerator, MiniLM
import time
import re

rag_exemption_list = [
    """I will show you a question and a list of text segments. All the segments can be concatenated to form a complete answer to the question. Your task is to assess whether each text segment contains errors or not. \nPlease generate using the following format:\nAnswer: List the ids of the segments with errors (separated by commas). Please only output the ids, no more details. If all the segments are correct, output \"ALL_CORRECT\".\n\nHere is one example:\nQuestion: 8923164*7236571?\nSegments: \n1. The product of 8923164 and 7236571 is: 6,461,216,222,844\n2. So, 8923164 multiplied by 7236571 is equal to 6,461,216,222,844.\n\nBelow are your outputs:\nAnswer: 1,2\nIt means segment 1,2 contain errors.""",
    """I will show you a question and a list of text segments. All the segments can be concatenated to form a complete answer to the question. Your task is to determine whether each text segment contains factual errors or not. \nPlease generate using the following format:\nAnswer: List the ids of the segments with errors (separated by commas). Please only output the ids, no more details. If all the segments are correct, output \"ALL_CORRECT\".\n\nHere is one example:\nQuestion: A company offers a 10% discount on all purchases over $100. A customer purchases three items, each costing $80. Does the customer qualify for the discount?\nSegments: \n1. To solve this problem, we need to use deductive reasoning. We know that the company offers a 10% discount on purchases over $100, so we need to calculate the total cost of the customer's purchase.\n2. The customer purchased three items, each costing $80, so the total cost of the purchase is: 3 x $80 = $200.\n3. Since the total cost of the purchase is greater than $100, the customer qualifies for the discount. \n4. To calculate the discounted price, we can multiply the total cost by 0.1 (which represents the 10% discount): $200 x 0.1 = $20.\n5. So the customer is eligible for a discount of $20, and the final cost of the purchase would be: $200 - $20 = $180.\n6. Therefore, the customer would pay a total of $216 for the three items with the discount applied.\n\nBelow are your outputs:\nAnswer: 2,3,4,5,6\nIt means segment 2,3,4,5,6 contains errors.""",
]

mmlu_check_list = [
    """Given the following question and four candidate answers (A, B, C and D), choose the answer."""
]


config = configparser.ConfigParser()
config.read('config.ini')

import logging
from logging.handlers import RotatingFileHandler
log_file_path = './log/pes2o.log'
handler = RotatingFileHandler(
    log_file_path, 
    maxBytes=50 * 1024 * 1024,  # 50 MB limit
    backupCount=5               # Keep 5 backup log files
)

logging.basicConfig(
    handlers=[handler],
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

app = FastAPI()

MODEL_PATH = "meta-llama/Llama-3.3-70B-Instruct-Turbo"
device = torch.device('cuda:0')
hipporag_retriver, hipporag2_retriver, tog_retriver = None, None, None
llm_generator = None
@app.on_event("startup")
async def startup():
    global hipporag_retriver, hipporag2_retriver, tog_retriver, llm_generator
    sentence_encoder_model = SentenceTransformer('all-MiniLM-L12-v2', device=device, truncate_dim=32)
    sentence_encoder = MiniLM(sentence_encoder_model)
    client = OpenAI(api_key=config['settings']['DEEPINFRA_API_KEY'],base_url="https://api.deepinfra.com/v1/openai")
    llm_generator = LLMGenerator(None, client, False, True)
    hipporag2_retriver = load_api_start_up(sentence_encoder, llm_generator, "pes2o_abstract")
    

@app.on_event("shutdown")
async def shutdown():
    global hipporag_retriver, hipporag2_retriver, tog_retriver, llm_generator
    print("Shutting down the model...")
    # shut down the retrievers for the RAG model
    del hipporag_retriver
    del hipporag2_retriver
    del tog_retriver


class ModelCard(BaseModel):
    id: str
    object: str="model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "lcy"
    root: Optional[str] = None
    parent: Optional[str] = None
    permission: Optional[list] = None

class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelCard] = []

class ChatMessage(BaseModel):
    role: Literal["user", "system", "assistant"]
    content: str = None
    name: Optional[str] = None

class DeltaMessage(BaseModel):
    role: Optional[Literal["user", "assistant", "system"]] = None
    content: Optional[str] = None

class UsageInfo(BaseModel):
    prompt_tokens: int = 0
    total_tokens: int = 0
    completion_tokens: Optional[int] = 0


class ChatCompletionRequest(BaseModel):
    model: str
    messages: List[ChatMessage]
    temperature: Optional[float] = 0.8
    top_p: Optional[float] = 0.8
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False
    tools: Optional[Union[dict, List[dict]]] = None
    repetition_penalty: Optional[float] = 1.1
    retriever: Optional[str] = "hipporag2"
    knowledge_graph: Optional[str] =  "pes2o_abstract"
    retriever_config: Optional[dict] = {
            "topN":5, 
            "number_of_source_nodes_per_ner": 10, 
            "sampling_area": 250
            }

    class Config:
        extra = "allow"
    

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Literal["stop", "length", "function_call"]

class ChatCompletionResponseStreamChoice(BaseModel):
    delta: DeltaMessage
    finish_reason: Optional[Literal["stop", "length"]]
    index: int

class ChatCompletionResponse(BaseModel):
    model: str
    id: str
    object: Literal["chat.completion", "chat.completion.chunk"]
    choices: List[Union[ChatCompletionResponseChoice, ChatCompletionResponseStreamChoice]]
    created: Optional[int] = Field(default_factory=lambda: int(time.time()))
    usage: Optional[UsageInfo] = None

@app.get("/health")
async def health_check():
    return Response(status_code=200)

@app.get("/v1/models", response_model=ModelList)
async def list_models():
    model_card = ModelCard(
        id=MODEL_PATH
    )
    return ModelList(data=[model_card])

@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def create_chat_completion(request: ChatCompletionRequest):
    global hipporag_retriver, hipporag2_retriver, tog_retriver, llm_generator
    try:
        if len(request.messages) < 1 or request.messages[-1].role == "assistant":
            print(request)
            # raise HTTPException(status_code=400, detail="Invalid request")
        logging.info(f"Request: {request}")
        gen_params = dict(
            messages=request.messages,
            temperature=0.8,
            top_p=request.top_p,
            max_tokens=request.max_tokens or 1024,
            echo=False,
            stream=False,
            repetition_penalty=request.repetition_penalty,
            tools=request.tools,
            retriever=request.retriever,
            knowledge_graph=request.knowledge_graph,
            retriever_config=request.retriever_config
        )

        # rag related params


        # print("RETRIEVER_MSG: ", retriever_msg)

        retriever = request.retriever
        retriever_config = request.retriever_config
        knowledge_graph = request.knowledge_graph
        logging.info(f"Knowledge Graph: {knowledge_graph}")
        logging.info(f"Retriever: {retriever}")

        # request.messages = request.messages[1:]

        last_message = request.messages[-1]

        system_prompt = 'You are a helpful asssistant.'
        if last_message.role == 'assistant':
            question = request.messages[-2].content
        else:
            system_prompt = request.messages[-2].content
            question = request.messages[-1].content

        is_exemption = False 
        for exemption in rag_exemption_list:
            if exemption in question:
                is_exemption = True
                break  

        is_mmlu = False
        for exemption in mmlu_check_list:
            if exemption in question:
                is_mmlu = True
                break
        if is_mmlu:
            rag_text = question 
        else:
            parts = question.rsplit("Question:", 1)
            if len(parts) > 1:
                rag_text = "Question:" + parts[-1]
            else:
                rag_text = None
        logging.info(f"RAG_TEXT: {rag_text}, EXEMPTION: {is_exemption}, MMLU: {is_mmlu}, RETRIEVER: {retriever}")
        if retriever == 'hipporag' and not is_exemption:
            topN = retriever_config.get("topN", 5)
            number_of_source_nodes_per_ner = retriever_config.get("number_of_source_nodes_per_ner", 5)
            sampling_area = retriever_config.get("sampling_area", 100)
            hipporag_retriver.set_resources(knowledge_graph)
            hipporag_retriver.set_model('meta-llama/Llama-3.3-70B-Instruct-Turbo')
            if rag_text is None:
                hippo_passages = None
            else:
                hippo_passages, hippo_scores = hipporag_retriver.retrieve_passages(rag_text, topN, number_of_source_nodes_per_ner, sampling_area)
            if hippo_passages is None or len(hippo_passages) == 0:
                context = "No retrieved context, Please answer the question with your own knowledge."
            else:
                context = "\n".join([f"Passage {i+1}: {text}" for i, text in enumerate(reversed(hippo_passages))])
            # context = "HIPPO_RAG_RETRIEVED_CONTEXT" + str(knowledge_graph) + str(retriever_config)
            logging.info(f"HippoRAG: {context}")
        elif retriever == 'hipporag2' and not is_exemption:
            topN = retriever_config.get("topN", 5)
            number_of_source_nodes_per_ner = retriever_config.get("number_of_source_nodes_per_ner", 5)
            sampling_area = retriever_config.get("sampling_area", 100)
            hipporag2_retriver.set_resources(knowledge_graph)
            hipporag2_retriver.set_model('meta-llama/Llama-3.3-70B-Instruct-Turbo')
            if rag_text is None:
                hippo_passages = None
            else:
                hippo_passages, hippo_scores = hipporag2_retriver.retrieve_passages(rag_text, topN, number_of_source_nodes_per_ner, sampling_area)
            if hippo_passages is None or len(hippo_passages) == 0:
                context = "No retrieved context, Please answer the question with your own knowledge."
            else:
                context = "\n".join([f"Passage {i+1}: {text}" for i, text in enumerate(reversed(hippo_passages))])
            # context = "HIPPO_RAG_RETRIEVED_CONTEXT" + str(knowledge_graph) + str(retriever_config)
            logging.info(f"HippoRAG2: {context}")
            # context = "HIPPO_RAG2_RETRIEVED_CONTEXT"+ str(knowledge_graph) + str(retriever_config)
            
        elif retriever == 'tog' and not is_exemption:
            raise NotImplementedError("tog retriever is not implemented yet")
            tog_triples, tog_generated_answer = tog_retriver.retrieve(question, request.retriever_config, request.knowledge_graph)
            # context = "TOG_RETRIEVED_CONTEXT" + str(knowledge_graph) + str(retriever_config)

        if is_mmlu:
            rag_chat_content = [
                {
                    "role": "system",
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user",
                    "content": f"""Here is the context: {context} \n\n
                    If the context is not useful, you can answer the question with your own knowledge. \n {question}\nThink step by step. Your response should end with 'The answer is ([the_answer_letter])' where the [the_answer_letter] is one of A, B, C and D."""
                }
            ]
        elif not is_exemption:
            rag_chat_content = [
                {
                    "role": "system",
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user",
                    "content": f"""{question} Reference doc: {context}"""
                }
            ]
        else:
            rag_chat_content = [
                {
                    "role": "system",
                    "content": f"{system_prompt}"
                },
                {
                    "role": "user",
                    "content": f"""{question} """
                }
            ]
        logging.info(rag_chat_content)
        llm_generator.model_name = 'meta-llama/Meta-Llama-3.1-8B-Instruct'
        response = llm_generator.generate_with_custom_messages(
            custom_messages=rag_chat_content,
            max_new_tokens=gen_params["max_tokens"],
            temperature=gen_params["temperature"],
            frequency_penalty = 1.1
        )


        # print("INPUT: ", rag_chat_content)
        # print("OUTPUT: ", response)

        message = ChatMessage(
            role="assistant",
            content=response.strip()
        )
        # print("MESSAGE: ", message)
        choice_data = ChatCompletionResponseChoice(
            index=0,
            message=message,
            finish_reason="stop"
        )
        return ChatCompletionResponse(
            model=request.model,
            id="",
            object="chat.completion",
            choices=[choice_data]
        )
    except Exception as e:
        print("ERROR: ", e)
        print("Catched error")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run("app_neo4j_retriever_pes2o:app", host="0.0.0.0", port=10088, reload=False)
    