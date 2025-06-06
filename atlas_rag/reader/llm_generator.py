import json
from openai import OpenAI
from tenacity import retry, wait_fixed, stop_after_delay, stop_after_attempt
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from atlas_rag.retriever.filter_template import messages as filter_messages, validate_filter_output
from atlas_rag.retriever.rag_qa_prompt import prompt_template

# from https://github.com/OSU-NLP-Group/HippoRAG/blob/main/src/qa/qa_reader.py
# prompts from hipporag qa_reader
cot_system_instruction = ('As an advanced reading comprehension assistant, your task is to analyze text passages and corresponding questions meticulously. If the information is not enough, you can use your own knowledge to answer the question.'
                          'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                          'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')
cot_system_instruction_no_doc = ('As an advanced reading comprehension assistant, your task is to analyze the questions and then answer them. '
                                 'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                 'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')
# This is the instruction for the KG-based QA task
cot_system_instruction_kg = ('As an advanced reading comprehension assistant, your task is to analyze extracted information and corresponding questions meticulously. If the knowledge graph information is not enough, you can use your own knowledge to answer the question. '
                                'Your response start after "Thought: ", where you will methodically break down the reasoning process, illustrating how you arrive at conclusions. '
                                'Conclude with "Answer: " to present a concise, definitive response as a noun phrase, no elaborations.')

class LLMGenerator():
    def __init__(self, client, model_name):
        self.model_name = model_name
        self.client : OpenAI = client
        self.cot_system_instruction = "".join(cot_system_instruction)
        self.cot_system_instruction_no_doc = "".join(cot_system_instruction_no_doc)
        self.cot_system_instruction_kg = "".join(cot_system_instruction_kg)

    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(5))
    def _generate_response(self, messages, max_new_tokens=32768, temperature=0.7):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=temperature,
        )
        return response.choices[0].message.content
    
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(5))
    def filter_generation(self, messages):
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=0.0,
            top_p=0.1,
            max_tokens=4096,
            response_format={"type": "json_object"},
            # Additional parameters for stability
            frequency_penalty=0,
            presence_penalty=0,
        )
        return response.choices[0].message.content
    def _generate_batch_response(self, batch_messages, max_new_tokens=32768, temperature=0.7):
        # Use ThreadPoolExecutor for concurrent requests if using API
        with ThreadPoolExecutor() as executor:
            future_to_index = {
                executor.submit(self._generate_response, msg, max_new_tokens, temperature): idx 
                for idx, msg in enumerate(batch_messages)
            }
            results = [None] * len(batch_messages)  # Pre-allocate results list
            for future in as_completed(future_to_index):
                index = future_to_index[future]  # Get the original index
                results[index] = future.result()  # Place the result in the correct position
        return results

    def generate(self, question, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": self.cot_system_instruction_no_doc},
            {"role": "user", "content": question},
        ]
        return self._generate_response(messages, max_new_tokens=max_new_tokens)

    def generate_with_context(self, question, context, max_new_tokens=1024, frequency_penalty=None, temperature = 0.7, seed = None):
        messages = [
            {"role": "system", "content": self.cot_system_instruction},
            {"role": "user", "content": f"{context}\n\n{question}\nThought:"},
        ]
        # return self._generate_response(messages, max_new_tokens=max_new_tokens, frequency_penalty=frequency_penalty, temperature = temperature, seed = seed)
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)

    def generate_with_context_one_shot(self, question, context, max_new_tokens=4096):
        messages = deepcopy(prompt_template)
        messages.append(
            {"role": "user", "content": f"{context}\n\nQuestions:{question}\nThought:"},
            
        )
        return self._generate_response(messages, max_new_tokens=max_new_tokens)
    def generate_with_context_kg(self, question, context, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": self.cot_system_instruction_kg},
            {"role": "user", "content": f"{context}\n\n{question}"},
        ]
        return self._generate_response(messages, max_new_tokens=max_new_tokens)

    def filter_triples_with_entity(self,question, nodes, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": """
            Your task is to filter text cnadidates based on their relevance to a given query.
            The query requires careful analysis and possibly multi-hop reasoning to connect different pieces of information.
            You must only select relevant texts from the provided candidate list that have connection to the query, aiding in reasoning and providing an accurate answer.
            The output should be in JSON format, e.g., {"nodes": [e1, e2, e3, e4]}, and if no texts are relevant, return an empty list, {"nodes": []}.
            Do not include any explanations, additional text, or context Only provide the JSON output.
            Do not change the content of each object in the list. You must only use text from the candidate list and cannot generate new text."""},

            {"role": "user", "content": f"""{question} \n Output Before Filter: {nodes} \n Output After Filter:"""}
        ]
        try:
            response = json.loads(self._generate_response(messages, max_new_tokens=max_new_tokens))
            # loop through the reponse json and check if all node is original nodes else go to exception
            return response
        except Exception as e:
            # If all retries fail, return the original triples
            return json.loads(nodes)

    def filter_triples_with_entity_event(self,question, triples):
        messages = deepcopy(filter_messages)
        messages.append(
            {"role": "user", "content": f"""[ ## question ## ]]
        {question}

        [[ ## fact_before_filter ## ]]
        {triples}"""})
        
        try:
            response = self.filter_generation(messages)
            cleaned_data = validate_filter_output(response)
            return cleaned_data['fact']
        except Exception as e:
            # If all retries fail, return the original triples
            return []
    def generate_with_custom_messages(self, custom_messages, max_new_tokens=1024):
        return self._generate_response(custom_messages, max_new_tokens)
    
    def ner(self, text):
        messages = [
            {"role": "system", "content": "Please extract the entities from the following question and output them separated by comma, in the following format: entity1, entity2, ..."},
            {"role": "user", "content": f"Extract the named entities from: {text}"},
        ]
        return self._generate_response(messages)
 
