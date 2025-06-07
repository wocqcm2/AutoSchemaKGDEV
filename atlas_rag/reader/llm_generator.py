import json
from openai import OpenAI, NOT_GIVEN
from tenacity import retry, wait_fixed, stop_after_delay, stop_after_attempt
from copy import deepcopy
from atlas_rag.retrieval.filter_template import messages as filter_messages, validate_filter_output
from atlas_rag.retrieval.prompt_template import prompt_template
from atlas_rag.billion.prompt_template import ner_prompt, validate_keyword_output, keyword_filtering_prompt
from transformers.pipelines import Pipeline
import jsonschema

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
        self.client : OpenAI|Pipeline  = client
        if isinstance(client, OpenAI):
            self.inference_type = "openai"
        elif isinstance(client, Pipeline):
            self.inference_type = "pipeline"
        else:
            raise ValueError("Unsupported client type. Please provide either an OpenAI client or a Huggingface Pipeline Object.")
        self.cot_system_instruction = "".join(cot_system_instruction)
        self.cot_system_instruction_no_doc = "".join(cot_system_instruction_no_doc)
        self.cot_system_instruction_kg = "".join(cot_system_instruction_kg)

    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(5))
    def _generate_response(self, messages, do_sample=True, 
                           max_new_tokens=32768,
                           temperature = 0.7,
                           frequency_penalty = None,
                           response_format = {"type": "text"}  # Default response format,
                           ):
        if self.inference_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                frequency_penalty= NOT_GIVEN if frequency_penalty is None else frequency_penalty,
                response_format = response_format if response_format is not None else {"type": "text"},
            )
            return response.choices[0].message.content
        elif self.inference_type == "pipeline":
            # Convert messages to a single string for Hugging Face
            input_text = " ".join([msg["content"] for msg in messages])
            response = self.client(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=True,
            )
            return response[0]["generated_text"]
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")
    
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(5))
    def filter_generation(self, messages):
        """
        Filter the generation using the configured client.

        Args:
            messages: The input messages for the model.

        Returns:
            The filtered response.
        """
        if self.inference_type == "openai":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.0,
                top_p=0.1,
                max_tokens=4096,
                response_format={"type": "json_object"},
                frequency_penalty=0,
                presence_penalty=0,
            )
            return response.choices[0].message.content
        elif self.inference_type == "huggingface":
            response = self.client(
                messages,
                max_new_tokens=4096,
                temperature=0.0,
                top_p=0.1,
                do_sample=True,
            )
            return response[0]["generated_text"]
        else:
            raise ValueError(f"Unsupported client type: {self.client_type}")

    # def _generate_batch_response(self, batch_messages, max_new_tokens=32768, temperature=0.7):
    #     # Use ThreadPoolExecutor for concurrent requests if using API
    #     with ThreadPoolExecutor() as executor:
    #         future_to_index = {
    #             executor.submit(self._generate_response, msg, max_new_tokens, temperature): idx 
    #             for idx, msg in enumerate(batch_messages)
    #         }
    #         results = [None] * len(batch_messages)  # Pre-allocate results list
    #         for future in as_completed(future_to_index):
    #             index = future_to_index[future]  # Get the original index
    #             results[index] = future.result()  # Place the result in the correct position
    #     return results

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
    def generate_with_custom_messages(self, custom_messages, do_sample=True, max_new_tokens=1024, temperature=0.8, frequency_penalty = None):
        return self._generate_response(custom_messages, do_sample, max_new_tokens, temperature, frequency_penalty)
    
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_filter_keywords_with_entity(self, question, keywords):
        messages = deepcopy(keyword_filtering_prompt)
        
        messages.append({
            "role": "user",
            "content": f"""[[ ## question ## ]]
            {question}
            [[ ## keywords_before_filter ## ]]
            {keywords}"""
        })
        
        try:
            response = self.generate_with_custom_messages(messages, response_format={"type": "json_object"}, temperature=0.0, max_new_tokens=2048)
            
            # Validate and clean the response
            cleaned_data = validate_keyword_output(response)
            
            return cleaned_data['keywords']
        except Exception as e:
            return keywords
    
    def ner(self, text):
        messages = [
            {"role": "system", "content": "Please extract the entities from the following question and output them separated by comma, in the following format: entity1, entity2, ..."},
            {"role": "user", "content": f"Extract the named entities from: {text}"},
        ]
        return self._generate_response(messages)
    
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_ner(self, text):
        messages = deepcopy(ner_prompt)
        messages.append(
            {
                "role": "user", 
                "content": f"[[ ## question ## ]]\n{text}" 
            }
        )
        
        # Generate raw response from LLM
        raw_response = self._generate_response(messages, max_new_tokens=4096, temperature=0.7, frequency_penalty=1.1, response_format={"type": "json_object"})
        
        try:
            # Validate and clean the response
            cleaned_data = validate_keyword_output(raw_response)
            return cleaned_data['keywords']
        
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            return []  # Fallback to empty list or raise custom exception
 
