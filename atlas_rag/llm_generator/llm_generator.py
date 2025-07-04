import json
from openai import OpenAI, AzureOpenAI, NOT_GIVEN
from tenacity import retry, stop_after_attempt, stop_after_delay, wait_fixed, wait_exponential, wait_random
from copy import deepcopy

from atlas_rag.llm_generator.prompt.rag_prompt import cot_system_instruction, cot_system_instruction_kg, cot_system_instruction_no_doc, prompt_template
from atlas_rag.llm_generator.prompt.filter_triple_prompt import validate_filter_output, messages as filter_messages
from atlas_rag.llm_generator.prompt.lkg_prompt import ner_prompt, validate_keyword_output, keyword_filtering_prompt
from atlas_rag.retriever.base import BaseEdgeRetriever, BasePassageRetriever
from atlas_rag.kg_construction.utils.json_processing.json_repair import fix_and_validate_response

from transformers.pipelines import Pipeline
import jsonschema
from typing import Union
from logging import Logger


import time
stage_to_prompt_type = {
    1: "entity_relation",
    2: "event_entity",
    3: "event_relation",
}
retry_decorator = retry(
    stop=(stop_after_delay(120) | stop_after_attempt(5)),  # Max 2 minutes or 5 attempts
    wait=wait_exponential(multiplier=1, min=2, max=30) + wait_random(min=0, max=2),
)
class LLMGenerator():
    def __init__(self, client, model_name):
        self.model_name = model_name
        self.client : OpenAI|Pipeline  = client
        if isinstance(client, OpenAI|AzureOpenAI):
            self.inference_type = "openai"
        elif isinstance(client, Pipeline):
            self.inference_type = "pipeline"
        else:
            raise ValueError("Unsupported client type. Please provide either an OpenAI client or a Huggingface Pipeline Object.")

    @retry_decorator
    def _generate_response(self, messages, do_sample=True, 
                           max_new_tokens=8192,
                           temperature = 0.7,
                           frequency_penalty = None,
                           response_format = {"type": "text"},
                           return_text_only=True,
                           return_thinking=False,
                           reasoning_effort=None
                           ):
        if self.inference_type == "openai":
            start_time = time.time()
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                max_tokens=max_new_tokens,
                temperature=temperature,
                frequency_penalty= NOT_GIVEN if frequency_penalty is None else frequency_penalty,
                response_format = response_format if response_format is not None else {"type": "text"},
                timeout = 120,
                reasoning_effort= NOT_GIVEN if reasoning_effort is None else reasoning_effort,
            )
            time_cost = time.time() - start_time
            content = response.choices[0].message.content
            if content is None and hasattr(response.choices[0].message, 'reasoning_content'):
                content = response.choices[0].message.reasoning_content
            else:
                content = response.choices[0].message.content
            if '</think>' in content and not return_thinking:
                content = content.split('</think>')[-1].strip()
            else:
                if hasattr(response.choices[0].message, 'reasoning_content') and response.choices[0].message.reasoning_content is not None:
                    content = '<think>' + response.choices[0].message.reasoning_content + '</think>' + content
            if return_text_only:
                return content
            else:
                completion_usage_dict = response.usage.model_dump()
                completion_usage_dict['time'] = time.time() - start_time 
                return content, completion_usage_dict
                
        elif self.inference_type == "pipeline":
            # Convert messages to a single string for Hugging Face
            start_time = time.time()
            response = self.client(
                messages,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=do_sample,
            )
            time_cost = time.time() - start_time
            content = response[0]['generated_text'].strip()
            if '</think>' in content and not return_thinking:
                content = content.split('</think>')[-1].strip()

            if return_text_only:
                return content
            else:
                # return both content and usage
                content = response[0]['generated_text'].strip()
                token_count = len(content.split())  # Approximate token count
                time_cost = time.time() - start_time  # Calculate time cost
                completion_usage_dict = {
                    'completion_tokens': token_count,
                    'time': time_cost
                }
                return content, completion_usage_dict
        else:
            raise ValueError(f"Unsupported client type: {self.inference_type}")

    def generate_cot(self, question, max_new_tokens=1024):
        messages = [
            {"role": "system", "content": "".join(cot_system_instruction_no_doc)},
            {"role": "user", "content": question},
        ]
        return self._generate_response(messages, max_new_tokens=max_new_tokens)

    def generate_with_context(self, question, context, max_new_tokens=1024, temperature = 0.7):
        messages = [
            {"role": "system", "content": "".join(cot_system_instruction)},
            {"role": "user", "content": f"{context}\n\n{question}\nThought:"},
        ]
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)

    def generate_with_context_one_shot(self, question, context, max_new_tokens=4096, temperature = 0.7):
        messages = deepcopy(prompt_template)
        messages.append(
            {"role": "user", "content": f"{context}\n\nQuestions:{question}\nThought:"},
        )
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)
    
    def generate_with_context_kg(self, question, context, max_new_tokens=1024, temperature = 0.7):
        messages = [
            {"role": "system", "content": "".join(cot_system_instruction_kg)},
            {"role": "user", "content": f"{context}\n\n{question}"},
        ]
        return self._generate_response(messages, max_new_tokens=max_new_tokens, temperature = temperature)

    @retry_decorator
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
        
    @retry_decorator
    def filter_triples_with_entity_event(self,question, triples):
        messages = deepcopy(filter_messages)
        messages.append(
            {"role": "user", "content": f"""[ ## question ## ]]
        {question}

        [[ ## fact_before_filter ## ]]
        {triples}"""})
        try:
            response = self._generate_response(messages, max_new_tokens=4096, temperature=0.0, response_format={"type": "json_object"})
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
 
    @retry(stop=(stop_after_delay(60) | stop_after_attempt(6)), wait=wait_fixed(2))
    def large_kg_tog_ner(self, text):
        messages = [
            {"role": "system", "content": "You are an advanced AI assistant that extracts named entities from given text. "},
            {"role": "user", "content": f"Extract the named entities from: {text}"}
        ]
        
        # Generate raw response from LLM
        raw_response = self._generate_response(messages, max_new_tokens=4096, temperature=0.7, frequency_penalty=1.1, response_format={"type": "json_object"})
        
        try:
            # Validate and clean the response
            cleaned_data = validate_keyword_output(raw_response)
            return cleaned_data['keywords']
        
        except (json.JSONDecodeError, jsonschema.ValidationError) as e:
            return []  # Fallback to empty list or raise custom exception

    def generate_with_react(self, question, context=None, max_new_tokens=1024, search_history=None, logger=None):
        react_system_instruction = (
            'You are an advanced AI assistant that uses the ReAct framework to solve problems through iterative search. '
            'Follow these steps in your response:\n'
            '1. Thought: Think step by step and analyze if the current context is sufficient to answer the question. If not, review the current context and think critically about what can be searched to help answer the question.\n'
            '   - Break down the question into *1-hop* sub-questions if necessary (e.g., identify key entities like people or places before addressing specific events).\n'
            '   - Use the available context to make inferences about key entities and their relationships.\n'
            '   - If a previous search query (prefix with "Previous search attempt") was not useful, reflect on why and adjust your strategy—avoid repeating similar queries and consider searching for general information about key entities or related concepts.\n'
            '2. Action: Choose one of:\n'
            '   - Search for [Query]: If you need more information, specify a new query. The [Query] must differ from previous searches in wording and direction to explore new angles.\n'
            '   - No Action: If the current context is sufficient.\n'
            '3. Answer: Provide one of:\n'
            '   - A concise, definitive response as a noun phrase if you can answer.\n'
            '   - "Need more information" if you need to search.\n\n'
            'Format your response exactly as:\n'
            'Thought: [your reasoning]\n'
            'Action: [Search for [Query] or No Action]\n'
            'Answer: [concise noun phrase if you can answer, or "Need more information" if you need to search]\n\n'
        )
        
        # Build context with search history if available
        full_context = []
        if search_history:
            for i, (thought, action, observation) in enumerate(search_history):
                search_history_text = f"\nPrevious search attempt {i}:\n"
                search_history_text += f"{action}\n  Result: {observation}\n"
                full_context.append(search_history_text)
        if context:
            full_context_text = f"Current Retrieved Context:\n{context}\n"
            full_context.append(full_context_text)
        if logger:
            logger.info(f"Full context for ReAct generation: {full_context}")
        
        # Combine few-shot examples with system instruction and user query
        messages = [
            {"role": "system", "content": react_system_instruction},
            {"role": "user", "content": f"Search History:\n\n{''.join(full_context)}\n\nQuestion: {question}" 
            if full_context else f"Question: {question}"}
        ]
        if logger:
            logger.info(f"Messages for ReAct generation: {search_history}Question: {question}")
        return self._generate_response(messages, max_new_tokens=max_new_tokens)

    def generate_with_rag_react(self, question: str, retriever: Union['BaseEdgeRetriever', 'BasePassageRetriever'], max_iterations: int = 5, max_new_tokens: int = 1024, logger: Logger = None):
        """
        Generate a response using RAG with ReAct framework, starting with an initial search using the original query.
        
        Args:
            question (str): The question to answer
            retriever: The retriever instance to use for searching
            max_iterations (int): Maximum number of ReAct iterations
            max_new_tokens (int): Maximum number of tokens to generate per iteration
            
        Returns:
            tuple: (final_answer, search_history)
                - final_answer: The final answer generated
                - search_history: List of (thought, action, observation) tuples
        """
        search_history = []
        
        # Perform initial search with the original query
        if isinstance(retriever, BaseEdgeRetriever):
            initial_context, _ = retriever.retrieve(question, topN=5)
            current_context = ". ".join(initial_context)
        elif isinstance(retriever, BasePassageRetriever):
            initial_context, _ = retriever.retrieve(question, topN=5)
            current_context = "\n".join(initial_context)
        
        # Start ReAct process with the initial context
        for iteration in range(max_iterations):
            # First, analyze if we can answer with current context
            analysis_response = self.generate_with_react(
                question=question,
                context=current_context,
                max_new_tokens=max_new_tokens,
                search_history=search_history,
                logger = logger
            )
            
            if logger:
                logger.info(f"Analysis response: {analysis_response}")
                
            try:
                # Parse the analysis response
                thought = analysis_response.split("Thought:")[1].split("\n")[0]
                if logger:
                    logger.info(f"Thought: {thought}")
                action = analysis_response.split("Action:")[1].split("\n")[0]
                answer = analysis_response.split("Answer:")[1].strip()
                
                # If the answer indicates we can answer with current context
                if answer.lower() != "need more information":
                    search_history.append((thought, action, "Using current context"))
                    return answer, search_history
                
                # If we need more information, perform the search
                if "search" in action.lower():
                    # Extract search query from the action
                    search_query = action.split("search for")[-1].strip()
                    
                    # Perform the search
                    if isinstance(retriever, BaseEdgeRetriever):
                        new_context, _ = retriever.retrieve(search_query, topN=3)
                        # Filter out contexts that are already in current_context
                        current_contexts = current_context.split(". ")
                        new_context = [ctx for ctx in new_context if ctx not in current_contexts]
                        new_context = ". ".join(new_context)
                    elif isinstance(retriever, BasePassageRetriever):
                        new_context, _ = retriever.retrieve(search_query, topN=3)
                        # Filter out contexts that are already in current_context
                        current_contexts = current_context.split("\n")
                        new_context = [ctx for ctx in new_context if ctx not in current_contexts]
                        new_context = "\n".join(new_context)
                    
                    # Store the search results as observation
                    if new_context:
                        observation = f"Found information: {new_context}"
                    else:
                        observation = "No new information found. Consider searching for related entities or events."
                    search_history.append((thought, action, observation))
                    
                    # Update context with new search results
                    if new_context:
                        current_context = f"{current_context}\n{new_context}"
                        if logger:
                            logger.info(f"New search results: {new_context}")
                    else:
                        if logger:
                            logger.info("No new information found, suggesting to try related entities")

                else:
                    # If no search is needed but we can't answer, something went wrong
                    search_history.append((thought, action, "No action taken but answer not found"))
                    return "Unable to find answer", search_history
                
            except Exception as e:
                if logger:
                    logger.error(f"Error parsing ReAct response: {e}")
                return analysis_response, search_history
        
        # If we've reached max iterations, return the last answer
        return answer, search_history
    
    @retry_decorator
    def triple_extraction(self, messages, max_tokens=4096, stage=None, record = False):
        prompt_type = stage_to_prompt_type.get(stage, None)
        responses = []

        # Normalize messages input
        if isinstance(messages[0], dict):
            messages = [messages]  # Wrap single message list in a list
        # messages is list of list of dict
        for input_data in messages:
            try:
                content, completion_usage_dict = self._generate_response(
                    messages = input_data,
                    max_new_tokens = max_tokens,
                    temperature=0.0,
                    frequency_penalty=0.5,
                    reasoning_effort="none",
                    return_text_only=False
                )
                if prompt_type:
                    corrected, error = fix_and_validate_response(content, prompt_type)
                    if error:
                        raise ValueError(f"Validation failed for prompt_type '{prompt_type}'")
                if corrected and corrected.strip():
                    if record:
                        responses.append((corrected, completion_usage_dict))
                    else:
                        responses.append(corrected)
            except Exception as e:
                print(f"Failed to generate valid response for input: {input_data} - Error: {str(e)}")
                # add empty response to maintain index alignment
                if record:
                    completion_usage_dict = {
                        'completion_tokens': 0,
                        'total_tokens': 0,
                        'time': 0
                    }
                    responses.append(("[]", completion_usage_dict))
                else:
                    responses.append("[]")
        return responses