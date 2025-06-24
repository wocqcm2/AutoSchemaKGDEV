from openai import OpenAI, AzureOpenAI
from transformers import Pipeline
from atlas_rag.utils.json_repair import fix_and_validate_response
from tenacity import retry, stop_after_attempt, retry_if_exception_type, wait_fixed

max_retries = 3
retry_decorator = retry(
    stop=stop_after_attempt(max_retries),
    wait=wait_fixed(1),
    retry=retry_if_exception_type(Exception),
    reraise=True
)

stage_to_prompt_type = {
    1: "entity_relation",
    2: "event_entity",
    3: "event_relation",
}

def client_generation(model: OpenAI, model_name, messages, max_new_tokens=4096, temperature=0.1, frequency_penalty = 1.1):
    """Generate a response using the OpenAI client."""
    response = model.chat.completions.create(
        model=model_name,
        messages=messages,
        max_tokens=max_new_tokens,
        temperature=temperature,
        frequency_penalty=frequency_penalty
    )
    return response.choices[0].message.content

@retry_decorator
def generate_and_validate(client: OpenAI|Pipeline, model_name, input_data, max_new_tokens, temperature, prompt_type, inference_type, frequency_penalty):
    """Generate and validate a response with retries."""
    if inference_type == "openai":
        content = client_generation(client, model_name, input_data, max_new_tokens, temperature, frequency_penalty)
    elif inference_type == "pipeline":
        generated = client(input_data, max_new_tokens=max_new_tokens, temperature=temperature, return_full_text=False, repetition_penalty = frequency_penalty)
        content = generated[0]['generated_text'].strip()
    else:
        raise ValueError("Unsupported inference type")
    
    if prompt_type:
        corrected, error = fix_and_validate_response(content, prompt_type)
        if error:
            raise ValueError(f"Validation failed for prompt_type '{prompt_type}'")
        return corrected
    return content
    
class TripleGenerator:
    def __init__(self, client: Pipeline | OpenAI | AzureOpenAI, model_name, max_new_tokens=4096, temperature=0.1, frequency_penalty=1.1):
        """Initialize the TripleGenerator with a client and generation parameters."""
        self.client = client
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.frequency_penalty = frequency_penalty
        self.model_name = model_name
        if isinstance(client, (OpenAI, AzureOpenAI)):
            self.inference_type = "openai"
        elif isinstance(client, Pipeline):
            self.inference_type = "pipeline"
        else:
            raise ValueError("Unsupported client type. Please provide either an OpenAI client or a Huggingface Pipeline Object.")

    def generate(self, messages, max_tokens=4096, stage=None):
        """
        Generate triples from the input messages using the specified model.
        
        Args:
            messages: List of inputs. For OpenAI, each input is a list of message dictionaries.
                      For Pipeline, each input is a string.
            max_tokens: Maximum number of tokens to generate.
            stage: Optional stage (1, 2, 3) to determine the prompt type for validation.
        
        Returns:
            A list of valid, non-empty corrected JSON strings.
        """
        prompt_type = stage_to_prompt_type.get(stage, None)
        responses = []

        # Normalize messages input
        if isinstance(messages[0], dict):
            messages = [messages]  # Wrap single message list in a list
        # messages is list of list of dict
        for input_data in messages:
            try:
                corrected = generate_and_validate(
                    self.client,
                    self.model_name,
                    input_data,
                    max_tokens,
                    self.temperature,
                    prompt_type,
                    self.inference_type,
                    self.frequency_penalty
                )
                if corrected and corrected.strip():
                    responses.append(corrected)
            except Exception as e:
                print(f"Failed to generate valid response for input: {input_data} - Error: {str(e)}")
                # add empty response to maintain index alignment
                responses.append("[]")
        return responses
