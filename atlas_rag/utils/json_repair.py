import json
from typing import Any, Dict, List, Union, TextIO, Optional
import json_repair

def validate_response(response: str, prompt_type: str, event_list: Optional[List[str]] = None):
    """Validate JSON response based on the prompt type"""
    try:
        json_start_token = response.find("[")
        json_text = response[json_start_token:].strip() if json_start_token != -1 else response.strip()
        
        json_text = json_text.strip()

        json_text = json_text.replace("\n", "")
        json_text = json_text.replace("\r", "")
        json_text = json_text.replace("\t", "")
        

        data = json_repair.loads(json_text)
    except json.JSONDecodeError:
        raise ValueError("Invalid JSON format")

    if not isinstance(data, list):
        raise TypeError("Response must be a JSON array")

    required_keys = {
        "entity_relation": {"Head", "Relation", "Tail"},
        "event_entity": {"Event", "Entity"},
        "event_relation": {"Head", "Relation", "Tail"}
    }

    for idx, item in enumerate(data):
        # Basic structure validation
        if not isinstance(item, dict):
            raise TypeError(f"Item {idx} must be a JSON object")

        # Check required keys
        missing = required_keys[prompt_type] - item.keys()
        if missing:
            raise KeyError(f"Item {idx} missing required keys: {missing}")

        # Type checking
        if prompt_type == "entity_relation":
            for key in ["Head", "Relation", "Tail"]:
                if not isinstance(item[key], str) or not item[key].strip():
                    raise TypeError(f"Item {idx} {key} must be a non-empty string")

        elif prompt_type == "event_entity":
            if not isinstance(item["Event"], str) or not item["Event"].strip():
                raise TypeError(f"Item {idx} Event must be a non-empty string")
            if not isinstance(item["Entity"], list) or not item["Entity"]:
                raise TypeError(f"Item {idx} Entity must be a non-empty array")
            for ent in item["Entity"]:
                if not isinstance(ent, str) or not ent.strip():
                    raise TypeError(f"Item {idx} Entity list contains invalid entry: {ent}")

        elif prompt_type == "event_relation":
            # Check exact wording for event_relation_2
            if event_list:
                if item["Head"] not in event_list:
                    raise ValueError(f"Item {idx} Head not in event list: {item['Head']}")
                if item["Tail"] not in event_list:
                    raise ValueError(f"Item {idx} Tail not in event list: {item['Tail']}")

            # Sentence validation
            for key in ["Head", "Tail"]:
                if not isinstance(item[key], str) or not item[key].strip():
                    raise TypeError(f"Item {idx} {key} must be a non-empty sentence")

    return True

def normalize_key(key):
    return key.strip().lower()

def fix_and_validate_response(response: str, prompt_type: str):
    """Attempt to fix and validate JSON response based on the prompt type."""
    # Extract the JSON list from the response
    json_start_token = response.find("[")
    if json_start_token == -1:
        # add [ at the start
        response = "[" + response.strip() + "]"
    parsed_objects = json_repair.loads(response)
    if len(parsed_objects) == 0:
        return "[]", True
    # Define required keys for each prompt type
    required_keys = {
        "entity_relation": {"Head", "Relation", "Tail"},
        "event_entity": {"Event", "Entity"},
        "event_relation": {"Head", "Relation", "Tail"}
    }
    
    corrected_data = []
    for idx, item in enumerate(parsed_objects):
        if not isinstance(item, dict):
            print(f"Item {idx} must be a JSON object. Problematic item: {item}")
            continue
        
        # Correct the keys
        corrected_item = {}
        for key, value in item.items():
            norm_key = normalize_key(key)
            matching_expected_keys = [exp_key for exp_key in required_keys[prompt_type] if normalize_key(exp_key) in norm_key]
            if len(matching_expected_keys) == 1:
                corrected_key = matching_expected_keys[0]
                corrected_item[corrected_key] = value
            else:
                corrected_item[key] = value
        
        # Check for missing keys in corrected_item
        missing = required_keys[prompt_type] - corrected_item.keys()
        if missing:
            print(f"Item {idx} missing required keys: {missing}. Problematic item: {item}")
            continue
        
        # Validate and correct the values in corrected_item
        if prompt_type == "entity_relation":
            for key in ["Head", "Relation", "Tail"]:
                if not isinstance(corrected_item[key], str) or not corrected_item[key].strip():
                    print(f"Item {idx} {key} must be a non-empty string. Problematic item: {corrected_item}")
                    continue
        
        elif prompt_type == "event_entity":
            if not isinstance(corrected_item["Event"], str) or not corrected_item["Event"].strip():
                print(f"Item {idx} Event must be a non-empty string. Problematic item: {corrected_item}")
                continue
            if not isinstance(corrected_item["Entity"], list) or not corrected_item["Entity"]:
                print(f"Item {idx} Entity must be a non-empty array. Problematic item: {corrected_item}")
                continue
            else:
                corrected_item["Entity"] = [ent.strip() for ent in corrected_item["Entity"] if isinstance(ent, str)]
        
        elif prompt_type == "event_relation":
            for key in ["Head", "Tail"]:
                if not isinstance(corrected_item[key], str) or not corrected_item[key].strip():
                    print(f"Item {idx} {key} must be a non-empty sentence. Problematic item: {corrected_item}")
                    continue
        
        corrected_data.append(corrected_item)
    
    if not corrected_data:
        return "[]", True
    
    corrected_json_string = json.dumps(corrected_data, ensure_ascii=False)
    return corrected_json_string, False