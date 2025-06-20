
TRIPLE_INSTRUCTIONS = {
    "en":{
        "system": "You are a helpful assistant who always response in a valid JSON array",
        "entity_relation": """Given a passage, summarize all the important entities and the relations between them in a concise manner. Relations should briefly capture the connections between entities, without repeating information from the head and tail entities. The entities should be as specific as possible. Exclude pronouns from being considered as entities. 
        You must **strictly output in the following JSON format**:\n
        [
            {
                "Head": "{a noun}",
                "Relation": "{a verb}",
                "Tail": "{a noun}",
            }...
        ]""",

        "event_entity":  """Please analyze and summarize the participation relations between the events and entities in the given paragraph. Each event is a single independent sentence. Additionally, identify all the entities that participated in the events. Do not use ellipses. 
        You must **strictly output in the following JSON format**:\n
        [
            {
                "Event": "{a simple sentence describing an event}",
                "Entity": ["entity 1", "entity 2", "..."]
            }...
        ] """,
    
        "event_relation":  """Please analyze and summarize the relationships between the events in the paragraph. Each event is a single independent sentence. Identify temporal and causal relationships between the events using the following types: before, after, at the same time, because, and as a result. Each extracted triple should be specific, meaningful, and able to stand alone.  Do not use ellipses.  
        You must **strictly output in the following JSON format**:\n
        [
            {
                "Head": "{a simple sentence describing the event 1}",
                "Relation": "{temporal or causality relation between the events}",
                "Tail": "{a simple sentence describing the event 2}"
            }...
        ]""",
        "passage_start" : """Here is the passage."""
    },
    "zh-CN": {
        "system": """"你是一个始终以有效JSON数组格式回应的助手""",
        "entity_relation": """给定一段文字，提取所有重要实体及其关系，并以简洁的方式总结。关系描述应清晰表达实体间的联系，且不重复头尾实体的信息。实体需具体明确，排除代词。  
        返回格式必须为以下JSON结构,内容需用简体中文表述:
        [  
            {  
                "Head": "{名词}",  
                "Relation": "{动词或关系描述}",  
                "Tail": "{名词}"  
            }...  
        ]""",

        "event_entity": """分析段落中的事件及其参与实体。每个事件应为独立单句，列出所有相关实体（需具体，不含代词）。  
        返回格式必须为以下JSON结构,内容需用简体中文表述:
        [  
            {  
                "Event": "{描述事件的简单句子}",  
                "Entity": ["实体1", "实体2", "..."]  
            }...  
        ]""",
       
        "event_relation": """分析事件间的时序或因果关系,关系类型包括:之前,之后,同时,因为,结果.每个事件应为独立单句。  
        返回格式必须为以下JSON结构.内容需用简体中文表述. 
        [  
            {  
                "Head": "{事件1描述}",  
                "Relation": "{时序/因果关系}",  
                "Tail": "{事件2描述}"  
            }...  
        ]""",
        
        "PASSAGE_START": "给定以下段落："
    },
    "zh-HK": {
        "system": "你是一個始終以有效JSON數組格式回覆的助手",
        "entity_relation": """給定一段文字，提取所有重要實體及其關係，並以簡潔的方式總結。關係描述應清晰表達實體間的聯繫，且不重複頭尾實體的信息。實體需具體明確，排除代詞。  
        返回格式必須為以下JSON結構,內容需用繁體中文表述:
        [  
            {  
                "Head": "{名詞}",  
                "Relation": "{動詞或關係描述}",  
                "Tail": "{名詞}"  
            }...  
        ]""",

        "event_entity": """分析段落中的事件及其參與實體。每個事件應為獨立單句，列出所有相關實體（需具體，不含代詞）。  
        返回格式必須為以下JSON結構,內容需用繁體中文表述:
        [  
            {  
                "Event": "{描述事件的簡單句子}",  
                "Entity": ["實體1", "實體2", "..."]  
            }...  
        ]""",
       
        "event_relation": """分析事件間的時序或因果關係,關係類型包括:之前,之後,同時,因為,結果.每個事件應為獨立單句。  
        返回格式必須為以下JSON結構.內容需用繁體中文表述. 
        [  
            {  
                "Head": "{事件1描述}",  
                "Relation": "{時序/因果關係}",  
                "Tail": "{事件2描述}"  
            }...  
        ]""",
        
        "passage_start": "給定以下段落："
    }
}

