from openai import OpenAI
message = [
    {
        'role':'system',
        'content':'The following are multiple choice questions (with answers) about miscellaneous.'
    },
    {
        'role':'user', 
        'content':"""Given the following question and four candidate answers (A, B, C and D), choose the answer.\n
    Question: What would be the most likely thing one would do with the compound MgSO4 7H2O?\n
    A. power a car\nB. blow up a building\n
    C. soak ones feet\nD. fertilize a lawn\n"""
    }
]

base_url ="http://0.0.0.0:10089/v1/"
client = OpenAI(api_key="EMPTY", base_url=base_url)

# knowledge graph en_simple_wiki_v0
response = client.chat.completions.create(
    model="llama",
    messages=message,
    max_tokens=2048,
    temperature=0.5,
)
print(response.choices[0].message.content)