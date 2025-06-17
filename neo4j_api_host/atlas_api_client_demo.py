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

wiki_base_url = "http://0.0.0.0:10087/v1/"
wiki_client = OpenAI(api_key="EMPTY", base_url=wiki_base_url)
pes2o_base_url = "http://0.0.0.0:10088/v1/"
pes2o_client = OpenAI(api_key="EMPTY", base_url=pes2o_base_url)
cc_base_url ="http://0.0.0.0:10089/v1/"
cc_client = OpenAI(api_key="EMPTY", base_url=cc_base_url)

# knowledge graph en_simple_wiki_v0
for client in [wiki_client, pes2o_client, cc_client]:
    response = client.chat.completions.create(
        model="llama",
        messages=message,
        max_tokens=2048,
        temperature=0.5,
    )
    print(response.choices[0].message.content)