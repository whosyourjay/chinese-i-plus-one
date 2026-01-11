from openai import OpenAI

endpoint = "https://segmentor-resource.cognitiveservices.azure.com/openai/v1/"
model_name = "gpt-5-nano"
deployment_name = "gpt-5-nano-segmentor"

with open('api.key', 'r') as f:
    api_key = f.read().strip()

client = OpenAI(
    base_url=f"{endpoint}",
    api_key=api_key
)

sentence = "少女的话席卷周围，"
word = "席卷"
our_messages = [
    {
        "role": "user",
        "content": f"Separate the following Chinese sentence into individual words. Output as a json array. {sentence}"
    },
    {
        "role": "user",
        "content": f"In the sentence {sentence}，define the word {word}. Give a single word or short phrase as appropriate. Try to make the definition match the part of speech of the word in this sentence."
    }
]

completion = client.chat.completions.create(
    model=deployment_name,
    messages = our_messages[0:1],
)
print(completion.choices[0].message.content)
completion = client.chat.completions.create(
    model=deployment_name,
    messages = our_messages[1:2],
)
print(completion.choices[0].message.content)

