from openai import OpenAI

client = OpenAI(
    api_key="edge123456",
    base_url="http://192.168.2.78:7891/v1",
)
def embedding(inputt):

    # compute the embedding of the text

    embedding = client.embeddings.create(
        input=inputt,
        model="m3s-base"
    )
    out = embedding.data[0].embedding

    return out

# print(embedding('d'))
# print(embedding.data[0].embedding)
# print(len(embedding.data[0].embedding))
