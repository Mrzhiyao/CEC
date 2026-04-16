from openai import OpenAI

client = OpenAI(
    api_key="edge123456",
    base_url="http://YOUR_EMBEDDING_SERVER_IP:7891/v1",
)
def embedding(inputt):

    # compute the embedding of the text

    embedding = client.embeddings.create(
        input=inputt,
        model="m3s-base"
    )
    out = embedding.data[0].embedding

    return out

# print(embedding('In what municipality is Kinsac which shares a province with the Alexander Graham Bell Institute?'))
# print(embedding.data[0].embedding)
# print(len(embedding.data[0].embedding))
