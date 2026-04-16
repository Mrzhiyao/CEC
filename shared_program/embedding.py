from openai import OpenAI

client = OpenAI(
    api_key="edge123456",
    base_url="http://192.168.2.78:7891/v1",
)

def embedding(inputt):
    while True:
        try:
            result = client.embeddings.create(
                input=inputt,
                model="m3e-large"
            )
            return result.data[0].embedding
        except:
            pass
