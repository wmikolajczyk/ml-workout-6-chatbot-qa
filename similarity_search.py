from langchain import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={"device": "mps"})
vector_db = FAISS.load_local("vector_db_faiss", embedding_model)


def similarity_search(query: str):
    q_embedding = embedding_model.embed_documents([query])[0]
    relevant_doc = vector_db.similarity_search_by_vector(q_embedding, k=3)
    return relevant_doc


if __name__ == '__main__':
    docs1 = similarity_search("Top Gun")
    print("")