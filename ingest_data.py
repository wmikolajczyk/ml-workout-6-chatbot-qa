from langchain import FAISS

from get_dataset import prepare_dataset
from langchain.embeddings import HuggingFaceEmbeddings


def main():
    data = prepare_dataset()
    texts = data["text"]
    metadatas = data["metadata"]

    # embedding model
    embedding_model = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2", model_kwargs={"device": "mps"})
    text_embeddings = embedding_model.embed_documents(texts)

    text_embedding_pairs = list(zip(texts, text_embeddings))

    # vector db
    vector_db = FAISS.from_embeddings(text_embedding_pairs, embedding_model, metadatas)
    vector_db.save_local("vector_db_faiss")


if __name__ == "__main__":
    main()
