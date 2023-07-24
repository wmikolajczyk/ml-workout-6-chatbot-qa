from langchain import OpenAI, PromptTemplate, LLMChain, HuggingFacePipeline
import gradio as gr
from similarity_search import similarity_search, vector_db, embedding_model

llm_model = OpenAI(temperature=0)  # OPENAI_API_KEY
# llm_model = HuggingFacePipeline.from_model_id("google/flan-t5-base", task="text2text-generation")
template = """Use the following pieces of context to answer the question at the end. 
If you don't know the answer, just say that you don't know, don't try to make up an answer. 
Use three sentences maximum and keep the answer as concise as possible. 
Always say "thanks for asking!" at the end of the answer. 
{context}
Question: {question}
Helpful Answer:"""
QA_CHAIN_PROMPT = PromptTemplate(input_variables=["context", "question"], template=template)
LLM_CHAIN = LLMChain(prompt=QA_CHAIN_PROMPT, llm=llm_model, verbose=True)


def generate_answer(question: str) -> str:
    context_docs = similarity_search(question)
    context = "\n\n".join(doc.page_content for doc in context_docs)
    answer = LLM_CHAIN.run(question=question, context=context)
    return answer


def main():
    demo = gr.Interface(fn=generate_answer, inputs="text", outputs="text")
    demo.launch()


if __name__ == "__main__":
    main()
