import os
from langchain import OpenAI

from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains import RetrievalQA
import pinecone

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENVIRONMENT"),
)

if __name__ == "__main__":
    print("Hello VectorStore!")
    loader = TextLoader(
        "/Users/gag/MLProjects/intro-to-vector-db/mediumblogs/mediumblog1.txt"
    )
    document = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=0)
    texts = text_splitter.split_documents(documents=document)
    print(len(texts))

    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name="medium-blogs-data-index"
    )
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        return_source_documents=True,
    )
    query = "What is the vettor basa? Give me a 40 characters answer for beginners!"
    result = qa({"query": query})

    print(result)
