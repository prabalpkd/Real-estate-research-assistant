from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path


from langchain_classic.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain_community.document_loaders import UnstructuredURLLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
#from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
#from langchain_community.vectorstores import Chroma

load_dotenv() ##to call groq in the cloud

#constants
CHUNK_SIZE=1000
EMBEDDING_MODEL = 'Alibaba-NLP/gte-base-en-v1.5'
VECTORSTORE_DIR = Path(__file__).parent / "resources/vectorstore"
COLLECTION_NAME = 'real-estate'

llm=None
vector_store=None

def initialize_components():
    global llm,vector_store

    if llm is None:
        #initializing the llm
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9,max_tokens=500)

    if vector_store is None:
        ef=HuggingFaceEmbeddings(
            model_name= EMBEDDING_MODEL,
            model_kwargs= {"trust_remote_code":True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function= ef,
            persist_directory=str(VECTORSTORE_DIR),
        )

def process_urls(urls):
    """
    This function scraps data from a url and stores it in a vector db.
    :param urls:
    :return:
    """
    yield "Initializing components...✅"
    initialize_components()

    yield "Resetting vector store...✅"
    vector_store.reset_collection()

    yield "Loading data...✅"
    loader = UnstructuredURLLoader(urls=urls, headers={"User-Agent": "Mozilla/5.0"}) #Loading the data
    data = loader.load()

    yield "Splitting text into chunks...✅"
    #Splitting the data ino chunks
    text_splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n","\n","."," "],
        chunk_size = CHUNK_SIZE
    )
    docs = text_splitter.split_documents(data)

    ## this docs we need to insert into a vector database, So creating a separate function to initialize chromadb.
    yield "Add chunks to the vector db...✅"
    uuids=[str(uuid4()) for _ in range(len(docs))]
    vector_store.add_documents(docs, ids=uuids)

    yield "Done adding docs to the vector database...✅"

def generate_answer(query):
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")

    chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vector_store.as_retriever())
    result = chain.invoke({"question": query}, return_only_output=True)
    sources = result.get("sources", "")
    return result['answer'], sources

    # The RetrievalQAWithSourcesChain will use the vector database as a retriever. When we ask a question,
    # the embedding will be created for it and based on the embedding it will pull out the relevant chunks from the vectordb
    # And from those relevant chunks it will make a query to the llm



### This will take a couple of urls and it will process the urls and store into vector database.
if __name__=="__main__":
    urls=[
        "https://www.cnbc.com/2024/12/21/how-the-federal-reserves-rate-policy-affects-mortgages.html",
        "https://www.cnbc.com/2024/12/20/why-mortgage-rates-jumped-despite-fed-interest-rate-cut.html"
    ]
    process_urls(urls)
    answer, sources = generate_answer("Tell me what was the 30 year fixed mortgage rate along with the date?")
    print(f"Answer:{answer}")
    print(f"Sources:{sources}")

# Can I buy some property in mumbai with 1 million?
# https://www.cnbc.com/2025/08/14/cnbcs-inside-india-newsletter-as-indias-rich-venture-abroad-many-anchor-fortunes-in-real-estate.html