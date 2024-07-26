import os
import io
from PyPDF2 import PdfReader
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.vectorstores import Chroma
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferWindowMemory
import chainlit as cl

# Constants
FILE_NAME = 'instidata.pdf'
FILE_PATH = os.path.join(os.path.dirname(__file__), FILE_NAME)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MODEL_NAME = "gemini-1.5-flash-001"
MEMORY_WINDOW = 20

# Initialize components
text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
embeddings_model = CohereEmbeddings()
memory = ConversationBufferWindowMemory(k=MEMORY_WINDOW, input_key="question", output_key="answer")

system_template = """
Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.

Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)

def load_pdf_text(file_path):
    """Load and extract text from the PDF file."""
    with open(file_path, 'rb') as file:
        pdf_reader = PdfReader(io.BytesIO(file.read()))
        return "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())

def create_docsearch(text_splitter, pdf_text, embeddings_model):
    """Create document search index from PDF text."""
    texts = text_splitter.split_text(pdf_text)
    metadatas = [{"source": f"instidata-{i}"} for i in range(len(texts))]
    return Chroma.from_texts(texts, embeddings_model, metadatas=metadatas)

@cl.on_chat_start
def start():
    pdf_text = load_pdf_text(FILE_PATH)
    docsearch = create_docsearch(text_splitter, pdf_text, embeddings_model)
    cl.user_session.set("docsearch", docsearch)
    cl.user_session.set("memory", memory)

@cl.on_message
async def on_message(msg: cl.Message):
    llm = ChatGoogleGenerativeAI(model=MODEL_NAME)
    docsearch = cl.user_session.get("docsearch")
    memory = cl.user_session.get("memory")

    if not docsearch:
        response = "Document search index is not available."
    else:
        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
            memory=memory,
        )
        try:
            res = await chain.ainvoke({"question": msg.content})
            answer = res["answer"]
        except Exception as e:
            answer = f"An error occurred: {str(e)}"

        response = answer
        memory.save_context({"question": msg.content}, {"answer": response})

    await cl.Message(content=response).send()