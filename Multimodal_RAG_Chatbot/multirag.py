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

FILE_NAME = 'market.pdf'
FILE_PATH = os.path.join(os.path.dirname(__file__), FILE_NAME)
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 100
MODEL_NAME = "gemini-1.5-flash-001"
MEMORY_WINDOW = 20

llm = ChatGoogleGenerativeAI(model=MODEL_NAME)

left_hand_terms = [
    "a candlestick image",
    "a bullish candle and a bearish candle",
    "a red candlestick",
    "a doji",
    "candlestick patterns",
    "a hammer",
    "image of several patterns including gravestone, doji, traditional and long-legged",
    "image of shooting-star pattern",
    "image of hanging man",
    "image of bullish checkmate",
    "image of bearish candlestick",
    "image of morning star",
    "image of evening star",
    "image of bearish engulfing pattern",
    "image of bullish engulfing pattern",
    "image of harami/inside bar",
    "image of bullish and bearish kicker",
    "image of piercing line",
    "image of dark cloud cover",
    "image of three white soldiers",
    "image of three black crows",
    "image of tweezer bottoms and top",
    "image of doji at support",
    "image of Hammer, Dojis, and Bullish Checkmate at 200 Exponential Moving average"
]

image_list = [
    {"a candlestick image": "market\\1.png"},
    {"a bullish candle and a bearish candle": "market\\2.png"},
    {"a red candlestick": "market\\3.png"},
    {"a doji": "market\\4.png"},
    {"candle stick patterns": "market\\5.png"},
    {"a hammer": "market\\6.png"},
    {"image of several patterns including gravestone,doji,traditional and long-legged": "market\\7.png"},
    {"image of shooting-star pattern": "market\\8.png"},
    {"image of hanging man": "market\\9.png"},
    {"image of bullish checkmate": "market\\10.png"},
    {"image of bearish candlestick": "market\\11.png"},
    {"image of morning star": "market\\12.png"},
    {"image of evening star": "market\\13.png"},
    {"image of bearish engulfing pattern": "market\\14.png"},
    {"image of bullish engulfing pattern": "market\\15.png"},
    {"image of harami/inside bar": "market\\16.png"},
    {"image of bullish and bearish kicker": "market\\17.png"},
    {"image of piercing line": "market\\18.png"},
    {"image of dark cloud cover": "market\\19.png"},
    {"image of three white soldiers": "market\\20.png"},
    {"image of three black crows": "market\\21.png"},
    {"image of tweezer bottoms and top": "market\\22.jpg"},
    {"image of doji at support": "market\\23.jpg"},
    {"image of Hammer, Dojis, and Bullish Checkmate at 200 Exponential Moving average": "market\\24.png"}
]

def image_search(sentence):
    return llm.invoke(f"Use the sentence and reply a single most similar word in image_list\n\nSentence:{sentence}\n\nimage_list:{image_list}").content

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

import asyncio

async def initialize_llm():
    return ChatGoogleGenerativeAI(model=MODEL_NAME)

@cl.on_chat_start
async def start():
    global llm
    llm = await initialize_llm()
    
    pdf_text = load_pdf_text(FILE_PATH)
    docsearch = create_docsearch(text_splitter, pdf_text, embeddings_model)
    cl.user_session.set("docsearch", docsearch)
    cl.user_session.set("memory", memory)

@cl.on_message
async def on_message(msg: cl.Message):
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
            response = f"An error occurred: {str(e)}"
            await cl.Message(content=response).send()
            return

        try:
            image_key = llm.invoke(f"From the given list return only one element in the exact same format as given in the list without any change or punctuation marks such that it best corresponds to the statement given.\n\nList:{str(left_hand_terms)}\n\nStatement:{answer}").content.strip()

            print(image_key)
            image_key = image_key.strip().strip("'").strip('"')

            image_path = None
            for image_dict in image_list:
                if image_key in image_dict:
                    image_path = image_dict[image_key]
                    break

            print(image_path)
            if image_path:
                image = cl.Image(path=image_path, name="image1", display="inline")
                await cl.Message(content="Here is the relevant image:", elements=[image]).send()
            else:
                await cl.Message(content="No image found for the given description.").send()

        except Exception as e:
            await cl.Message(content=f"An error occurred while searching for the image: {str(e)}").send()

    await cl.Message(content=answer).send()