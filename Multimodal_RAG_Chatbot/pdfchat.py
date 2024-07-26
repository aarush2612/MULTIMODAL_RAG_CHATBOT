from langchain_google_genai import ChatGoogleGenerativeAI
import chainlit as cl
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import LLMChain
from langchain_community.utilities import SerpAPIWrapper
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
from PyPDF2 import PdfReader
from langchain.vectorstores import Chroma
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain_cohere import CohereEmbeddings
from langchain.chains import RetrievalQAWithSourcesChain
import io
import requests
import re

text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=500)

system_template = """Use the following pieces of context to answer the user's question.
If you don't know the answer, just say that you don't know, don't try to make up an answer.
ALWAYS return a "SOURCES" part in your answer.
The "SOURCES" part should be a reference to the source of the document from which you got your answer.

Example of your response should be:

The answer is foo
SOURCES: xyz

Begin!
----------------
{summaries}"""

messages = [
    SystemMessagePromptTemplate.from_template(system_template),
    HumanMessagePromptTemplate.from_template("{question}"),
]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

embeddings_model = CohereEmbeddings(cohere_api_key="4VDGdDR9TGJXDU1le3Y5gtqjZFfGOBQBKvvC85C6")

@cl.on_chat_start
def start():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")

    search = SerpAPIWrapper()

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for answering questions about current events or questions that require logical analysis. Ask targeted questions for best results."
        )
    ]

    prefix = """You are Lumiella. A helpful, friendly, informative, and intelligent chatbot, created by Aarush, who studies at IITB and you're based on boson LLM model. Always provide detailed yet concise answers. You have access to the following tools if absolutely necessary:"""
    suffix = """Begin!

    {chat_history}
    Question: {input}
    {agent_scratchpad}"""

    prompt = ZeroShotAgent.create_prompt(
        tools,
        prefix=prefix,
        suffix=suffix,
        input_variables=["input", "chat_history", "agent_scratchpad"]
    )
    memory = ConversationBufferWindowMemory(memory_key="chat_history", k=20)

    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True, memory=memory, handle_parsing_errors=True)
    cl.user_session.set("agent_chain", agent_chain)
    cl.user_session.set("llm", llm)

@cl.on_message
async def on_message(msg: cl.Message):
    agent = cl.user_session.get("agent_chain")
    memory = agent.memory

    llm1 = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")

    cb = cl.LangchainCallbackHandler(stream_final_answer=True)

    pdfs = [file for file in msg.elements if "pdf" in file.mime]
    texts = [file for file in msg.elements if "text" in file.mime]
    url_pattern = re.compile(r'https?://\S+')
    urls = url_pattern.findall(msg.content)

    all_texts, all_metadatas = [], []

    if pdfs:
        for pdf in pdfs:
            file_path = pdf.path
            with open(file_path, 'rb') as file:
                file_content = file.read()
            pdf_stream = io.BytesIO(file_content)
            pdf_reader = PdfReader(pdf_stream)
            pdf_text = "".join(page.extract_text() for page in pdf_reader.pages)

            texts_split = text_splitter.split_text(pdf_text)
            metadatas = [{"source": f"{pdf.name}-{i}"} for i in range(len(texts_split))]

            all_texts.extend(texts_split)
            all_metadatas.extend(metadatas)

    if texts:
        for text in texts:
            file_path = text.path
            with open(file_path, 'r') as file:
                text_content = file.read().strip()
            if text_content:
                texts_split = text_splitter.split_text(text_content)
                metadatas = [{"source": f"{text.name}-{i}"} for i in range(len(texts_split))]

                all_texts.extend(texts_split)
                all_metadatas.extend(metadatas)

    if all_texts:
        embeddings = embeddings_model
        docsearch = await cl.make_async(Chroma.from_texts)(
            all_texts, embeddings, metadatas=all_metadatas
        )

        cl.user_session.set("docsearch", docsearch)
        cl.user_session.set("metadatas", all_metadatas)
        cl.user_session.set("texts", all_texts)

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm1,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
        )
        res = await chain.acall(msg.content, callbacks=[cb])
        answer = res["answer"]
        sources = res["sources"]
        response = f"{answer}\n\nSOURCES: {sources}"
        await cl.Message(content=response).send()
    elif urls:
        for url in urls:
            response = requests.get(url)
            file_extension = url.split('.')[-1]

            if file_extension == 'pdf':
                pdf_stream = io.BytesIO(response.content)
                pdf = PdfReader(pdf_stream)
                pdf_text = "".join(page.extract_text() for page in pdf.pages)

                texts_split = text_splitter.split_text(pdf_text)
                metadatas = [{"source": f"url-{url}"} for _ in range(len(texts_split))]

                embeddings = embeddings_model
                docsearch = await cl.make_async(Chroma.from_texts)(
                    texts_split, embeddings, metadatas=metadatas
                )

                cl.user_session.set("metadatas", metadatas)
                cl.user_session.set("texts", texts_split)
                cl.user_session.set("docsearch", docsearch)

                chain = RetrievalQAWithSourcesChain.from_chain_type(
                    llm1,
                    chain_type="stuff",
                    retriever=docsearch.as_retriever(),
                )
                res = await chain.acall(msg.content, callbacks=[cb])
                answer = res["answer"]
                sources = res["sources"]
                response = f"{answer}\n\nSOURCES: {sources}"
                await cl.Message(content=response).send()
            elif file_extension in ['txt', 'md']:
                text_content = response.text.strip()

                if text_content:
                    text_message = HumanMessage(content=f"Text content: {text_content}. Use this information to answer the following query: {msg.content}")
                    text_response = llm1.invoke([text_message]).content
                    await cl.Message(content=text_response).send()

                    memory.save_context(
                        {"content": msg.content},
                        {"content": text_response}
                    )
                else:
                    await cl.Message(content="The provided text content is empty. Therefore, there is no text to answer the query.").send()
            else:
                await cl.Message(content="Unsupported file format in URL. Please provide a PDF or text file.").send()
    else:
        docsearch = cl.user_session.get("docsearch")
        if docsearch:
            chain = RetrievalQAWithSourcesChain.from_chain_type(
                llm1,
                chain_type="stuff",
                retriever=docsearch.as_retriever(),
            )
            res = await chain.acall(msg.content, callbacks=[cb])
            answer = res["answer"]
            sources = res["sources"]
            response = f"{answer}\n\nSOURCES: {sources}"
            await cl.Message(content=response).send()
        else:
            res = await agent.acall(msg.content, callbacks=[cb])

            memory.save_context(
                {"content": msg.content},
                {"content": res["output"]}
            )

            if cb.has_streamed_final_answer:
                await cb.final_stream.update()
            else:
                await cl.Message(content=res["output"]).send()