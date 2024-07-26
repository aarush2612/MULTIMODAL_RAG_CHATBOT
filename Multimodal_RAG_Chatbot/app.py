from langchain_google_genai import ChatGoogleGenerativeAI
import chainlit as cl
from langchain.agents import ZeroShotAgent, Tool, AgentExecutor
from langchain import LLMChain
from langchain_community.utilities import SerpAPIWrapper
from langchain import LLMMathChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_core.messages import HumanMessage
from langchain.text_splitter import RecursiveCharacterTextSplitter
import base64
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
from PIL import Image
import tempfile
import os
import speech_recognition as sr
from moviepy.editor import VideoFileClip

def image2base64(image_path):
    with open(image_path, "rb") as img:
        encoded_string = base64.b64encode(img.read())
    return encoded_string.decode("utf-8")

def transcribe_audio(file_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(file_path) as source:
        audio_data = recognizer.record(source)
        return recognizer.recognize_google(audio_data)

def extract_frames(video_path):
    clip = VideoFileClip(video_path)
    frames = []
    for i, frame in enumerate(clip.iter_frames()):
        if i % 100 == 0:
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
            frame_image = Image.fromarray(frame)
            frame_image.save(temp_file.name)
            frames.append(temp_file.name)
            if i >= 500:
                break
        else:
            pass
    return frames

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

messages = [SystemMessagePromptTemplate.from_template(system_template),   HumanMessagePromptTemplate.from_template("{question}"),]
prompt = ChatPromptTemplate.from_messages(messages)
chain_type_kwargs = {"prompt": prompt}

embeddings_model = CohereEmbeddings(cohere_api_key="4VDGdDR9TGJXDU1le3Y5gtqjZFfGOBQBKvvC85C6")

@cl.on_chat_start
def start():
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash-001")

    search = SerpAPIWrapper()
    llm_math_chain = LLMMathChain(llm=llm, verbose=True)

    tools = [
        Tool(
            name="Search",
            func=search.run,
            description="Useful for answering questions about current events or questions that require logical analysis. Ask targeted questions for best results."
        ),
        Tool(
            name="Calculator",
            func=llm_math_chain.invoke,
            description="Handles mathematical queries and calculations."
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
    response = ""

    images = [file for file in msg.elements if "image" in file.mime]
    pdfs = [file for file in msg.elements if "pdf" in file.mime]
    texts = [file for file in msg.elements if "text" in file.mime]
    audio_files = [file for file in msg.elements if "audio" in file.mime]
    video_files = [file for file in msg.elements if "video" in file.mime]

    if images:
        loading_message = cl.Message(content="Uploading image...", type="loading")
        await loading_message.send()

        base64_image = image2base64(images[0].path)
        image_url = f"data:image/png;base64,{base64_image}"

        image_message = HumanMessage(
            content=[
                {"type": "text", "text": msg.content if msg.content else "What's in this image?"},
                {"type": "image_url", "image_url": image_url},
            ]
        )
        image_response = llm1.invoke([image_message]).content
        response += image_response

        msg_response = cl.Message(content="")
        for chunk in response:
            await msg_response.stream_token(chunk)

        memory.save_context(
            {"content": msg.content if msg.content else "Image received"},
            {"content": image_response}
        )

        cl.user_session.set("image_content", base64_image)

    elif pdfs:
        file_path = pdfs[0].path

        with open(file_path, 'rb') as file:
            file_content = file.read()

        message = cl.Message(content=f"Processing {pdfs[0].name}...")
        await message.send()

        pdf_stream = io.BytesIO(file_content)
        pdf = PdfReader(pdf_stream)
        pdf_text = ""
        for page in pdf.pages:
            pdf_text += page.extract_text()

        texts = text_splitter.split_text(pdf_text)
        metadatas = [{"source": f"{i}-pl"} for i in range(len(texts))]

        embeddings = embeddings_model
        docsearch = await cl.make_async(Chroma.from_texts)(
            texts, embeddings, metadatas=metadatas
        )

        chain = RetrievalQAWithSourcesChain.from_chain_type(
            llm1,
            chain_type="stuff",
            retriever=docsearch.as_retriever(),
        )

        cl.user_session.set("metadatas", metadatas)
        cl.user_session.set("texts", texts)
        cl.user_session.set("pdf_content", pdf_text)

        message.content = f"Processing {pdfs[0].name} done. You can now ask questions!"
        await message.update()

        cb.answer_reached = True
        res = await chain.acall(msg.content, callbacks=[cb])

        answer = res["answer"]
        sources = res["sources"].strip()
        source_elements = []

        metadatas = cl.user_session.get("metadatas")
        all_sources = [m["source"] for m in metadatas]
        texts = cl.user_session.get("texts")

        if sources:
            found_sources = []
            for source in sources.split(","):
                source_name = source.strip().replace(".", "")
                try:
                    index = all_sources.index(source_name)
                except ValueError:
                    continue
                text = texts[index]
                found_sources.append(source_name)
                source_elements.append(cl.Text(content=text, name=source_name))

            if found_sources:
                answer += f"\nSources: {', '.join(found_sources)}"
            else:
                answer += "\nNo sources found"

        memory.save_context(
            {"content": msg.content},
            {"content": answer}
        )

        if cb.has_streamed_final_answer:
            cb.final_stream.elements = source_elements
            await cb.final_stream.update()
        else:
            await cl.Message(content=answer, elements=source_elements).send()

    elif audio_files:
        file_path = audio_files[0].path
        try:
         transcription = transcribe_audio(file_path)
         transcription_message = HumanMessage(content=f"The user has input an audio file and some program has converted it into transcription for you to use. The user does not know this and you don't have to mention this. \n\nAudio transcription: {transcription}.\n\n Use this information to answer the following query: {msg.content}")
         transcription_response = llm1.invoke([transcription_message]).content
         await cl.Message(content=transcription_response).send()
        except:
            await cl.Message("Audio file could not be read as PCM WAV, AIFF/AIFF-C, or Native FLAC; check if file is corrupted or in another format").send()
        
        memory.save_context(
            {"content": msg.content},
            {"content": transcription_response}
        )

    elif video_files:
        file_path = video_files[0].path
        frames = extract_frames(file_path)
        frame_urls = [f"data:image/png;base64,{image2base64(frame)}" for frame in frames]
        frame_message = HumanMessage(
            content=[
                {"type": "text", "text": msg.content},
                *[{"type": "image_url", "image_url": url} for url in frame_urls],
            ]
        )
        frame_response = llm1.invoke([frame_message]).content
        await cl.Message(content=frame_response).send()

        memory.save_context(
            {"content": msg.content},
            {"content": frame_response}
        )


    elif texts:
        file_path = texts[0].path
        with open(file_path, 'r') as file:
            text_content = file.read().strip()

        if not text_content:
            await cl.Message(content="The provided text content is empty. Therefore, there is no text to answer the query.").send()
        else:
            text_message = HumanMessage(content=f"Text content: {text_content}. Use this information to answer the following query: {msg.content}")
            text_response = llm1.invoke([text_message]).content
            await cl.Message(content=text_response).send()
        
        memory.save_context(
            {"content": msg.content},
            {"content": text_response}
        )

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