from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.tools import create_retriever_tool
from langchain_openai import ChatOpenAI
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.agents import AgentExecutor, create_json_chat_agent, create_tool_calling_agent
from langchain_community.document_loaders import ImageCaptionLoader, UnstructuredURLLoader
from langchain import hub
from langdetect import detect
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
import streamlit as st
import re2

import os

LINKS_PATTERN = r'https?://[^\s/$.?#].[^\s]*'
links_regex = re2.compile(LINKS_PATTERN)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=20, separators=[" ", ",", "\n"])

if "messages" not in st.session_state:
    st.session_state.messages = []

@st.cache_resource
def get_chroma():
    return Chroma(collection_name="typhoon-tools", embedding_function=embedding_function)

embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
db = get_chroma()
retriever = db.as_retriever()

retrieval_tool = create_retriever_tool(
    retriever,
    "links_files_context",
    "Context for users' links and images captions"
)

tools = [retrieval_tool, TavilySearchResults(max_results=5)]

llm = ChatOpenAI(model="experimental-typhoon-next-70b-0524", temperature=0.7, base_url="https://api.opentyphoon.ai/v1")

prompt = hub.pull("chanakan/typhoon-tools")

question = st.chat_input("พิมพ์อะไรสักหน่อยสิ")
#images = ["https://miro.medium.com/v2/resize:fit:1400/1*M4S6A5QXflFmjTBohtyYjw.jpeg", "https://cms-b-assets.familysearch.org/dims4/default/92f0f61/2147483647/strip/true/crop/750x500+0+0/resize/1240x827!/quality/90/?url=https%3A%2F%2Ffamilysearch-brightspot.s3.amazonaws.com%2Fcb%2F2b%2Fab7608ce1f477c824c31846ed2f3%2Feiffel-tower-sunrise.jpg"]
images = st.file_uploader("Don't Upload รูปตรงนี้ได้นะ")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

def is_thai(text):
    return detect(text) == 'th'

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


agent = create_json_chat_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True)

if question:
    with st.chat_message("user"):
        st.markdown(question)
    st.session_state.messages.append({"role": "user", "content": question})
    links = links_regex.findall(question)
    if len(links) > 0:
        responses = []
        loader = UnstructuredURLLoader(urls=links)
        data = loader.load()
        documents = text_splitter.split_documents(data)
        db.add_documents(documents=documents)

    if images:
        responses = []
        loader = ImageCaptionLoader(images=images.getvalue())
        img_docs = loader.load()
        db.add_documents(documents=img_docs)


    response = agent_executor.invoke({"input": question, "chat_history": st.session_state.messages})

    isthai = is_thai(response['output'])
    count = 0
    while not isthai and count != 2:
        query = f"ตอบซ้ำข้อความนี้เป็นภาษาไทย: {response['output']} ตอบโดยใช้ tool ชื่อ Final Answer เท่่านั้น"
        response = agent_executor.invoke({"input": query, "chat_history": st.session_state.messages})
        isthai = is_thai(response['output'])
        count += 1

    with st.chat_message("assistant"):
        st.markdown(response['output'])

    st.session_state.messages.append({"role": "assistant", "content": response['output']})


