from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import StrOutputParser
from langchain.schema.runnable.config import RunnableConfig
from langchain.memory import ConversationBufferMemory, ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.runnables import RunnableLambda
from pinecone import Pinecone as pine
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_community.retrievers import PineconeHybridSearchRetriever
from operator import itemgetter
import chainlit as cl
from typing import Optional
import os
from pinecone_text.sparse import BM25Encoder



def init_vectorstore():
    # initialize the vector store object
    pc = pine(api_key=os.getenv('PINECONE_API_KEY'))
    index = pc.Index(os.getenv('PINECONE_INDEX'))

    bm25_encoder = BM25Encoder().default()
    embed_model = OpenAIEmbeddings(
                    model=os.getenv('EMBEDDINGS_MODEL'),
                    api_key=os.getenv('OPENAI_API_KEY')
                )

    vectorstore = PineconeHybridSearchRetriever(
        embeddings=embed_model, sparse_encoder=bm25_encoder, index=index, alpha=1, top_k=3
    )

    return vectorstore

# create vectorstore object
retriever=init_vectorstore()


@cl.on_chat_start
async def on_chat_start():
    store = {}

    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()
        return store[session_id]

    model = ChatOpenAI(streaming=True,
                       model=os.getenv('OPENAI_COMPLETION_MODEL'),
                       openai_api_key=os.getenv('OPENAI_API_KEY'))

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a chatbot that addresses instructions as complete and detailed as possible based on the the following context: {context}"
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "instruccion: {question}"),
        ]
    )

    output_parser = StrOutputParser()

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": itemgetter("question") | retriever | RunnableLambda(format_docs),
            "question": itemgetter("question"),
            "history": itemgetter("history")
        }
        | prompt
        | model
        | output_parser
    )

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="question",
        history_messages_key="history",
    )

    cl.user_session.set("runnable", chain_with_history)


@cl.on_message
async def on_message(message: cl.Message):
    runnable = cl.user_session.get("runnable")
    user_id = cl.user_session.get("id")

    msg = cl.Message(content="")

    for chunk in await cl.make_async(runnable.stream)(
        {"question": message.content},
        config={
            'configurable': {'session_id': user_id},
            'callbacks': [cl.LangchainCallbackHandler()]
            }
    ):
        await msg.stream_token(chunk)

    await msg.send()


@cl.password_auth_callback
def auth_callback(username: str, password: str):
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "android050828"):
        return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"}
        )
    else:
        return None
