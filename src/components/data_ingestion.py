import os
import sys
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate,MessagesPlaceholder
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_history_aware_retriever
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from src.utils import load_document
from src.exception import CustomException
from src.logger import logging
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GOOGLE_API_KEY"]=os.getenv("GOOGLE_API_KEY")
os.environ["LANGCHAIN_PROJECT"]= "Project with gemini flash"

class Response:
    def __init__(self):
        try:
            self.llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
            self.__chat_history = [
    {"role": "user", "content": "Can you tell me about your experience with project management?"},
    {"role": "assistant", "content": "I have over five years of experience in project management. I have successfully led multiple projects from inception to completion, ensuring they were delivered on time and within budget. For example, at XYZ Corporation, I managed a team of 10 to develop a new software application, which resulted in a 20% increase in productivity for the client. Additionally, I am proficient in using project management tools like Jira and Trello to streamline workflows and track progress."}
]
            
           
            self.__contextualize_system_q_prompt = (
            "Given a chat history and the latest user question, "
            "which might reference context in the chat history, "
            "formulate a standalone question which can be understood "
            "without the chat history. DO NOT answer the question, "
            "just reformulate it if needed and otherwise return as it is."
           )

            self.__system_prompt = (
                "Please provide the answer for the given question and also personalize the answer based on the given context, including details from the resume. "
                "Please provide the most accurate response based on the question. "
                "Use the following pieces of retrieved context, including context details, to answer the question. "
                "If you don't know the answer, say that you don't know."
                "\n\n"
                "{context}"
            )


            self.__qa_prompt = ChatPromptTemplate.from_messages([
                ("system",self.__system_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}"),]
            )

            self.__contextual_q_prompt = ChatPromptTemplate.from_messages([
                ("system",self.__contextualize_system_q_prompt),
                MessagesPlaceholder("chat_history"),
                ("human","{input}")
            ])
            
            logging.info("Connected with Google Generative AI")
            """
            self.__prompt_template = ChatPromptTemplate.from_messages([
                ("system", "You are a Question-Answering chatbot. Please provide the answer for the given question as a interviee and also personalize the answer based on the given context, including details from the resume. Please provide the most accurate response based on the question. Use the following pieces of retrieved context, including resume details, to answer the question. If you don't know the answer, say that you don't know."),
                MessagesPlaceholder("chat_history"),
                ("human", "{input}")
            ])"""


            self.__memory = ConversationBufferMemory()
            logging.info("Prompts created")
        except Exception as e:
            raise CustomException(e, sys)
        
    def create_embeddings(self, document):
        try:
            self.__vs_local = load_document()
            self._vector = self.__vs_local.vector_store(document)
            logging.info("Document loaded and vectorized")
            return self._vector
        except Exception as e:
            logging.error(f"Error in create_embeddings: {e}")
            raise CustomException(e, sys)

    def create_response(self):
        try:
            if not hasattr(self, '_vector'):
                raise ValueError("Vector not initialized. Call create_embeddings first.")
            self.__retriever = self._vector.as_retriever()

            history_aware_retriever = create_history_aware_retriever(self.llm,self.__retriever,self.__contextual_q_prompt)
            question_answer_chain = create_stuff_documents_chain(self.llm,self.__qa_prompt)
            self.rag_chain = create_retrieval_chain(history_aware_retriever,question_answer_chain)
            
            
            return self.rag_chain
        except Exception as e:
            logging.error(f"Error in create_response: {e}")
            raise CustomException(e, sys)
