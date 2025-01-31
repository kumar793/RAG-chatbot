from langchain_google_genai import  GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_astradb import AstraDBVectorStore
from src.exception import CustomException
from src.logger import logging
import os 
import sys
from dotenv import load_dotenv
load_dotenv()
ASTRA_DB_API_ENDPOINT = os.getenv("ASTRA_DB_API_ENDPOINT")
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")


class load_document:
    def vector_store(self,document_path):
        try:
            self.__documents = PyPDFLoader(document_path).load()
            self.__text_splitter = RecursiveCharacterTextSplitter(chunk_size = 300, chunk_overlap = 20 ).split_documents(self.__documents)
            logging.info("Document loaded and split into chunks")
            self.__embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
            self.__vector_store = AstraDBVectorStore(
                                collection_name="genint",
                                embedding=self.__embeddings,
                                api_endpoint=ASTRA_DB_API_ENDPOINT,
                                token=ASTRA_DB_APPLICATION_TOKEN)
            logging.info("Connected with AstraDB")
            self.__vector_store.add_documents(self.__text_splitter)
            logging.info("Document added to AstraDB")
            return self.__vector_store
        except Exception as e:
            raise CustomException(e, sys)
