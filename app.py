from flask import Flask, render_template, redirect, jsonify, request, session
from flask_session import Session
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from src.components import data_ingestion
from src.exception import CustomException
from src.logger import logging
import sys

app = Flask(__name__)
app.config['SECRET_KEY'] = 'getint'
app.config['SESSION_TYPE'] = 'filesystem'
Session(app)
logging.info("Session started")
store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    global store
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

@app.route('/')
def index():
    try:
        if 'messages' not in session:
            session['messages'] = [
                {"role": "assistant", "content": "Hi, I am a web search chatbot. How can I help you?"}
            ]
            logging.info("Session started")
        return render_template('index.html', messages=session['messages'])
    except Exception as e: 
        logging.error(f"Error: {e} at initializing the session")
        raise CustomException(e, sys)

@app.route("/create_document", methods=["GET", "POST"])
def upload_documents():
    try:
       
        session.uploaded_files = request.files.getlist("file")
        session['session_id'] = request.form.get("session_id", "default_session")
        
        if session.uploaded_files:
            session.r = data_ingestion.Response()
            for uploaded_file in session.uploaded_files:
                temppdf = './temp.pdf'
                with open(temppdf, 'wb') as file:
                    file.write(uploaded_file.read())
                session.r.create_embeddings(temppdf)
            logging.info("Document uploaded successfully and create embeddings.")
            session.modified = True 
            return redirect("/")
    except Exception as e:
        logging.info(f"Error: {e} at document upload.")
        raise CustomException(e, sys)
    
@app.route("/create_rag", methods=["POST","GET"])
def create_rag():
    session.conversational_rag_chain = RunnableWithMessageHistory(
                session.r.create_response(),
                get_session_history,
                input_messages_key="input",
                history_messages_key="chat_history",
                output_messages_key="answer"
            )
    session.modified = True 
    
    logging.info("Rag chain created successfully.",session.conversational_rag_chain,session.SECRET_KEY,session.SESSION_TYPE)
    return redirect("/")
@app.route("/chat", methods=["POST"])


def DocumentQA():
    try:
        user_input = request.json.get('message')
        logging.info("received user input")
        if not user_input:
            return jsonify({"error": "No input provided"}), 400

        if 'messages' not in session:
            session['messages'] = []

        session['messages'].append({"role": "user", "content": user_input})

        session_id = session.get('session_id', 'default_session')
        session_history = get_session_history(session_id)
        logging.info("Session history created successfully.")
        logging.info("Chat response generation started.")
        response = session.conversational_rag_chain.invoke(
            {"input": user_input},
            config={"configurable": {"session_id": session_id}}
        )
        logging.info("Chat response generated successfully.")
        session.modified = True 

        # Extract the actual response content
        assistant_response = response.get('answer', 'Sorry, I could not generate a response.')

        # Append the assistant's response to the session messages
        session['messages'].append({"role": "assistant", "content": assistant_response})

        return jsonify({"messages": session['messages']})
    except Exception as e:
        logging.error(f"Error: {e} at chat response.")
        raise CustomException(e, sys)
        
if __name__ == "__main__":
    app.run(debug=True)
