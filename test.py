import os
# Imports for multi-turn chat memory
from langchain_classic.chains import create_history_aware_retriever, create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Standard RAG imports
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma

# 1. SETUP
os.environ["GOOGLE_API_KEY"] = "AIzaSyB-fqeetfgxYHUvPOntkKlrZsQGNZECQjw"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

def initialize_kb():
    folder_path = './knowledge_base'
    if not os.path.exists(folder_path): os.makedirs(folder_path)
    
    if not any(os.scandir(folder_path)):
        print(f"\n[!] Drop a PDF/TXT into '{folder_path}' then restart.")
        return None

    print("--- Building Memory from Documents ---")
    pdf_loader = DirectoryLoader(folder_path, glob="./*.pdf", loader_cls=PyPDFLoader)
    txt_loader = DirectoryLoader(folder_path, glob="./*.txt", loader_cls=TextLoader)
    docs = pdf_loader.load() + txt_loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
    chunks = splitter.split_documents(docs)
    
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    return vectorstore.as_retriever()

def create_conversational_rag_chain(retriever):
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3)

    # A. Contextualize Question: Rephrases follow-ups to be "standalone"
    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question "
        "which might reference context in the chat history, "
        "formulate a standalone question which can be understood "
        "without the chat history."
    )
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    # B. Answer Question: The actual response prompt
    system_prompt = (
        "You are a helpful assistant. Use the retrieved context to answer the question.\n\n"
        "{context}"
    )
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)
    return create_retrieval_chain(history_aware_retriever, question_answer_chain)

# --- EXECUTION ---
if __name__ == "__main__":
    retriever = initialize_kb()
    if retriever:
        rag_chain = create_conversational_rag_chain(retriever)
        chat_history = [] # This stores your conversation
        
        print("\n--- CHATBOT READY (REMEMBERS PREVIOUS MESSAGES) ---")
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ['exit', 'quit']: break
            
            # Send current question + previous history
            response = rag_chain.invoke({"input": user_input, "chat_history": chat_history})
            
            # Update history with the new turn
            chat_history.append(HumanMessage(content=user_input))
            chat_history.append(AIMessage(content=response["answer"]))
            
            print(f"AI: {response['answer']}")