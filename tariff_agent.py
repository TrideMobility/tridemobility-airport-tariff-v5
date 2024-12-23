import os
import shutil
import tempfile
import streamlit as st
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
from langchain.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool, AgentExecutor, ConversationalAgent
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document
from langchain.callbacks import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate

# ==========================
# Configuration
# ==========================
@dataclass
class Config:
    """Application configuration."""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    LLM_MODEL: str = "gpt-4o"
    EMBEDDING_MODEL: str = "text-embedding-ada-002"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    MAX_RETRIES: int = 3
    TEMPERATURE: float = 0
    TOP_K_RESULTS: int = 30
    LOG_FILE: str = "app.log"

# Initialize logging
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# ==========================
# Enhanced Prompts
# ==========================
class Prompts:
    """Collection of system prompts."""
    
    QA_PROMPT = """You are a specialized airport tariff analysis assistant. Your role is to provide precise information from airport tariff documents.

IMPORTANT RULES:
1. ONLY use information from the provided context. DO NOT use external knowledge.
2. If information isn't in the context, respond with:
   "I apologize, but I cannot find this specific information in the provided tariff documents."
3. For each piece of information, cite the specific airport, document section, and page number.
4. Present monetary values in their original currency with clear labeling.
5. When comparing airports, use tables with clear headers.

Context: {context}

Question: {question}

Response: Analyzing the tariff documents..."""

    AGENT_SYSTEM_PROMPT = """You are an expert airport tariff analyst with capabilities to:
1. Compare tariff structures across different airports
2. Break down complex fee calculations
3. Explain regulatory compliance aspects
4. Analyze historical tariff trends

Guidelines:
1. Always show detailed fee breakdowns
2. Include relevant terms and conditions
3. Highlight any seasonal variations
4. Note applicable GST/tax implications
"""

# ==========================
# Document Processing
# ==========================
class DocumentProcessor:
    """Handles document loading and processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )

    @st.cache_data
    def load_documents(_self, pdf_paths: List[str]) -> List[Document]:
        """Load and process PDF documents with enhanced error handling.
        
        Note: Using _self parameter for Streamlit caching compatibility."""
        docs = []
        for path in pdf_paths:
            try:
                loader = PyPDFLoader(path)
                pdf_docs = loader.load()
                
                if not pdf_docs:
                    raise ValueError(f"No content extracted from {path}")
                
                airport_name = Path(path).stem.split('_')[0].capitalize()
                
                for doc in pdf_docs:
                    chunks = _self.text_splitter.split_text(doc.page_content)
                    for chunk in chunks:
                        metadata = {
                            "source": path,
                            "airport": airport_name,
                            "page": doc.metadata.get("page", 0),
                            "processed_date": datetime.now().isoformat()
                        }
                        docs.append(Document(page_content=chunk, metadata=metadata))
                
                logging.info(f"Successfully processed {path}: {len(chunks)} chunks created")
                
            except Exception as e:
                logging.error(f"Error processing {path}: {str(e)}")
                st.error(f"Failed to process {path}. Error: {str(e)}")
                continue
        
        if not docs:
            raise ValueError("No documents were successfully processed")
            
        return docs

# ==========================
# Vector Store Management
# ==========================
class VectorStoreManager:
    """Manages vector store operations."""
    
    def __init__(self, config: Config):
        self.config = config
        self.embeddings = OpenAIEmbeddings(
            model=config.EMBEDDING_MODEL,
            openai_api_key=config.OPENAI_API_KEY
        )

    @st.cache_resource
    def create_vectorstore(_self, _docs: List[Document]) -> FAISS:
        """Create and configure FAISS vector store.
        
        Note: Using underscore prefix for parameters to skip Streamlit hashing."""
        return FAISS.from_documents(_docs, _self.embeddings)

    def create_retriever(self, vectorstore: FAISS):
        """Configure retriever with optimal parameters."""
        return vectorstore.as_retriever(
            search_type="similarity_score_threshold",
            search_kwargs={
                "k": self.config.TOP_K_RESULTS,
                "score_threshold": 0.7
            }
        )
    

def airport_tariff_qa_func(qa_chain, query: str) -> str:
    """
    Calls the QA chain, which returns { 'result': ..., 'source_documents': [...] }.
    Returns only the 'result' key so that the Agent does not break on multiple outputs.
    """
    raw_output = qa_chain({"query": query})
    # raw_output["source_documents"] -> list of Documents retrieved
    return raw_output["result"]


# ==========================
# Agent System
# ==========================
from langchain.prompts import PromptTemplate

class AgentSystem:
    """Manages the QA agent system."""
    
    def __init__(self, config: Config, retriever):
        self.config = config
        self.retriever = retriever
        self.llm = ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            streaming=True
        )

    def build_qa_chain(self):
        """Create the QA chain with improved configuration."""
        
        # Create a PromptTemplate from your QA_PROMPT text
        qa_prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=Prompts.QA_PROMPT
        )
        
        # Set return_source_documents=True to see which documents were used
        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self.retriever,
            return_source_documents=True,
            chain_type_kwargs={
                "prompt": qa_prompt_template
            }
        )

    def create_agent(self):
        """Create the conversational agent with enhanced capabilities."""
        qa_chain = self.build_qa_chain()
        
        # Use the custom function as the Tool
        tools = [
            Tool(
                name="AirportTariffQA",
                func=lambda query: airport_tariff_qa_func(qa_chain, query),
                description="Answers questions about airport tariffs using the provided documents.",
                return_direct=True
            )
        ]

        memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="output"
        )

        # System message from your prompts
        system_message = Prompts.AGENT_SYSTEM_PROMPT

        agent = ConversationalAgent.from_llm_and_tools(
            llm=self.llm,
            tools=tools,
            verbose=True,
            system_message=system_message
        )

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            verbose=True,
            memory=memory,
            handle_parsing_errors=True
        )


# ==========================
# Streamlit UI
# ==========================
class UI:
    """Manages the Streamlit user interface."""
    
    def __init__(self):
        st.set_page_config(
            page_title="Airport Tariff Analysis System",
            page_icon="✈️",
            layout="wide"
        )
        
    def render(self):
        """Render the main UI."""
        st.title("✈️ Airport Tariff Analysis System")
        st.markdown("""
        ### Compare and analyze tariffs across different airports
        Upload PDF documents containing airport tariffs and ask questions to analyze:
        - Fee structures and calculations
        - Cross-airport comparisons
        - Seasonal variations
        - Regulatory compliance
        """)

        # Sidebar for configuration
        with st.sidebar:
            st.header("Configuration")
            uploaded_files = st.file_uploader(
                "Upload Tariff Documents",
                type="pdf",
                accept_multiple_files=True
            )
            
            st.markdown("### Sample Questions")
            st.markdown("""
            - What are the landing charges for a 100-ton aircraft?
            - Compare parking charges between airports
            - What are the peak hour surcharges?
            """)

        return uploaded_files

def main():
    """Main application entry point."""
    try:
        config = Config()
        if not config.OPENAI_API_KEY:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            st.stop()

        ui = UI()
        uploaded_files = ui.render()

        if not uploaded_files:
            st.info("Please upload PDF documents to begin analysis.")
            return

        with st.spinner("Processing documents..."):
            # Save uploaded files to temporary location
            temp_dir = Path("temp_pdfs")
            temp_dir.mkdir(exist_ok=True)
            
            temp_paths = []
            for uploaded_file in uploaded_files:
                temp_path = temp_dir / uploaded_file.name
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                temp_paths.append(temp_path)
            
            doc_processor = DocumentProcessor(config)
            docs = doc_processor.load_documents([str(path) for path in temp_paths])

            vector_manager = VectorStoreManager(config)
            vectorstore = vector_manager.create_vectorstore(docs)
            
            # Cleanup temporary files
            for path in temp_paths:
                path.unlink(missing_ok=True)
            temp_dir.rmdir()
            retriever = vector_manager.create_retriever(vectorstore)

            agent_system = AgentSystem(config, retriever)
            agent_chain = agent_system.create_agent()

        # Query interface
        query = st.text_input("What would you like to know about the airport tariffs?")
        
        if st.button("Analyze", type="primary"):
            if not query.strip():
                st.warning("Please enter a question.")
                return

            try:
                with st.spinner("Analyzing tariffs..."):
                    callbacks = [StreamlitCallbackHandler(st.container())]
                    response = agent_chain.run(
                        input=query,
                        callbacks=callbacks
                    )
                    
                    st.markdown("### Analysis Results")
                    st.markdown(response)
                    
            except Exception as e:
                logging.error(f"Query processing error: {str(e)}")
                st.error("An error occurred while processing your query. Please try again.")
    
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        st.error(f"A critical error occurred: {str(e)}")
        st.error("Please check the logs for more details and ensure all requirements are properly installed.")

if __name__ == "__main__":
    main()