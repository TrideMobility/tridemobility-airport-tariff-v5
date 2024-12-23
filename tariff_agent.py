import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"  # Prevent tokenizer parallelism warning
import streamlit as st
from typing import List, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import logging
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.agents import Tool, AgentExecutor
from langchain.memory import ConversationBufferMemory
from langchain.schema import Document, HumanMessage, AIMessage, SystemMessage
from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.agents.openai_functions_agent.base import OpenAIFunctionsAgent
from langchain.agents.openai_functions_agent.agent_token_buffer_memory import AgentTokenBufferMemory
from langchain.tools.retriever import create_retriever_tool

@dataclass
class Config:
    """Application configuration."""
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY")
    EMBEDDING_MODEL: str = "text-embedding-3-small"  # OpenAI's embedding model
    LLM_MODEL: str = "gpt-4o-mini"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    MAX_RETRIES: int = 3
    TEMPERATURE: float = 0
    TOP_K_RESULTS: int = 15
    LOG_FILE: str = "app.log"

# Initialize logging
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class Prompts:
    """Collection of system prompts."""
    
    SYSTEM_PROMPT = SystemMessage(
        content="""You are an expert airport tariff analyst. Your role is to provide precise information from airport tariff documents.

IMPORTANT RULES:
1. You MUST use the provided retriever tool to access information for every query.
2. Only use information from the provided context. DO NOT use external knowledge.
3. If information isn't found, respond with:
   "I apologize, but I cannot find this specific information in the provided tariff documents."
4. For each piece of information, cite the specific airport and document section.
5. Present monetary values in their original currency with clear labeling.
6. When comparing airports, use tables with clear headers."""
    )

class DocumentProcessor:
    """Handles document loading and processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        # Create uploads directory if it doesn't exist
        self.uploads_dir = Path("uploads")
        self.uploads_dir.mkdir(exist_ok=True)

    def save_uploaded_file(self, uploaded_file) -> Path:
        """Save an uploaded file and return its path."""
        file_path = self.uploads_dir / uploaded_file.name
        if not file_path.exists():  # Only save if file doesn't exist
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            logging.info(f"Saved new file: {file_path}")
        return file_path

    @st.cache_data
    def load_documents(_self, pdf_paths: List[str]) -> List[Document]:
        """Load and process PDF documents with enhanced error handling."""
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

class VectorStoreManager:
    """Manages vector store operations."""
    
    def __init__(self, config: Config):
        self.config = config
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                openai_api_key=config.OPENAI_API_KEY
            )
        except Exception as e:
            logging.error(f"Error initializing embeddings: {str(e)}")
            raise

    @st.cache_resource
    def create_vectorstore(_self, _docs: List[Document]) -> FAISS:
        """Create and configure FAISS vector store."""
        return FAISS.from_documents(_docs, _self.embeddings)

class AgentSystem:
    """Manages the QA agent system."""
    
    def __init__(self, config: Config, vectorstore: FAISS):
        self.config = config
        self.vectorstore = vectorstore
        self.llm = ChatOpenAI(
            openai_api_key=config.OPENAI_API_KEY,
            model=config.LLM_MODEL,
            temperature=config.TEMPERATURE,
            streaming=True
        )

    def create_agent(self):
        """Create the agent with enhanced capabilities."""
        # Create a retriever tool using the vectorstore
        retriever = self.vectorstore.as_retriever(
            search_kwargs={"k": self.config.TOP_K_RESULTS}
        )
        
        # Create the retriever tool with more specific description
        retriever_tool = create_retriever_tool(
            retriever,
            name="search_tariffs",
            description="""Use this tool to search through airport tariff documents. 
            Always use this tool first to find relevant information before providing an answer.
            The tool will return the most relevant passages from the tariff documents."""
        )

        tools = [retriever_tool]

        # Initialize memory with token limit
        memory = AgentTokenBufferMemory(
            memory_key="chat_history",
            llm=self.llm,
            max_token_limit=2000,
            return_messages=True
        )

        # Create the agent with more specific instructions
        prompt = SystemMessage(content="""You are an expert airport tariff analyst. Your role is to provide precise information from airport tariff documents.

IMPORTANT RULES:
1. ALWAYS use the search_tariffs tool first to find relevant information before answering.
2. Only use information found in the tariff documents. DO NOT make assumptions or use external knowledge.
3. If the search tool doesn't return relevant information, say: "I apologize, but I cannot find this specific information in the provided tariff documents."
4. Always cite your sources by mentioning the specific airport, document section, and page number.
5. For monetary values, include the original currency and clear labeling.
6. When comparing airports, present information in a clear, structured format.
7. If the search results are unclear or seem incomplete, perform another search with different keywords.""")

        agent = OpenAIFunctionsAgent.from_llm_and_tools(
            llm=self.llm,
            tools=tools,
            system_message=prompt
        )

        executor = AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=3,
            verbose=True,
            return_intermediate_steps=True  # Enable intermediate steps
        )
        
        return executor

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
            doc_processor = DocumentProcessor(config)
            
            # Save uploaded files persistently
            saved_paths = []
            for uploaded_file in uploaded_files:
                file_path = doc_processor.save_uploaded_file(uploaded_file)
                saved_paths.append(str(file_path))
            
            # Load and process documents
            docs = doc_processor.load_documents(saved_paths)

            vector_manager = VectorStoreManager(config)
            vectorstore = vector_manager.create_vectorstore(docs)
            
            agent_system = AgentSystem(config, vectorstore)
            agent_chain = agent_system.create_agent()

        # Display already processed files
        if saved_paths:
            st.sidebar.markdown("### Processed Documents")
            for path in saved_paths:
                st.sidebar.text(Path(path).name)

        query = st.text_input("What would you like to know about the airport tariffs?")
        
        if st.button("Analyze", type="primary"):
            if not query.strip():
                st.warning("Please enter a question.")
                return

            try:
                with st.spinner("Analyzing tariffs..."):
                    callbacks = [StreamlitCallbackHandler(st.container())]
                    
                    # Invoke the agent with proper input format and return intermediate steps
                    response = agent_chain.invoke(
                        {
                            "input": query,
                            "chat_history": []
                        },
                        config={
                            "callbacks": callbacks
                        }
                    )
                    
                    st.markdown("### Analysis Results")
                    
                    # Display intermediate steps if present
                    # if "intermediate_steps" in response:
                    #     st.markdown("#### Thought Process:")
                    #     for step in response["intermediate_steps"]:
                    #         action = step[0]
                    #         st.markdown(f"**Tool:** {action.tool}")
                    #         st.markdown(f"**Input:** {action.tool_input}")
                    #         st.markdown(f"**Output:** {step[1]}")
                    #         st.markdown("---")
                    
                    # Display final output
                    st.markdown("#### Final Response:")
                    if isinstance(response, dict) and "output" in response:
                        st.markdown(response["output"])
                    else:
                        st.markdown(str(response))
                    
            except Exception as e:
                logging.error(f"Query processing error: {str(e)}")
                # st.error("An error occurred while processing your query. Please try again.")
    
    except Exception as e:
        logging.error(f"Application error: {str(e)}")
        st.error(f"A critical error occurred: {str(e)}")
        st.error("Please check the logs for more details and ensure all requirements are properly installed.")

if __name__ == "__main__":
    main()