import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import streamlit as st
import streamlit.components.v1 as components



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
    EMBEDDING_MODEL: str = "text-embedding-3-small"
    LLM_MODEL: str = "gpt-4o-mini"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 100
    MAX_RETRIES: int = 3
    TEMPERATURE: float = 0
    TOP_K_RESULTS: int = 15
    LOG_FILE: str = "app.log"
    DATA_DIR: str = "data"  # Directory for PDF files
    VECTORSTORE_DIR: str = "vectorstore"  # Directory for storing vectorstore
    LAST_UPDATE_FILE: str = "last_update.txt"  # File to track last update time

# Initialize logging
logging.basicConfig(
    filename=Config.LOG_FILE,
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

class DocumentProcessor:
    """Handles document loading and processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=config.CHUNK_SIZE,
            chunk_overlap=config.CHUNK_OVERLAP
        )
        # Ensure data directory exists
        self.data_dir = Path(config.DATA_DIR)
        self.data_dir.mkdir(exist_ok=True)

    def get_pdf_files(self) -> List[str]:
        """Get all PDF files from the data directory."""
        return [str(f) for f in self.data_dir.glob("*.pdf")]

    def load_documents(self) -> List[Document]:
        """Load and process PDF documents from the data directory."""
        pdf_paths = self.get_pdf_files()
        if not pdf_paths:
            raise ValueError("No PDF files found in the data directory")

        docs = []
        for path in pdf_paths:
            try:
                loader = PyPDFLoader(path)
                pdf_docs = loader.load()
                
                if not pdf_docs:
                    raise ValueError(f"No content extracted from {path}")
                
                airport_name = Path(path).stem.split('_')[0].capitalize()
                
                for doc in pdf_docs:
                    chunks = self.text_splitter.split_text(doc.page_content)
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
        self.vectorstore_path = Path(config.VECTORSTORE_DIR)
        self.vectorstore_path.mkdir(exist_ok=True)
        self.last_update_file = Path(config.VECTORSTORE_DIR) / config.LAST_UPDATE_FILE
        
        try:
            self.embeddings = OpenAIEmbeddings(
                model=self.config.EMBEDDING_MODEL,
                openai_api_key=config.OPENAI_API_KEY
            )
        except Exception as e:
            logging.error(f"Error initializing embeddings: {str(e)}")
            raise

    def get_data_files_hash(self) -> str:
        """Get a hash of all PDF files in the data directory."""
        import hashlib
        data_dir = Path(self.config.DATA_DIR)
        if not data_dir.exists():
            return ""
        
        pdf_files = sorted(data_dir.glob("*.pdf"))
        hash_content = []
        for pdf_file in pdf_files:
            hash_content.extend([
                pdf_file.name,
                str(pdf_file.stat().st_mtime_ns)
            ])
        return hashlib.md5("|".join(hash_content).encode()).hexdigest()

    def should_rebuild_vectorstore(self) -> bool:
        """Check if vectorstore needs to be rebuilt."""
        if not list(self.vectorstore_path.glob("*.faiss")):
            return True
            
        if not self.last_update_file.exists():
            return True
            
        current_hash = self.get_data_files_hash()
        stored_hash = self.last_update_file.read_text().strip() if self.last_update_file.exists() else ""
        
        return current_hash != stored_hash

    def save_vectorstore(self, vectorstore: FAISS):
        """Save vectorstore and update hash."""
        vectorstore.save_local(str(self.vectorstore_path))
        self.last_update_file.write_text(self.get_data_files_hash())
        logging.info("Vectorstore saved successfully")

    def load_vectorstore(self) -> FAISS:
        """Load existing vectorstore."""
        return FAISS.load_local(str(self.vectorstore_path), self.embeddings,allow_dangerous_deserialization=True)

    def create_vectorstore(self, docs: List[Document]) -> FAISS:
        """Create new FAISS vector store."""
        vectorstore = FAISS.from_documents(docs, self.embeddings)
        self.save_vectorstore(vectorstore)
        return vectorstore

@st.cache_resource
def get_vectorstore(config: Config) -> FAISS:
    """Get or create vectorstore."""
    vector_manager = VectorStoreManager(config)
    
    try:
        if vector_manager.should_rebuild_vectorstore():
            logging.info("Building new vectorstore...")
            doc_processor = DocumentProcessor(config)
            docs = doc_processor.load_documents()
            vectorstore = vector_manager.create_vectorstore(docs)
            logging.info("New vectorstore created and saved")
        else:
            logging.info("Loading existing vectorstore...")
            vectorstore = vector_manager.load_vectorstore()
            logging.info("Existing vectorstore loaded")
        
        return vectorstore
    except Exception as e:
        logging.error(f"Error with vectorstore: {str(e)}")
        raise

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

        return AgentExecutor.from_agent_and_tools(
            agent=agent,
            tools=tools,
            memory=memory,
            handle_parsing_errors=True,
            max_iterations=3,
            verbose=True
        )
    

class UI:
    """Manages the Streamlit user interface."""
    
    def __init__(self, config: Config):
        self.config = config
        st.set_page_config(
            page_title="Tride Mobility Airport Tariff Analysis System",
            layout="wide"
        )
    
    def render(self):
        """Render the main UI."""
        # Hide Streamlit's default elements
        hide_streamlit_style = """
            <style>
                #MainMenu {visibility: hidden;}
                footer {visibility: hidden;}
                .stDeployButton {display: none;}
            </style>
        """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)
        
        # Render custom header with images
        # Render custom header with images
        st.markdown(
        """
        <div style="display: flex; align-items: center;">
            <img src="https://raw.githubusercontent.com/DNAdithya/Tride_ML_Models/main/tride_logo.png" 
             style="width: 100px; margin-right: 20px;" alt="Tride Logo">
            <img src="https://raw.githubusercontent.com/DNAdithya/Tride_ML_Models/main/gmr_logo.png" 
             style="width: 100px;" alt="GMR Logo">
        </div>
        """,
        unsafe_allow_html=True
        )

        
        st.title("Tride Mobility Airport Tariff Analysis System")
        st.markdown("""
        ### Compare and analyze tariffs across different airports
        
        System automatically processes PDF documents from the data directory to analyze:
        - Fee structures and calculations
        - Cross-airport comparisons
        - Seasonal variations
        - Regulatory compliance
        """)
        
        with st.sidebar:
            st.header("Available Documents")
            # Display PDF files from data directory
            pdf_files = DocumentProcessor(self.config).get_pdf_files()
            
            if pdf_files:
                st.markdown("### Processed Documents")
                for path in pdf_files:
                    st.text(Path(path).name)
            else:
                st.warning("No PDF files found in data directory")
            
            st.markdown("### Sample Questions")
            st.markdown("""
            - How does base airport charges varies between Hyderabad and Delhi ?
            - what is the operating expenses of the delhi and hyderbad airports  ?
            - what is the traffic forecast  of the three aiports ?
            """)

def main():
    """Main application entry point."""
    try:
        config = Config()
        if not config.OPENAI_API_KEY:
            st.error("OpenAI API key not found. Please set the OPENAI_API_KEY environment variable.")
            st.stop()

        ui = UI(config)
        ui.render()

        # Get or create vectorstore (cached for the session)
        try:
            vectorstore = get_vectorstore(config)
        except ValueError as e:
            st.error(str(e))
            st.info("Please add PDF files to the 'data' directory and restart the application.")
            return
        except Exception as e:
            st.error(f"Error initializing the system: {str(e)}")
            return

        # Initialize agent system
        agent_system = AgentSystem(config, vectorstore)
        agent_chain = agent_system.create_agent()

        query = st.text_input("What would you like to know about the airport tariffs?")
        
        if st.button("Analyze", type="primary"):
            if not query.strip():
                st.warning("Please enter a question.")
                return

            try:
                with st.spinner("Analyzing tariffs..."):
                    callbacks = [StreamlitCallbackHandler(st.container())]
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