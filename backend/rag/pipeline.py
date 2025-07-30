"""
RAG Pipeline using LangChain with Ollama and OpenAI support.
Integrates with the existing hybrid retrieval system (vector + knowledge graph).
"""
import os
from typing import List, Dict, Optional, Union
from enum import Enum
import logging

# LangChain imports
from langchain.schema import Document
from langchain.prompts import ChatPromptTemplate, PromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
from langchain.callbacks.base import BaseCallbackHandler

# LLM imports
try:
    from langchain_openai import ChatOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False
    print("OpenAI not available. Install with: pip install langchain-openai")

try:
    from langchain_community.llms import Ollama
    from langchain_ollama import ChatOllama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False
    print("Ollama not available. Install with: pip install langchain-community")

# Local imports
from ..retrieval.hybrid_retriever import HybridRetriever

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelProvider(Enum):
    OPENAI = "openai"
    OLLAMA = "ollama"

class StreamingCallback(BaseCallbackHandler):
    """Callback handler for streaming responses."""
    
    def __init__(self):
        self.tokens = []
    
    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.tokens.append(token)
        print(token, end="", flush=True)

class RAGPipeline:
    def __init__(
        self,
        provider: ModelProvider = ModelProvider.OLLAMA,
        model_name: str = "llama3.1",
        temperature: float = 0.1,
        retriever: Optional[HybridRetriever] = None,
        openai_api_key: Optional[str] = None
    ):
        """
        Initialize RAG pipeline with specified LLM provider.
        
        Args:
            provider: Model provider (OPENAI or OLLAMA)
            model_name: Name of the model to use
            temperature: Sampling temperature
            retriever: Hybrid retriever instance
            openai_api_key: OpenAI API key (if using OpenAI)
        """
        self.provider = provider
        self.model_name = model_name
        self.temperature = temperature
        self.retriever = retriever or HybridRetriever()
        
        # Initialize LLM
        self.llm = self._setup_llm(openai_api_key)
        
        # Setup prompts
        self.qa_prompt = self._create_qa_prompt()
        self.contextualize_prompt = self._create_contextualize_prompt()
        
        # Create the RAG chain
        self.rag_chain = self._create_rag_chain()
        
        logger.info(f"Initialized RAG pipeline with {provider.value} ({model_name})")

    def _setup_llm(self, openai_api_key: Optional[str]) -> Union['ChatOpenAI', 'ChatOllama', 'Ollama']:
        """Setup the LLM based on provider."""
        if self.provider == ModelProvider.OPENAI:
            if not OPENAI_AVAILABLE:
                raise ImportError("OpenAI dependencies not installed")
            
            api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key required")
            
            return ChatOpenAI(
                model=self.model_name,
                temperature=self.temperature,
                api_key=api_key # type: ignore
            )
        
        elif self.provider == ModelProvider.OLLAMA:
            if not OLLAMA_AVAILABLE:
                raise ImportError("Ollama dependencies not installed")
            
            # Try ChatOllama first (for conversation), fallback to Ollama
            try:
                return ChatOllama(
                    model=self.model_name,
                    temperature=self.temperature,
                    base_url="http://localhost:11434"  # Default Ollama URL
                )
            except Exception as e:
                logger.warning(f"ChatOllama failed, using Ollama: {e}")
                return Ollama(
                    model=self.model_name,
                    temperature=self.temperature,
                    base_url="http://localhost:11434"
                )
        
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

    def _create_qa_prompt(self) -> ChatPromptTemplate:
        """Create the main QA prompt template."""
        template = """You are a helpful AI assistant that answers questions based on the provided context. Use the context information to provide accurate, detailed answers. No need to always mention that answers are based on context, just provide the information directly. Only mention the context if it is relevant to the question or your providing citation.

Context Information:
{context}

Question: {question}

Instructions:
1. Answer based primarily on the provided context
2. If the context doesn't contain enough information, say so clearly
3. Use the entity information to understand the domain and relationships
4. When citing sources, use the document names shown in the context headers (e.g., "According to [Document Name]..." or "As stated in [Document Name]...")
5. If you're unsure, express uncertainty rather than guessing

Answer:"""

        return ChatPromptTemplate.from_template(template)

    def _create_contextualize_prompt(self) -> ChatPromptTemplate:
        """Create prompt for contextualizing follow-up questions."""
        template = """Given a chat history and the latest user question which might reference the chat history, formulate a standalone question which can be understood without the chat history. Do NOT answer the question, just reformulate it if needed and otherwise return it as is.

Chat History:
{chat_history}

Follow Up Input: {question}
Standalone Question:"""

        return ChatPromptTemplate.from_template(template)

    def _create_rag_chain(self):
        """Create the RAG processing chain."""
        def format_docs(docs):
            """Format retrieved documents for the prompt."""
            formatted_docs = []
            
            for i, doc in enumerate(docs):
                # Extract content and metadata
                content = doc.get("content", "")
                similarity = doc.get("similarity", "N/A")
                metadata = doc.get("metadata", {})
                source = metadata.get("path", "unknown")
                
                # Extract document name from path
                import os
                if source != "unknown":
                    doc_name = os.path.basename(source)
                    # Remove file extension for cleaner display
                    doc_name = os.path.splitext(doc_name)[0]
                else:
                    doc_name = "unknown"
                
                # Extract graph context
                graph_context = doc.get("graph_context", {})
                entities = graph_context.get("entities", [])
                related_docs = graph_context.get("related_documents", [])
                
                # Format entity information
                entity_names = []
                if entities:
                    for entity in entities[:5]:  # Top 5 entities
                        if isinstance(entity, dict):
                            entity_names.append(entity.get("name", str(entity)))
                        else:
                            entity_names.append(str(entity))
                
                # Build formatted document
                formatted_doc = f"--- Document {i+1}: {doc_name} ---\n"
                formatted_doc += f"Content: {content}\n"
                formatted_doc += f"Source: {source} (similarity: {similarity})\n"
                
                if entity_names:
                    formatted_doc += f"Key Entities: {', '.join(entity_names)}\n"
                
                if related_docs:
                    formatted_doc += f"Related Documents: {len(related_docs)} documents\n"
                
                formatted_docs.append(formatted_doc)
            
            return "\n\n".join(formatted_docs)
        
        def retrieve_docs(input_dict):
            """Retrieve documents using hybrid retrieval."""
            # Extract question from input
            if isinstance(input_dict, dict):
                question = input_dict.get("question", "")
            else:
                question = str(input_dict)
            
            if not question or not isinstance(question, str):
                logger.error(f"Invalid question format: {type(question)} - {question}")
                return []
            
            logger.debug(f"Retrieving docs for question: '{question}'")
            
            # Perform hybrid search
            results = self.retriever.search(
                query=question,
                n_results=5,
                use_graph_expansion=True,
                graph_expansion_limit=3
            )
            
            return results
        
        # Create the chain
        rag_chain = (
            {
                "context": lambda x: format_docs(retrieve_docs(x)), 
                "question": lambda x: x.get("question", "") if isinstance(x, dict) else str(x)
            }
            | self.qa_prompt
            | self.llm
            | StrOutputParser()
        )
        
        return rag_chain

    def ask(self, question: str, stream: bool = False) -> str:
        """
        Ask a question and get an answer from the RAG system.
        
        Args:
            question: The question to ask
            stream: Whether to stream the response
            
        Returns:
            The answer from the RAG system
        """
        try:
            logger.info(f"Processing question: {question}")
            
            if stream:
                # For streaming responses
                callback = StreamingCallback()
                response = self.rag_chain.invoke(
                    {"question": question},
                    config={"callbacks": [callback]}
                )
                return response
            else:
                # Regular response
                response = self.rag_chain.invoke({"question": question})
                return response
                
        except Exception as e:
            logger.error(f"Error processing question: {e}")
            return f"Sorry, I encountered an error while processing your question: {str(e)}"

    def ask_with_context(self, question: str, chat_history: Optional[List[Dict]] = None) -> str:
        """
        Ask a question with chat history context.
        
        Args:
            question: The current question
            chat_history: Previous conversation history
            
        Returns:
            The answer from the RAG system
        """
        if not chat_history:
            return self.ask(question)
        
        try:
            # For context-aware questions, we'll modify the retrieval and prompt
            # First, try to determine if this is a context-dependent question
            context_indicators = ['my last', 'previous', 'earlier', 'before', 'that', 'it', 'this']
            is_context_dependent = any(indicator in question.lower() for indicator in context_indicators)
            
            if is_context_dependent:
                # For context-dependent questions, include chat history in the prompt
                return self._ask_with_chat_history(question, chat_history)
            else:
                # For regular questions, just use normal RAG
                return self.ask(question)
            
        except Exception as e:
            logger.error(f"Error with contextual question: {e}")
            return self.ask(question)  # Fallback to regular ask

    def _ask_with_chat_history(self, question: str, chat_history: List[Dict]) -> str:
        """Handle context-dependent questions with chat history."""
        try:
            # Format recent chat history
            history_str = ""
            for entry in chat_history[-3:]:  # Last 3 exchanges
                if "human" in entry:
                    history_str += f"Previous Question: {entry['human']}\n"
                if "assistant" in entry:
                    history_str += f"Previous Answer: {entry['assistant'][:200]}...\n"
            
            # Create a context-aware prompt
            context_prompt = f"""You are a helpful AI assistant. You have access to document context and previous conversation history.

Previous Conversation:
{history_str}

Current Question: {question}

Instructions:
1. If the current question refers to the previous conversation (using words like "my last question", "that", "it", etc.), use the conversation history to understand what the user is referring to
2. Use the document context below to provide accurate information
3. Be specific about what the user was asking about previously if relevant

Document Context:
{{context}}

Answer:"""

            # Create a temporary prompt template
            temp_prompt = ChatPromptTemplate.from_template(context_prompt)
            
            # Get retrieval context (use a more general query for context-dependent questions)
            if "last question" in question.lower() or "previous" in question.lower():
                # Use the last question from history for retrieval
                last_question = ""
                for entry in reversed(chat_history):
                    if "human" in entry:
                        last_question = entry["human"]
                        break
                retrieval_query = last_question if last_question else question
            else:
                retrieval_query = question
            
            # Get documents
            results = self.retriever.search(
                query=retrieval_query,
                n_results=3,  # Fewer results for context questions
                use_graph_expansion=True,
                graph_expansion_limit=2
            )
            
            # Format documents
            formatted_docs = []
            import os
            for i, doc in enumerate(results):
                content = doc.get("content", "")
                metadata = doc.get("metadata", {})
                source = metadata.get("path", "unknown")
                
                # Extract document name from path
                if source != "unknown":
                    doc_name = os.path.basename(source)
                    doc_name = os.path.splitext(doc_name)[0]  # Remove extension
                else:
                    doc_name = "unknown"
                    
                formatted_docs.append(f"Document: {doc_name}\nContent: {content[:300]}...\nSource: {source}")
            
            context_str = "\n\n".join(formatted_docs)
            
            # Create the final prompt
            final_prompt = temp_prompt.format(context=context_str)
            
            # Get response from LLM
            response = self.llm.invoke(final_prompt)
            
            # Extract text from response
            if hasattr(response, 'content'):
                return str(response.content)
            else:
                return str(response)
                
        except Exception as e:
            logger.error(f"Error in context-aware chat: {e}")
            # Fallback: provide info about the conversation
            if chat_history:
                last_question = ""
                for entry in reversed(chat_history):
                    if "human" in entry:
                        last_question = entry["human"]
                        break
                return f"Your last question was: '{last_question}'" if last_question else "I don't have access to previous questions in this conversation."
            return "I don't have access to previous conversation history."

    def get_retrieval_info(self, question: str) -> Dict:
        """Get detailed information about what was retrieved for a question."""
        try:
            results = self.retriever.search(
                query=question,
                n_results=5,
                use_graph_expansion=True,
                graph_expansion_limit=3
            )
            
            return {
                "question": question,
                "num_results": len(results),
                "results": results,
                "retrieval_stats": self.retriever.get_statistics()
            }
            
        except Exception as e:
            logger.error(f"Error getting retrieval info: {e}")
            return {"error": str(e)}

    def close(self):
        """Close connections."""
        if hasattr(self.retriever, 'close'):
            self.retriever.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
