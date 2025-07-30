"""
Document loaders for PDF, Markdown, and TXT files with improved error handling.
"""
from pathlib import Path
import logging
from typing import List, Dict, Optional
from PyPDF2 import PdfReader
from .splitter import TextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentLoader:
    @staticmethod
    def load_txt(file_path: Path) -> Optional[str]:
        """Load text file with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    logger.warning(f"Empty text file: {file_path}")
                    return None
                return content
        except UnicodeDecodeError:
            try:
                # Fallback to different encoding
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
                    logger.info(f"Loaded {file_path} with latin-1 encoding")
                    return content
            except Exception as e:
                logger.error(f"Failed to load text file {file_path}: {e}")
                return None
        except Exception as e:
            logger.error(f"Failed to load text file {file_path}: {e}")
            return None

    @staticmethod
    def load_md(file_path: Path) -> Optional[str]:
        """Load markdown file with error handling."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                if not content.strip():
                    logger.warning(f"Empty markdown file: {file_path}")
                    return None
                return content
        except Exception as e:
            logger.error(f"Failed to load markdown file {file_path}: {e}")
            return None

    @staticmethod
    def load_pdf(file_path: Path) -> Optional[str]:
        """Load PDF file with improved error handling."""
        try:
            reader = PdfReader(str(file_path))
            
            if len(reader.pages) == 0:
                logger.warning(f"PDF has no pages: {file_path}")
                return None
                
            text = ""
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
                except Exception as e:
                    logger.warning(f"Failed to extract text from page {page_num} in {file_path}: {e}")
                    continue
            
            if not text.strip():
                logger.warning(f"No text extracted from PDF: {file_path}")
                return None
                
            return text
            
        except Exception as e:
            logger.error(f"Failed to load PDF file {file_path}: {e}")
            return None

    @staticmethod
    def load_documents(folder: Path, chunk_size: int = 800, chunk_overlap: int = 100) -> List[Dict]:
        """
        Load all supported documents from a folder with improved error handling.
        Returns a list of dicts, each with:
            - path: file path
            - full_content: the full document text
            - chunks: list of split text chunks
            - metadata: file metadata (size, modified_time, etc.)
        """
        docs = []
        supported_extensions = {'.txt', '.md', '.pdf'}
        
        # Get all files recursively
        all_files = []
        try:
            all_files = [f for f in folder.rglob("*") if f.is_file() and f.suffix.lower() in supported_extensions]
        except Exception as e:
            logger.error(f"Failed to scan directory {folder}: {e}")
            return docs
        
        logger.info(f"Found {len(all_files)} supported files to process")
        
        for file_path in all_files:
            try:
                logger.info(f"Processing: {file_path}")
                
                # Load content based on file type
                content = None
                if file_path.suffix.lower() == ".txt":
                    content = DocumentLoader.load_txt(file_path)
                elif file_path.suffix.lower() == ".md":
                    content = DocumentLoader.load_md(file_path)
                elif file_path.suffix.lower() == ".pdf":
                    content = DocumentLoader.load_pdf(file_path)
                
                # Skip if content loading failed
                if content is None:
                    logger.warning(f"Skipping {file_path} - failed to load content")
                    continue
                
                # Split content into chunks
                chunks = []
                try:
                    if file_path.suffix.lower() == ".md":
                        chunks = TextSplitter.split_markdown(content, chunk_size, chunk_overlap)
                    else:
                        chunks = TextSplitter.split_text(content, chunk_size, chunk_overlap)
                except Exception as e:
                    logger.error(f"Failed to split content for {file_path}: {e}")
                    continue
                
                if not chunks:
                    logger.warning(f"No chunks generated for {file_path}")
                    continue
                
                # Gather file metadata
                try:
                    stat = file_path.stat()
                    metadata = {
                        "file_size": stat.st_size,
                        "modified_time": stat.st_mtime,
                        "file_type": file_path.suffix.lower(),
                        "num_chunks": len(chunks)
                    }
                except Exception as e:
                    logger.warning(f"Failed to get metadata for {file_path}: {e}")
                    metadata = {"file_type": file_path.suffix.lower(), "num_chunks": len(chunks)}
                
                docs.append({
                    "path": str(file_path),
                    "full_content": content,
                    "chunks": chunks,
                    "metadata": metadata
                })
                
                logger.info(f"Successfully processed {file_path}: {len(chunks)} chunks")
                
            except Exception as e:
                logger.error(f"Unexpected error processing {file_path}: {e}")
                continue
        
        logger.info(f"Successfully loaded {len(docs)} documents")
        return docs
