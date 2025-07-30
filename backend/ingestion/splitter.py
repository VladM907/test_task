"""
Text splitting utilities using LangChain's text splitters.
Supports splitting by headers/sections for Markdown and simple heuristics for TXT/PDF.
"""
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from typing import List, Dict

class TextSplitter:
    @staticmethod
    def split_markdown(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
        """Split markdown text using header-aware splitting."""
        try:
            # Split by headers first - format: [(header_pattern, header_name)]
            header_splitter = MarkdownHeaderTextSplitter(
                headers_to_split_on=[
                    ("#", "Header 1"),
                    ("##", "Header 2"), 
                    ("###", "Header 3")
                ]
            )
            sections = header_splitter.split_text(text)
            
            chunks = []
            char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n\n",    # Multiple line breaks (section breaks)
                    "\n\n",      # Double line breaks (paragraph breaks)
                    ".\n\n",     # Sentence + paragraph break
                    ".\n",       # Sentence + line break
                    ". ",        # Sentence ending with space
                    "?\n",       # Question + line break
                    "? ",        # Question + space
                    "!\n",       # Exclamation + line break
                    "! ",        # Exclamation + space
                    ";\n",       # Semicolon + line break
                    "; ",        # Semicolon + space
                    ":\n",       # Colon + line break
                    ": ",        # Colon + space
                    ",\n",       # Comma + line break
                    ", ",        # Comma + space
                    "\n",        # Single line break (less preferred)
                    " ",         # Space (last resort)
                    ""           # Character split (emergency)
                ]
            )
            
            for section in sections:
                # Extract text content from Document object
                section_text = section.page_content if hasattr(section, 'page_content') else str(section)
                
                # Further split large sections
                if len(section_text) > chunk_size:
                    chunks.extend(char_splitter.split_text(section_text))
                else:
                    chunks.append(section_text)
                    
            return chunks
            
        except Exception as e:
            # Fallback to regular text splitting if markdown splitting fails
            print(f"Warning: Markdown splitting failed ({e}), falling back to text splitting")
            return TextSplitter.split_text(text, chunk_size, chunk_overlap)

    @staticmethod
    def split_text(text: str, chunk_size: int = 800, chunk_overlap: int = 100) -> List[str]:
        """Generic splitter for TXT and PDF with improved error handling."""
        try:
            char_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=[
                    "\n\n\n",    # Multiple line breaks (section breaks)
                    "\n\n",      # Double line breaks (paragraph breaks)
                    ".\n\n",     # Sentence + paragraph break
                    ".\n",       # Sentence + line break
                    ". ",        # Sentence ending with space
                    "?\n",       # Question + line break
                    "? ",        # Question + space
                    "!\n",       # Exclamation + line break
                    "! ",        # Exclamation + space
                    ";\n",       # Semicolon + line break
                    "; ",        # Semicolon + space
                    ":\n",       # Colon + line break
                    ": ",        # Colon + space
                    ",\n",       # Comma + line break
                    ", ",        # Comma + space
                    "\n",        # Single line break (less preferred)
                    " ",         # Space (last resort)
                    ""           # Character split (emergency)
                ]
            )
            
            if not text or not text.strip():
                return []
                
            return char_splitter.split_text(text)
            
        except Exception as e:
            print(f"Error splitting text: {e}")
            # Simple fallback splitting
            return [text[i:i+chunk_size] for i in range(0, len(text), chunk_size-chunk_overlap)]
