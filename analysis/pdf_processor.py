"""
PDF text extraction and processing utilities
"""
import logging
import re
from typing import Dict, Any, Tuple

import PyPDF2
import pdfplumber

logger = logging.getLogger(__name__)

class PDFProcessor:
    """Handles PDF text extraction and processing"""
    
    @staticmethod
    def extract_text(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Extract text and metadata from PDF"""
        full_text = ""
        metadata = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = []
                for page in pdf.pages:
                    text = page.extract_text()
                    if text:
                        pages.append(text)
                
                full_text = "\n\n".join(pages)
                metadata['num_pages'] = len(pdf.pages)
                
        except Exception as e:
            logger.warning(f"pdfplumber failed, trying PyPDF2: {e}")
            
            try:
                with open(pdf_path, 'rb') as file:
                    pdf_reader = PyPDF2.PdfReader(file)
                    pages = []
                    
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            pages.append(text)
                    
                    full_text = "\n\n".join(pages)
                    metadata['num_pages'] = len(pdf_reader.pages)
                    
            except Exception as e2:
                logger.error(f"Both PDF extraction methods failed: {e2}")
        
        # Basic text processing
        full_text = PDFProcessor._clean_text(full_text)
        metadata.update(PDFProcessor._analyze_text(full_text))
        
        return full_text, metadata
    
    @staticmethod
    def _clean_text(text: str) -> str:
        """Clean and normalize extracted text"""
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        return text.strip()
    
    @staticmethod
    def _analyze_text(text: str) -> Dict[str, Any]:
        """Analyze text for basic metrics"""
        words = text.split()
        math_patterns = [r'\$[^$]+\$', r'\\[a-zA-Z]+', r'[0-9]+\.[0-9]+']
        math_count = sum(len(re.findall(pattern, text)) for pattern in math_patterns)
        ref_count = len(re.findall(r'\[\d+\]|\(\d{4}\)', text))
        
        return {
            'word_count': len(words),
            'char_count': len(text),
            'math_expressions': math_count,
            'reference_count': ref_count,
            'avg_sentence_length': len(words) / max(text.count('.'), 1)
        }