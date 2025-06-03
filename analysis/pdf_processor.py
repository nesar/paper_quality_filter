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



class EnhancedPDFProcessor(PDFProcessor):
    """Enhanced PDF processor for better physics text extraction"""
    
    @staticmethod
    def extract_text_enhanced(pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Enhanced text extraction optimized for physics papers"""
        full_text = ""
        metadata = {}
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = []
                for page_num, page in enumerate(pdf.pages):
                    # Extract text with layout preservation
                    text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True,
                        x_density=7.25,
                        y_density=13
                    )
                    
                    if text:
                        # Clean and process the page text
                        cleaned_text = EnhancedPDFProcessor._clean_text_enhanced(text)
                        if cleaned_text and len(cleaned_text.strip()) > 50:
                            pages.append(cleaned_text)
                
                # Join pages with proper spacing
                full_text = "\n\n".join(pages)
                metadata['num_pages'] = len(pdf.pages)
                metadata['extraction_method'] = 'pdfplumber_enhanced'
                
        except Exception as e:
            logger.warning(f"Enhanced pdfplumber failed, trying standard method: {e}")
            
            # Fallback to original method
            full_text, metadata = PDFProcessor.extract_text(pdf_path)
        
        # Enhanced text processing
        full_text = EnhancedPDFProcessor._post_process_text(full_text)
        metadata.update(EnhancedPDFProcessor._analyze_text_enhanced(full_text))
        
        return full_text, metadata
    
    @staticmethod
    def _clean_text_enhanced(text: str) -> str:
        """Enhanced text cleaning for physics papers"""
        if not text:
            return ""
        
        # Fix common OCR issues in physics papers
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = text.replace('α', 'alpha').replace('β', 'beta').replace('γ', 'gamma')
        text = text.replace('Δ', 'Delta').replace('∇', 'nabla').replace('∂', 'partial')
        
        # Fix spacing around mathematical symbols
        text = re.sub(r'([=+\-*/])', r' \1 ', text)
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        
        # Remove page headers/footers (common patterns)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            # Skip obvious headers/footers
            if (len(line) < 3 or 
                re.match(r'^\d+$', line) or  # Page numbers - FIXED: added $ and closing quote
                re.match(r'^Page \d+', line) or
                ('arXiv:' in line and len(line) < 50) or  # FIXED: added parentheses for logical grouping
                ('viXra:' in line and len(line) < 50)):   # FIXED: added parentheses for logical grouping
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    @staticmethod
    def _post_process_text(text: str) -> str:
        """Post-process extracted text for better structure"""
        if not text:
            return ""
        
        # Fix sentence boundaries
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Preserve equation formatting
        text = re.sub(r'\$([^$]+)\$', r' $\1$ ', text)  # FIXED: added missing $ and closing quote
        
        # Fix paragraph breaks
        text = re.sub(r'\n\s*\n', '\n\n', text)
        
        # Remove excessive whitespace but preserve structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\n\n+', '\n\n', text)
        
        return text.strip()
    
    @staticmethod
    def _analyze_text_enhanced(text: str) -> Dict[str, Any]:
        """Enhanced text analysis for physics content"""
        words = text.split()
        
        # Physics-specific analysis
        physics_terms = [
            'energy', 'force', 'field', 'particle', 'wave', 'quantum', 
            'relativity', 'momentum', 'mass', 'velocity', 'acceleration',
            'electromagnetic', 'gravitational', 'thermodynamic', 'statistical'
        ]
        
        math_terms = [
            'equation', 'formula', 'derivative', 'integral', 'differential',
            'matrix', 'vector', 'tensor', 'calculation', 'proof', 'theorem'
        ]
        
        # Count occurrences
        physics_count = sum(1 for word in words if word.lower() in physics_terms)
        math_count = sum(1 for word in words if word.lower() in math_terms)
        
        # Enhanced equation detection
        equation_patterns = [
            r'\$[^$]+\$',    # LaTeX inline - FIXED: added missing $
            r'\$\$[^$]+\$\$',  # LaTeX display - FIXED: added missing $
            r'[A-Za-z]\s*=\s*[^,.\n]{3,}',  # Simple equations
            r'\\frac\{[^}]+\}\{[^}]+\}',  # Fractions
            r'\\int\s*.*?dx',  # Integrals
            r'\\sum\s*.*?',  # Summations
        ]
        
        equation_count = sum(len(re.findall(pattern, text)) for pattern in equation_patterns)
        
        # Reference patterns (more comprehensive)
        ref_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d{4}\)',  # (2023), (1999), etc.
            r'et al\.',  # Author citations
            r'Ref\.\s*\d+',  # Ref. 1, Ref. 2, etc.
        ]
        
        ref_count = sum(len(re.findall(pattern, text)) for pattern in ref_patterns)
        
        # Calculate physics content density
        total_words = len(words)
        physics_density = (physics_count + math_count) / max(total_words, 1) * 100
        
        return {
            'word_count': total_words,
            'char_count': len(text),
            'physics_terms': physics_count,
            'math_terms': math_count,
            'physics_density': physics_density,
            'equation_count': equation_count,
            'reference_count': ref_count,
            'avg_sentence_length': total_words / max(text.count('.'), 1),
            'has_substantial_physics': physics_density > 2.0,
            'has_mathematical_content': equation_count > 5
        }
