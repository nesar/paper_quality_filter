"""
PDF text extraction and processing utilities
"""
import logging
import re
from typing import Dict, Any, Tuple

import PyPDF2
import pdfplumber

logger = logging.getLogger(__name__)


from nougat import NougatModel
from nougat.utils.checkpoint import get_checkpoint
# from nougat.utils.device import get_device
from PIL import Image
import torch
NOUGAT_AVAILABLE = True


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




class NougatPDFProcessor:
    """Enhanced PDF processor using Nougat for academic papers"""
    
    def __init__(self, model_name: str = "0.1.0-small", device: Optional[str] = None):
        """
        Initialize Nougat processor
        
        Args:
            model_name: Nougat model version ("0.1.0-small" or "0.1.0-base")
            device: Device to use ('cuda', 'cpu', or None for auto-detection)
        """
        self.model = None
        # self.device = device or get_device() if NOUGAT_AVAILABLE else "cpu"
        self.model_name = model_name
        
        if NOUGAT_AVAILABLE:
            self._initialize_model()
        else:
            logger.warning("Nougat not available. Install with: pip install nougat-ocr")
    
    def _initialize_model(self):
        """Initialize the Nougat model"""
        try:
            logger.info(f"Loading Nougat model {self.model_name} on {self.device}")
            
            # Get model checkpoint
            checkpoint = get_checkpoint(self.model_name)
            
            # Initialize model
            self.model = NougatModel.from_pretrained(checkpoint)
            self.model.to(self.device)
            self.model.eval()
            
            logger.info("Nougat model loaded successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize Nougat model: {e}")
            self.model = None
    
    def extract_text(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """
        Extract text and metadata from PDF using Nougat
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            Tuple of (extracted_text, metadata)
        """
        if not NOUGAT_AVAILABLE or self.model is None:
            logger.warning("Nougat not available, falling back to traditional methods")
            return self._fallback_extract_text(pdf_path)
        
        try:
            # Convert PDF to markdown using Nougat
            markdown_text = self._extract_with_nougat(pdf_path)
            
            if not markdown_text or len(markdown_text.strip()) < 100:
                logger.warning("Nougat extraction failed or produced minimal content, using fallback")
                return self._fallback_extract_text(pdf_path)
            
            # Process the markdown text
            processed_text = self._process_nougat_output(markdown_text)
            
            # Analyze the extracted text
            metadata = self._analyze_text_enhanced(processed_text)
            metadata['extraction_method'] = 'nougat'
            metadata['model_name'] = self.model_name
            
            logger.info(f"Successfully extracted {len(processed_text)} characters using Nougat")
            
            return processed_text, metadata
            
        except Exception as e:
            logger.error(f"Nougat extraction failed: {e}")
            return self._fallback_extract_text(pdf_path)
    
    def _extract_with_nougat(self, pdf_path: str) -> str:
        """Extract text using Nougat model"""
        from nougat.utils.dataset import LazyDataset
        from torch.utils.data import DataLoader
        
        # Create dataset for the single PDF
        dataset = LazyDataset(
            pdf_path,
            partial=False,  # Process entire document
            equations=True   # Extract equations
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=LazyDataset.ignore_none_collate
        )
        
        predictions = []
        
        with torch.no_grad():
            for batch in dataloader:
                if batch is None:
                    continue
                
                # Move batch to device
                for key in batch:
                    if isinstance(batch[key], torch.Tensor):
                        batch[key] = batch[key].to(self.device)
                
                # Generate prediction
                output = self.model.inference(
                    image=batch["image"],
                    early_stopping=False
                )
                
                predictions.extend(output)
        
        # Combine all predictions
        full_text = "\n\n".join(predictions)
        return full_text
    
    def _process_nougat_output(self, markdown_text: str) -> str:
        """Process and clean Nougat markdown output"""
        if not markdown_text:
            return ""
        
        # Clean up the markdown text
        text = self._clean_nougat_markdown(markdown_text)
        
        # Convert LaTeX equations to more readable format
        text = self._process_latex_equations(text)
        
        # Fix common formatting issues
        text = self._fix_formatting_issues(text)
        
        return text.strip()
    
    def _clean_nougat_markdown(self, text: str) -> str:
        """Clean Nougat markdown output"""
        # Remove markdown formatting that's not needed for analysis
        text = re.sub(r'^#+\s*', '', text, flags=re.MULTILINE)  # Remove headers
        text = re.sub(r'\*\*(.*?)\*\*', r'\1', text)  # Remove bold
        text = re.sub(r'\*(.*?)\*', r'\1', text)  # Remove italic
        text = re.sub(r'`(.*?)`', r'\1', text)  # Remove code formatting
        
        # Clean up table formatting
        text = re.sub(r'\|[^\n]*\|', '', text)  # Remove table rows
        text = re.sub(r'[\-\|:]+\n', '', text)  # Remove table separators
        
        # Fix line breaks and spacing
        text = re.sub(r'\n\s*\n\s*\n+', '\n\n', text)  # Multiple newlines to double
        text = re.sub(r'[ \t]+', ' ', text)  # Multiple spaces to single
        
        return text
    
    def _process_latex_equations(self, text: str) -> str:
        """Process LaTeX equations in the text"""
        # Nougat should preserve LaTeX equations well, but we can clean them up
        
        # Ensure proper spacing around equations
        text = re.sub(r'\$([^$]+)\$', r' $\1$ ', text)  # Inline equations
        text = re.sub(r'\$\$([^$]+)\$\$', r'\n$$\1$$\n', text)  # Display equations
        
        # Clean up equation environments
        text = re.sub(r'\\begin\{equation\}(.*?)\\end\{equation\}', 
                     r'\n\\begin{equation}\1\\end{equation}\n', text, flags=re.DOTALL)
        
        text = re.sub(r'\\begin\{align\}(.*?)\\end\{align\}', 
                     r'\n\\begin{align}\1\\end{align}\n', text, flags=re.DOTALL)
        
        return text
    
    def _fix_formatting_issues(self, text: str) -> str:
        """Fix common formatting issues in extracted text"""
        # Fix sentence boundaries
        text = re.sub(r'\.([A-Z])', r'. \1', text)
        text = re.sub(r'([a-z])([A-Z])', r'\1 \2', text)
        
        # Fix common OCR issues that might still occur
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        text = text.replace('α', 'alpha').replace('β', 'beta').replace('γ', 'gamma')
        text = text.replace('Δ', 'Delta').replace('∇', 'nabla').replace('∂', 'partial')
        
        # Remove excessive whitespace while preserving structure
        text = re.sub(r'[ \t]+', ' ', text)
        text = re.sub(r'\n\n\n+', '\n\n', text)
        
        return text
    
    def _fallback_extract_text(self, pdf_path: str) -> Tuple[str, Dict[str, Any]]:
        """Fallback to traditional PDF extraction methods"""
        logger.info("Using fallback PDF extraction methods")
        
        full_text = ""
        metadata = {}
        
        # Try pdfplumber first
        try:
            with pdfplumber.open(pdf_path) as pdf:
                pages = []
                for page in pdf.pages:
                    text = page.extract_text(
                        x_tolerance=3,
                        y_tolerance=3,
                        layout=True
                    )
                    if text:
                        pages.append(text)
                
                full_text = "\n\n".join(pages)
                metadata['num_pages'] = len(pdf.pages)
                metadata['extraction_method'] = 'pdfplumber_fallback'
                
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
                    metadata['extraction_method'] = 'pypdf2_fallback'
                    
            except Exception as e2:
                logger.error(f"All PDF extraction methods failed: {e2}")
                metadata['extraction_method'] = 'failed'
        
        # Clean and analyze the text
        if full_text:
            full_text = self._clean_fallback_text(full_text)
            metadata.update(self._analyze_text_enhanced(full_text))
        
        return full_text, metadata
    
    def _clean_fallback_text(self, text: str) -> str:
        """Clean text from fallback extraction methods"""
        # Basic cleaning similar to original processor
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\x00-\x7F]+', '', text)
        text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl')
        
        # Remove obvious headers/footers
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if (len(line) < 3 or 
                re.match(r'^\d+$', line) or
                re.match(r'^Page \d+', line) or
                ('arXiv:' in line and len(line) < 50) or
                ('viXra:' in line and len(line) < 50)):
                continue
            cleaned_lines.append(line)
        
        return '\n'.join(cleaned_lines)
    
    def _analyze_text_enhanced(self, text: str) -> Dict[str, Any]:
        """Enhanced text analysis for physics content"""
        if not text:
            return {}
        
        words = text.split()
        
        # Physics-specific analysis
        physics_terms = [
            'energy', 'force', 'field', 'particle', 'wave', 'quantum', 
            'relativity', 'momentum', 'mass', 'velocity', 'acceleration',
            'electromagnetic', 'gravitational', 'thermodynamic', 'statistical',
            'cosmology', 'astrophysics', 'plasma', 'nuclear', 'atomic'
        ]
        
        math_terms = [
            'equation', 'formula', 'derivative', 'integral', 'differential',
            'matrix', 'vector', 'tensor', 'calculation', 'proof', 'theorem',
            'lagrangian', 'hamiltonian', 'eigenvalue', 'transform'
        ]
        
        # Count occurrences
        physics_count = sum(1 for word in words if word.lower() in physics_terms)
        math_count = sum(1 for word in words if word.lower() in math_terms)
        
        # Enhanced equation detection (better for Nougat output)
        equation_patterns = [
            r'\$[^$]+\$',    # LaTeX inline
            r'\$\$[^$]+\$\$',  # LaTeX display
            r'\\begin\{equation\}.*?\\end\{equation\}',  # Equation environments
            r'\\begin\{align\}.*?\\end\{align\}',  # Align environments
            r'[A-Za-z]\s*=\s*[^,.\n]{3,}',  # Simple equations
            r'\\frac\{[^}]+\}\{[^}]+\}',  # Fractions
            r'\\int\s*.*?d[a-z]',  # Integrals
            r'\\sum\s*.*?',  # Summations
            r'\\partial\s*.*?\\partial',  # Partial derivatives
        ]
        
        equation_count = sum(len(re.findall(pattern, text, re.DOTALL)) 
                           for pattern in equation_patterns)
        
        # Reference patterns
        ref_patterns = [
            r'\[\d+\]',  # [1], [2], etc.
            r'\(\d{4}\)',  # (2023), (1999), etc.
            r'et al\.',  # Author citations
            r'Ref\.\s*\d+',  # Ref. 1, Ref. 2, etc.
            r'\\cite\{[^}]+\}',  # LaTeX citations
        ]
        
        ref_count = sum(len(re.findall(pattern, text)) for pattern in ref_patterns)
        
        # Calculate physics content density
        total_words = len(words)
        physics_density = (physics_count + math_count) / max(total_words, 1) * 100
        
        # Detect mathematical sophistication
        sophisticated_math_terms = [
            'lagrangian', 'hamiltonian', 'eigenvalue', 'eigenvector', 'tensor',
            'manifold', 'topology', 'differential geometry', 'lie group',
            'renormalization', 'gauge theory', 'symmetry breaking'
        ]
        
        sophisticated_count = sum(1 for word in words 
                                if any(term in word.lower() for term in sophisticated_math_terms))
        
        return {
            'word_count': total_words,
            'char_count': len(text),
            'physics_terms': physics_count,
            'math_terms': math_count,
            'physics_density': physics_density,
            'equation_count': equation_count,
            'reference_count': ref_count,
            'sophisticated_math_count': sophisticated_count,
            'avg_sentence_length': total_words / max(text.count('.'), 1),
            'has_substantial_physics': physics_density > 2.0,
            'has_mathematical_content': equation_count > 5,
            'has_sophisticated_math': sophisticated_count > 2,
            'latex_quality_score': self._assess_latex_quality(text)
        }
    
    def _assess_latex_quality(self, text: str) -> float:
        """Assess the quality of LaTeX preservation in the text"""
        latex_indicators = [
            r'\\frac\{[^}]+\}\{[^}]+\}',
            r'\\int\s*.*?d[a-z]',
            r'\\sum\s*.*?',
            r'\\partial',
            r'\\nabla',
            r'\\alpha', r'\\beta', r'\\gamma',
            r'\\begin\{.*?\}',
            r'\$.*?\$'
        ]
        
        total_indicators = sum(len(re.findall(pattern, text, re.DOTALL)) 
                             for pattern in latex_indicators)
        
        # Normalize by text length (per 1000 characters)
        if len(text) > 0:
            return min(1.0, total_indicators / (len(text) / 1000) / 10)
        return 0.0


# Convenience function for backward compatibility
def extract_text_with_nougat(pdf_path: str, 
                           model_name: str = "0.1.0-small",
                           device: Optional[str] = None) -> Tuple[str, Dict[str, Any]]:
    """
    Convenience function to extract text using Nougat
    
    Args:
        pdf_path: Path to PDF file
        model_name: Nougat model version
        device: Device to use
        
    Returns:
        Tuple of (extracted_text, metadata)
    """
    processor = NougatPDFProcessor(model_name=model_name, device=device)
    return processor.extract_text(pdf_path)
