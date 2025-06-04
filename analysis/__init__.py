"""
Enhanced Analysis package for physics papers
Now includes self-contained benchmark creation and high-quality RL training data
"""

from analysis.pdf_processor import PDFProcessor, EnhancedPDFProcessor
from analysis.classifier import SubtlePhysicsClassifier
from analysis.enhanced_benchmark_builder import SelfContainedBenchmarkBuilder
from analysis.enhanced_training_builder import ChainOfThoughtTrainingBuilder

__all__ = [
    'PDFProcessor', 
    'EnhancedPDFProcessor',
    'SubtlePhysicsClassifier',
    'SelfContainedBenchmarkBuilder',
    'ChainOfThoughtTrainingBuilder'
]