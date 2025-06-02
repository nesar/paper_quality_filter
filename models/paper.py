"""
Data models for representing scientific papers and quality assessments
"""
from dataclasses import dataclass
from typing import Dict, List, Optional, Any

@dataclass
class Paper:
    """Represents a scientific paper from arXiv or viXra"""
    id: str
    title: str
    authors: List[str]
    subject: str
    abstract: str
    submission_date: str
    pdf_url: str
    pdf_path: Optional[str] = None
    full_text: Optional[str] = None
    metadata: Dict[str, Any] = None

@dataclass
class QualityAssessment:
    """Paper quality assessment results focused on subtle physics issues"""
    paper_id: str
    overall_score: float
    stage_1_pass: bool
    stage_2_scores: Dict[str, float]
    stage_3_recommendation: str
    subtle_issues: List[str]
    physics_sophistication: float
    reasoning: str
    processing_timestamp: str