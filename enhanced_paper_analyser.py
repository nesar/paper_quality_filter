#!/usr/bin/env python3
"""
Enhanced Paper Quality Filter - Main Entry Point
A system for identifying papers with subtle physics issues and building reasoning benchmarks
"""

import asyncio
import argparse
import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Local imports
from scrapers.arxiv_scraper import ArXivScraper
from scrapers.vixra_scraper import ViXraScraper
from analysis.pdf_processor import PDFProcessor
from analysis.classifier import SubtlePhysicsClassifier
from utils.categories import ARXIV_CATEGORIES, VIXRA_CATEGORIES
from analysis.pdf_processor import EnhancedPDFProcessor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BenchmarkBuilder:
    """Builds reasoning benchmarks from analyzed papers"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.benchmark_dir = self.output_dir / "benchmark"
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
    def create_reasoning_benchmark(self, paper, assessment, full_text: str) -> Dict[str, Any]:
        """Create a reasoning benchmark from a paper with identified issues"""
        
        # Check if paper has substantial content and is in English
        if not self._is_suitable_for_benchmark(paper, full_text):
            logger.info(f"Paper {paper.id} not suitable for benchmark creation")
            return None
        
        # Clean and extract content
        clean_title = self._clean_title(paper.title)
        clean_abstract = self._clean_abstract(paper.abstract)
        
        # Extract mathematical derivations and equations
        equations = self._extract_equations(full_text)
        derivations = self._extract_derivations(full_text)
        assumptions = self._extract_assumptions(full_text)
        
        # Create benchmark questions based on the assessment
        questions = []
        
        # Question 1: General analysis (more generic)
        questions.append({
            "type": "general_analysis",
            "question": f"Analyze this physics paper and identify any issues in scientific reasoning, mathematical derivations, and underlying assumptions. Focus on subtle errors that require physics expertise to detect.\n\nAbstract: {clean_abstract}\n\nWhat specific technical problems can you identify in the approach, methodology, or conclusions?",
            "context": {
                "paper_id": paper.id,
                "subject": paper.subject,
                "authors": paper.authors,
                "title": clean_title
            },
            "ground_truth": {
                "issues_found": assessment.subtle_issues,
                "technical_scores": assessment.stage_2_scores,
                "sophistication": assessment.physics_sophistication,
                "recommendation": assessment.stage_3_recommendation,
                "reasoning": assessment.reasoning
            }
        })
        
        # Question 2: Mathematical consistency (only if substantial equations found)
        if equations and len(equations.strip()) > 100:
            questions.append({
                "type": "mathematical_analysis",
                "question": f"Examine the mathematical framework presented in this physics work. Identify any inconsistencies, dimensional problems, or derivation errors in the equations and calculations.\n\nKey equations: {equations[:1000]}...",
                "context": {
                    "paper_id": paper.id,
                    "subject": paper.subject
                },
                "ground_truth": {
                    "mathematical_issues": [issue for issue in assessment.subtle_issues if any(keyword in issue.lower() for keyword in ['equation', 'math', 'dimensional', 'calculation', 'derivation'])],
                    "math_score": assessment.stage_2_scores.get('mathematical_errors', 5)
                }
            })
        
        # Question 3: Physics assumptions (only if substantial assumptions found)
        if assumptions and len(assumptions.strip()) > 50:
            questions.append({
                "type": "assumption_analysis", 
                "question": f"Evaluate the physics assumptions and approximations made in this work. Are they appropriate for the context? Are there any overlooked effects or inappropriate simplifications?",
                "context": {
                    "paper_id": paper.id,
                    "subject": paper.subject
                },
                "ground_truth": {
                    "assumption_issues": [issue for issue in assessment.subtle_issues if any(keyword in issue.lower() for keyword in ['assumption', 'approximation', 'neglect', 'simplification'])],
                    "physics_score": assessment.stage_2_scores.get('physics_assumptions', 5)
                }
            })
        
        # Question 4: Logical reasoning (only if substantial derivations found)
        if derivations and len(derivations.strip()) > 100:
            questions.append({
                "type": "reasoning_chain",
                "question": f"Analyze the logical progression of arguments in this physics work. Identify any logical gaps, non-sequiturs, or places where the reasoning breaks down.",
                "context": {
                    "paper_id": paper.id,
                    "subject": paper.subject
                },
                "ground_truth": {
                    "logical_issues": [issue for issue in assessment.subtle_issues if any(keyword in issue.lower() for keyword in ['logic', 'reasoning', 'conclusion', 'follows', 'contradiction'])],
                    "consistency_score": assessment.stage_2_scores.get('logical_consistency', 5)
                }
            })
        
        benchmark_item = {
            "paper_metadata": {
                "id": paper.id,
                "title": clean_title,
                "authors": paper.authors,
                "subject": paper.subject,
                "submission_date": paper.submission_date,
                "source": "vixra" if "vixra" in paper.pdf_url else "arxiv"
            },
            "assessment_summary": {
                "overall_score": assessment.overall_score,
                "sophistication": assessment.physics_sophistication,
                "recommendation": assessment.stage_3_recommendation,
                "issues_count": len(assessment.subtle_issues)
            },
            "questions": questions,
            "created_at": datetime.now().isoformat()
        }
        
        return benchmark_item
    
    def _is_suitable_for_benchmark(self, paper, full_text: str) -> bool:
        """Check if paper is suitable for benchmark creation"""
        # More lenient minimum content length for viXra
        if not full_text or len(full_text.strip()) < 300:
            return False
        
        # Check if primarily English (more lenient for viXra)
        english_words = ['the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'for', 'with']
        text_sample = full_text[:1000].lower()
        english_count = sum(1 for word in english_words if word in text_sample)
        
        # Detect if this is from viXra
        is_vixra = "vixra" in paper.pdf_url.lower() if hasattr(paper, 'pdf_url') else False
        english_threshold = 2 if is_vixra else 3
        
        if english_count < english_threshold:
            return False
        
        # Check for substantial physics content (more lenient for viXra)
        physics_indicators = ['equation', 'theory', 'model', 'physics', 'energy', 'force', 'field', 'quantum', 'relativity']
        # Additional viXra-specific terms
        vixra_indicators = ['gravity', 'gravitational', 'universe', 'cosmic', 'space', 'time', 'mass', 'particle']
        
        physics_count = sum(1 for indicator in physics_indicators if indicator in text_sample)
        vixra_count = sum(1 for indicator in vixra_indicators if indicator in text_sample)
        
        combined_threshold = 1 if is_vixra else 2
        return (physics_count + vixra_count) >= combined_threshold
    
    def _clean_title(self, title: str) -> str:
        """Clean paper title, removing viXra admin notes and page counts"""
        if not title:
            return "Title not available"
        
        # Remove viXra admin notes
        title = re.sub(r'\(Note by viXra Admin:.*?\)', '', title, flags=re.IGNORECASE)
        
        # Remove page counts
        title = re.sub(r'^\d+\s+Pages?\.\s*', '', title, flags=re.IGNORECASE)
        
        # Clean up extra whitespace
        title = re.sub(r'\s+', ' ', title).strip()
        
        # If title is empty or too short after cleaning, provide default
        if not title or len(title) < 10:
            return "Physics paper (title not clearly specified)"
        
        return title
    
    def _clean_abstract(self, abstract: str) -> str:
        """Clean and truncate abstract"""
        if not abstract:
            return "Abstract not available"
        
        # Limit abstract length for readability
        if len(abstract) > 1000:
            abstract = abstract[:1000] + "..."
        
        return abstract.strip()
    
    def _extract_equations(self, text: str) -> str:
        """Extract mathematical equations from text"""
        # Look for LaTeX equations
        latex_patterns = [
            r'\$\$.*?\$\$',  # Display math
            r'\$.*?\$',      # Inline math
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'\\begin\{align\}.*?\\end\{align\}',
            r'\\begin\{eqnarray\}.*?\\end\{eqnarray\}'
        ]
        
        equations = []
        for pattern in latex_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            equations.extend(matches)
        
        # Also look for numbered equations and common physics expressions
        equation_indicators = re.findall(r'[Ee]quation?\s*\(\d+\)[^.]*[.!?]', text)
        equations.extend(equation_indicators)
        
        return "\n".join(equations[:10])  # Limit to prevent overflow
    
    def _extract_derivations(self, text: str) -> str:
        """Extract derivation sections from text"""
        # Look for derivation-like sections
        derivation_patterns = [
            r'[Dd]erivation[^.]*?(?:\n\n|\Z)',
            r'[Pp]roof[^.]*?(?:\n\n|\Z)',
            r'[Cc]alculation[^.]*?(?:\n\n|\Z)',
            r'Starting with.*?we obtain',
            r'Beginning with.*?we derive',
            r'From.*?it follows that'
        ]
        
        derivations = []
        for pattern in derivation_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            derivations.extend(matches)
        
        return "\n".join(derivations[:5])  # Limit to prevent overflow
    
    def _extract_assumptions(self, text: str) -> str:
        """Extract assumption statements from text"""
        assumption_patterns = [
            r'[Aa]ssum[ei][^.]*?[.!?]',
            r'[Ww]e consider[^.]*?[.!?]',
            r'[Ww]e neglect[^.]*?[.!?]',
            r'[Ff]or simplicity[^.]*?[.!?]',
            r'[Aa]pproximat[^.]*?[.!?]',
            r'[Ii]n the limit[^.]*?[.!?]'
        ]
        
        assumptions = []
        for pattern in assumption_patterns:
            matches = re.findall(pattern, text)
            assumptions.extend(matches)
        
        return "\n".join(assumptions[:10])  # Limit to prevent overflow
    
    def save_benchmark(self, benchmark_items: List[Dict]) -> str:
        """Save benchmark to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"physics_reasoning_benchmark_{timestamp}.json"
        filepath = self.benchmark_dir / filename
        
        benchmark_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_papers": len(benchmark_items),
                "total_questions": sum(len(item["questions"]) for item in benchmark_items),
                "version": "1.0"
            },
            "benchmark_items": benchmark_items
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Benchmark saved to {filepath}")
        return str(filepath)

class EnhancedBenchmarkBuilder(BenchmarkBuilder):
    """Enhanced benchmark builder following UGPhysics standards"""
    
    def create_reasoning_benchmark(self, paper, assessment, full_text: str) -> Dict[str, Any]:
        """Create UGPhysics-style reasoning benchmark"""
        
        if not self._is_suitable_for_benchmark_enhanced(paper, full_text):
            return None
        
        clean_title = self._clean_title(paper.title)
        clean_abstract = self._clean_abstract(paper.abstract)
        
        # Extract physics content with better parsing
        physics_content = self._extract_physics_content_structured(full_text)
        
        if not physics_content:
            return None
        
        # Create structured questions following UGPhysics methodology
        questions = self._create_ugphysics_style_questions(
            paper, assessment, physics_content, clean_abstract
        )
        
        if len(questions) < 2:  # Need at least 2 quality questions
            return None
        
        benchmark_item = {
            "paper_metadata": {
                "id": paper.id,
                "title": clean_title,
                "authors": paper.authors,
                "subject": paper.subject,
                "submission_date": paper.submission_date,
                "source": "vixra" if "vixra" in paper.pdf_url else "arxiv"
            },
            "assessment_summary": {
                "overall_score": assessment.overall_score,
                "sophistication": assessment.physics_sophistication,
                "recommendation": assessment.stage_3_recommendation,
                "issues_count": len(assessment.subtle_issues)
            },
            "questions": questions,
            "created_at": datetime.now().isoformat()
        }
        
        return benchmark_item
    
    def _extract_physics_content_structured(self, text: str) -> Dict[str, Any]:
        """Extract structured physics content for benchmark creation"""
        
        content = {
            "equations": [],
            "derivations": [],
            "problem_solutions": [],
            "physics_principles": [],
            "mathematical_steps": []
        }
        
        # Extract equations (LaTeX and plain text)
        equation_patterns = [
            r'\$\$.*?\$\$',  # Display math
            r'\$.*?\$',      # Inline math  
            r'\\begin\{equation\}.*?\\end\{equation\}',
            r'[A-Za-z]\s*=\s*[^,.\n]{3,50}',  # Simple equations like E = mc^2
            r'[∇∂].*?=.*?[^,.\n]{3,50}'       # Differential equations
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, text, re.DOTALL)
            content["equations"].extend([m.strip() for m in matches if len(m.strip()) > 5])
        
        # Extract derivations
        derivation_patterns = [
            r'(?:Derivation|Proof|To show|To derive).*?(?:Q\.E\.D\.|Therefore|Thus)[^.]*\.',
            r'(?:Starting with|From|Given).*?(?:equation|relation).*?(?:we get|we obtain)[^.]*\.'
        ]
        
        for pattern in derivation_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            content["derivations"].extend([m.strip() for m in matches if len(m.strip()) > 50])
        
        # Extract problem-solution pairs
        problem_patterns = [
            r'(?:Problem|Example|Exercise)\s*:?.*?(?:Solution|Answer).*?(?:\n\n|\Z)',
            r'(?:Find|Calculate|Determine|Show).*?(?:Given|where).*?(?:Solution|Answer|Therefore).*?'
        ]
        
        for pattern in problem_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            content["problem_solutions"].extend([m.strip() for m in matches if len(m.strip()) > 100])
        
        return content if any(content.values()) else None
    
    def _create_ugphysics_style_questions(self, paper, assessment, physics_content, abstract) -> List[Dict]:
        """Create questions following UGPhysics format and standards"""
        
        questions = []
        
        # Question Type 1: Mathematical Derivation (like UGPhysics examples)
        if physics_content["equations"] or physics_content["derivations"]:
            derivation_question = self._create_derivation_question(
                paper, physics_content, abstract
            )
            if derivation_question:
                questions.append(derivation_question)
        
        # Question Type 2: Physics Reasoning (identifying flawed reasoning)
        if assessment.subtle_issues:
            reasoning_question = self._create_reasoning_analysis_question(
                paper, assessment, abstract
            )
            if reasoning_question:
                questions.append(reasoning_question)
        
        # Question Type 3: Problem Solving (if complete solutions exist)
        if physics_content["problem_solutions"]:
            problem_question = self._create_problem_solving_question(
                paper, physics_content, abstract
            )
            if problem_question:
                questions.append(problem_question)
        
        # Question Type 4: Physics Principles Application
        physics_principles_question = self._create_principles_question(
            paper, abstract, assessment
        )
        if physics_principles_question:
            questions.append(physics_principles_question)
        
        return questions
    
    def _create_derivation_question(self, paper, physics_content, abstract) -> Optional[Dict]:
        """Create mathematical derivation question like UGPhysics examples"""
        
        # Select best equation or derivation
        best_content = None
        if physics_content["derivations"]:
            best_content = max(physics_content["derivations"], key=len)
        elif physics_content["equations"]:
            equations = [eq for eq in physics_content["equations"] if len(eq) > 20]
            if equations:
                best_content = equations[0]
        
        if not best_content or len(best_content) < 30:
            return None
        
        question = {
            "type": "mathematical_derivation",
            "question": f"""Given the physics context from this work on {paper.subject.lower()}:

Abstract: {abstract[:500]}...

Analyze the following mathematical derivation and identify any errors in the mathematical reasoning, dimensional analysis, or application of physics principles:

{best_content[:800]}

Provide a detailed analysis of:
1. Mathematical consistency of the derivation
2. Proper application of physics principles
3. Dimensional correctness of equations
4. Any logical gaps or unjustified steps""",
            
            "context": {
                "paper_id": paper.id,
                "subject": paper.subject,
                "derivation_type": "mathematical_derivation"
            },
            
            "ground_truth": {
                "analysis_type": "derivation_analysis",
                "expected_issues": [
                    "Mathematical inconsistencies may be present",
                    "Physics principles may be misapplied", 
                    "Dimensional analysis may reveal errors",
                    "Logical gaps may exist in reasoning"
                ],
                "difficulty": "intermediate",
                "reasoning_skills": ["mathematical_reasoning", "physics_application", "dimensional_analysis"]
            }
        }
        
        return question
    
    def _create_reasoning_analysis_question(self, paper, assessment, abstract) -> Optional[Dict]:
        """Create reasoning analysis question based on identified issues"""
        
        if not assessment.subtle_issues:
            return None
        
        # Focus on the most significant issues
        key_issues = assessment.subtle_issues[:3]
        
        question = {
            "type": "reasoning_analysis", 
            "question": f"""Analyze the physics reasoning in this work on {paper.subject.lower()}:

Abstract: {abstract[:500]}...

This work has been identified as having potential reasoning issues. Examine the approach and methodology for:

1. **Logical consistency**: Are the arguments internally consistent?
2. **Physics assumptions**: Are the underlying physics assumptions valid?
3. **Mathematical rigor**: Is the mathematical treatment appropriate?
4. **Literature context**: How does this relate to established physics?

Provide a critical analysis focusing on potential flaws in the reasoning process.""",
            
            "context": {
                "paper_id": paper.id,
                "subject": paper.subject,
                "analysis_type": "reasoning_critique"
            },
            
            "ground_truth": {
                "identified_issues": key_issues,
                "sophistication_level": assessment.physics_sophistication,
                "recommendation": assessment.stage_3_recommendation,
                "reasoning": assessment.reasoning[:500],
                "analysis_focus": ["logical_consistency", "physics_assumptions", "mathematical_rigor"]
            }
        }
        
        return question
    
    def _create_problem_solving_question(self, paper, physics_content, abstract) -> Optional[Dict]:
        """Create problem-solving question from extracted solutions"""
        
        if not physics_content["problem_solutions"]:
            return None
        
        best_solution = max(physics_content["problem_solutions"], key=len)
        
        # Extract problem and solution parts
        problem_part, solution_part = self._separate_problem_solution_advanced(best_solution)
        
        if len(problem_part) < 30 or len(solution_part) < 50:
            return None
        
        question = {
            "type": "problem_solving",
            "question": f"""Consider this physics problem from the domain of {paper.subject.lower()}:

**Problem**: {problem_part}

**Proposed Solution**: {solution_part[:600]}...

Evaluate this solution approach:
1. Is the problem setup correct?
2. Are the solution methods appropriate?
3. Are there any errors in the mathematical steps?
4. Is the final result reasonable?

Provide a detailed critique of the solution methodology.""",
            
            "context": {
                "paper_id": paper.id,
                "subject": paper.subject,
                "problem_type": "applied_physics"
            },
            
            "ground_truth": {
                "evaluation_criteria": [
                    "Problem setup correctness",
                    "Solution method appropriateness", 
                    "Mathematical accuracy",
                    "Result reasonableness"
                ],
                "solution_analysis": "Complete solution evaluation required",
                "difficulty": "intermediate"
            }
        }
        
        return question
    
    def _create_principles_question(self, paper, abstract, assessment) -> Optional[Dict]:
        """Create physics principles application question"""
        
        # Determine relevant physics principles based on subject
        principles_map = {
            "Quantum Physics": ["wave-particle duality", "uncertainty principle", "quantum superposition"],
            "General Relativity": ["equivalence principle", "spacetime curvature", "geodesic motion"],
            "Thermodynamics": ["conservation of energy", "entropy increase", "thermal equilibrium"],
            "Electromagnetism": ["Maxwell's equations", "charge conservation", "electromagnetic induction"],
            "High Energy Physics": ["conservation laws", "symmetry principles", "gauge invariance"]
        }
        
        relevant_principles = []
        for domain, principles in principles_map.items():
            if domain.lower() in paper.subject.lower():
                relevant_principles = principles
                break
        
        if not relevant_principles:
            relevant_principles = ["conservation of energy", "dimensional consistency", "physical reasonableness"]
        
        question = {
            "type": "principles_application",
            "question": f"""Examine this work in {paper.subject.lower()}:

Abstract: {abstract[:400]}...

Analyze how well this work applies fundamental physics principles. Consider:

1. **Conservation Laws**: Are relevant conservation laws properly applied?
2. **Symmetry Principles**: Are symmetries correctly identified and used?
3. **Dimensional Analysis**: Is dimensional consistency maintained?
4. **Physical Intuition**: Do the results align with physical expectations?

Focus particularly on the application of: {', '.join(relevant_principles[:3])}

Identify any violations or misapplications of these fundamental principles.""",
            
            "context": {
                "paper_id": paper.id,
                "subject": paper.subject,
                "principles_focus": relevant_principles[:3]
            },
            
            "ground_truth": {
                "principles_to_check": relevant_principles,
                "sophistication_required": assessment.physics_sophistication,
                "expected_analysis": "Systematic evaluation of physics principles application",
                "common_errors": [
                    "Conservation law violations",
                    "Dimensional inconsistencies", 
                    "Symmetry misapplications",
                    "Unphysical results"
                ]
            }
        }
        
        return question
    
    def _separate_problem_solution_advanced(self, text: str) -> Tuple[str, str]:
        """Advanced separation of problem and solution parts"""
        
        # Look for clear separators
        separators = [
            r'(?:Solution|Answer)\s*:',
            r'(?:Given|Find|Calculate|Determine).*?(?:Solution|Answer)',
            r'(?:Problem)\s*:.*?(?:Solution|Answer)\s*:'
        ]
        
        for separator in separators:
            match = re.search(separator, text, re.IGNORECASE | re.DOTALL)
            if match:
                split_point = match.end()
                problem = text[:split_point].strip()
                solution = text[split_point:].strip()
                return problem, solution
        
        # If no clear separator, split roughly in half
        mid_point = len(text) // 2
        return text[:mid_point].strip(), text[mid_point:].strip()
    
    def _is_suitable_for_benchmark_enhanced(self, paper, full_text: str) -> bool:
        """Enhanced suitability check for benchmark creation"""
        
        if not full_text or len(full_text.strip()) < 500:
            return False
        
        # Check for substantial physics content
        physics_indicators = [
            'equation', 'theory', 'model', 'energy', 'force', 'field', 
            'quantum', 'relativity', 'particle', 'wave', 'conservation'
        ]
        
        # Check for mathematical content
        math_indicators = [
            'calculate', 'derive', 'solve', 'proof', 'theorem', 'formula',
            'differential', 'integral', 'matrix', 'vector'
        ]
        
        text_lower = full_text[:2000].lower()
        physics_count = sum(1 for indicator in physics_indicators if indicator in text_lower)
        math_count = sum(1 for indicator in math_indicators if indicator in text_lower)
        
        # More lenient for viXra papers but still require some content
        is_vixra = "vixra" in paper.pdf_url.lower() if hasattr(paper, 'pdf_url') else False
        threshold = 2 if is_vixra else 3
        
        return (physics_count + math_count) >= threshold

class TrainingDataBuilder:
    """Builds training data for reinforcement learning with LLMs"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.training_dir = self.output_dir / "training_data"
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
    def extract_training_examples(self, paper, full_text: str) -> List[Dict[str, Any]]:
        """Extract step-by-step derivations and solutions for training"""
        
        # Check if paper is suitable for training data extraction
        if not self._is_suitable_for_training(paper, full_text):
            logger.info(f"Paper {paper.id} not suitable for training data extraction")
            return []
        
        training_examples = []
        
        # Extract worked examples and derivations with better filtering
        derivations = self._find_complete_derivations(full_text)
        solutions = self._find_worked_solutions(full_text)
        proofs = self._find_mathematical_proofs(full_text)
        
        # Process each type with improved logic
        for derivation in derivations[:3]:  # Limit to prevent overwhelming
            example = self._create_training_example(derivation, paper, "derivation")
            if example:
                training_examples.append(example)
        
        for solution in solutions[:3]:
            example = self._create_training_example(solution, paper, "solution")
            if example:
                training_examples.append(example)
                
        for proof in proofs[:2]:  # Fewer proofs as they tend to be longer
            example = self._create_training_example(proof, paper, "proof")
            if example:
                training_examples.append(example)
        
        return training_examples
    
    def _is_suitable_for_training(self, paper, full_text: str) -> bool:
        """Check if paper is suitable for training data extraction"""
        if not full_text or len(full_text.strip()) < 500:  # Reduced minimum length for viXra
            return False
        
        # More lenient English detection for viXra papers
        english_indicators = ['the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'for', 'with', 'equation', 'we', 'can', 'from']
        text_sample = full_text[:2000].lower()
        english_count = sum(1 for word in english_indicators if word in text_sample)
        
        # Detect if this is from viXra (more lenient requirements)
        is_vixra = "vixra" in paper.pdf_url.lower() if hasattr(paper, 'pdf_url') else False
        english_threshold = 5 if is_vixra else 8
        
        if english_count < english_threshold:
            return False
        
        # More flexible step-by-step content indicators for viXra
        step_indicators = ['step', 'first', 'second', 'next', 'then', 'therefore', 'thus', 'hence', 'derivation', 'proof', 'solution']
        # Additional physics reasoning indicators
        physics_indicators = ['calculate', 'derive', 'obtain', 'find', 'result', 'using', 'apply', 'given', 'assume', 'consider']
        
        step_count = sum(1 for indicator in step_indicators if indicator in text_sample)
        physics_count = sum(1 for indicator in physics_indicators if indicator in text_sample)
        
        # More lenient threshold for viXra
        combined_threshold = 2 if is_vixra else 3
        return (step_count + physics_count) >= combined_threshold
    
    def _find_complete_derivations(self, text: str) -> List[str]:
        """Find complete derivations with clear step-by-step structure"""
        # Enhanced patterns for viXra papers (often less formal)
        derivation_patterns = [
            r'(?:To derive|To show|To prove|We derive|We show|We calculate|We find).*?(?:Therefore|Thus|Hence|We obtain|This gives|Q\.E\.D\.)[^.]*\.',
            r'(?:Starting with|Beginning with|We start with|From|Given).*?(?:Therefore|Thus|Hence|We obtain|This gives|we get|we find)[^.]*\.',
            r'(?:Step \d+|First|Initially|Next|Then).*?(?:Finally|In conclusion|Therefore|we obtain|we get)[^.]*\.',
            r'(?:Let us|Consider|Suppose|Assume).*?(?:differential|equation|formula|energy|force|field).*?(?:solution|result|answer|we obtain|we get)[^.]*\.',
            # More flexible patterns for viXra
            r'(?:Using|Applying|From|By).*?(?:equation|formula|law|principle).*?(?:we get|we obtain|we find|this gives|therefore)[^.]*\.',
            r'(?:Substituting|Replacing|Setting|With).*?(?:=|equals).*?(?:we get|we obtain|we find|this gives)[^.]*\.'
        ]
        
        derivations = []
        for pattern in derivation_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            # More lenient filtering for viXra content
            for match in matches:
                if (len(match) > 100 and len(match) < 2500 and  # More flexible length
                    self._contains_english_physics_terms(match)):
                    derivations.append(match)
        
        return derivations[:7]  # Allow more derivations
    
    def _find_worked_solutions(self, text: str) -> List[str]:
        """Find worked solutions to physics problems"""
        solution_patterns = [
            r'(?:Problem|Example|Exercise).*?(?:Solution|Answer).*?(?:\n\n|\d+\.|\Z)',
            r'(?:Given|Known).*?(?:Find|Calculate|Determine).*?(?:Solution|Answer).*?(?:\n\n|\Z)',
            r'(?:Let us solve|To solve|Solving|We solve).*?(?:The result is|We find|The answer is|we get|we obtain)[^.]*\.',
            # More flexible patterns for viXra
            r'(?:Calculate|Computing|Finding|Determining).*?(?:=|equals|gives|yields).*?(?:\n|\.|;)',
            r'(?:Using|With|From).*?(?:equation|formula|relation).*?(?:we get|we obtain|we find|this gives|therefore)[^.]*\.'
        ]
        
        solutions = []
        for pattern in solution_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if (len(match) > 80 and len(match) < 1800 and  # More flexible length
                    self._contains_english_physics_terms(match)):
                    solutions.append(match)
        
        return solutions[:7]  # Allow more solutions
    
    def _find_mathematical_proofs(self, text: str) -> List[str]:
        """Find mathematical proofs and detailed calculations"""
        proof_patterns = [
            r'(?:Proof|Demonstration).*?(?:Q\.E\.D\.|This completes|End of proof)',
            r'(?:We prove|To prove).*?(?:Therefore|Hence|Thus)[^.]*\.',
            r'(?:By|Using|From).*?(?:equation|formula).*?(?:we get|we obtain|it follows)[^.]*\.'
        ]
        
        proofs = []
        for pattern in proof_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                if (len(match) > 100 and len(match) < 1200 and 
                    self._contains_english_physics_terms(match)):
                    proofs.append(match)
        
        return proofs[:3]
    
    def _contains_english_physics_terms(self, text: str) -> bool:
        """Check if text contains English physics terms"""
        english_terms = ['equation', 'energy', 'force', 'field', 'particle', 'wave', 'quantum', 'mass', 'velocity', 'acceleration']
        # Additional terms common in viXra papers
        vixra_terms = ['gravity', 'gravitational', 'universe', 'cosmic', 'theory', 'model', 'physics', 'space', 'time']
        common_english = ['the', 'and', 'of', 'to', 'a', 'in', 'we', 'can', 'from', 'with', 'is', 'are', 'this', 'that']
        
        text_lower = text.lower()
        physics_count = sum(1 for term in english_terms if term in text_lower)
        vixra_count = sum(1 for term in vixra_terms if term in text_lower)
        english_count = sum(1 for word in common_english if word in text_lower)
        
        # More lenient requirements: at least 1 physics term OR 2 vixra terms, AND some English
        has_physics_content = physics_count >= 1 or vixra_count >= 2
        has_english = english_count >= 2
        
        return has_physics_content and has_english
    
    def _create_training_example(self, text_block: str, paper, example_type: str) -> Optional[Dict[str, Any]]:
        """Create a structured training example from a text block"""
        
        # Clean and preprocess the text
        cleaned_text = self._clean_text(text_block)
        
        if len(cleaned_text) < 50:  # Skip if too short
            return None
        
        # Extract steps
        steps = self._extract_reasoning_steps(cleaned_text)
        
        if len(steps) < 2:  # Need at least 2 steps for chain of thought
            return None
        
        # Determine difficulty and topic
        difficulty = self._assess_difficulty(cleaned_text)
        topic = self._categorize_topic(cleaned_text, paper.subject)
        
        # Extract problem statement and solution
        problem, solution_steps = self._separate_problem_solution(steps)
        
        training_example = {
            "id": f"{paper.id}_{example_type}_{hash(cleaned_text) % 10000}",
            "paper_metadata": {
                "source_paper": paper.id,
                "title": self._clean_title_for_training(paper.title),
                "subject": paper.subject,
                "authors": paper.authors
            },
            "problem_statement": problem,
            "solution_steps": solution_steps,
            "metadata": {
                "difficulty": difficulty,
                "topic": topic,
                "example_type": example_type,
                "step_count": len(solution_steps),
                "prerequisites": self._identify_prerequisites(cleaned_text),
                "concepts": self._extract_physics_concepts(cleaned_text)
            },
            "raw_text": self._limit_raw_text(cleaned_text),
            "created_at": datetime.now().isoformat()
        }
        
        return training_example
    
    def _clean_title_for_training(self, title: str) -> str:
        """Clean title specifically for training data"""
        if not title:
            return "Physics paper"
        
        # Remove viXra admin notes and page counts
        title = re.sub(r'\(Note by viXra Admin:.*?\)', '', title, flags=re.IGNORECASE)
        title = re.sub(r'^\d+\s+Pages?\.\s*', '', title, flags=re.IGNORECASE)
        title = re.sub(r'\s+', ' ', title).strip()
        
        if not title or len(title) < 5:
            return "Physics paper"
        
        return title
    
    def _limit_raw_text(self, text: str) -> str:
        """Limit and clean raw text for training examples"""
        if not text:
            return ""
        
        # Limit length to prevent overwhelming training examples
        if len(text) > 1500:
            text = text[:1500] + "..."
        
        # Final encoding cleanup
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        return text.strip()
    
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text for training"""
        if not text:
            return ""
        
        # Handle encoding issues
        text = text.encode('utf-8', errors='ignore').decode('utf-8')
        
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Normalize equations (keep LaTeX formatting)
        text = re.sub(r'\$([^$]+)\$', r'$\1$', text)  # Ensure consistent LaTeX delimiters
        
        # Remove references in brackets
        text = re.sub(r'\[\d+\]', '', text)
        
        # Remove page numbers and figure references
        text = re.sub(r'Page \d+ of \d+', '', text)
        text = re.sub(r'Figure \d+', '', text)
        
        return text.strip()
    
    def _extract_reasoning_steps(self, text: str) -> List[str]:
        """Extract individual reasoning steps from text"""
        step_patterns = [
            r'(?:Step \d+|First|Second|Third|Next|Then|Finally|Therefore|Hence|Thus)[^.]*\.',
            r'(?:We have|We get|We obtain|We find|We calculate|We derive)[^.]*\.',
            r'(?:Using|Applying|From|By)[^.]*(?:equation|formula|law|principle)[^.]*\.',
            r'(?:Substituting|Replacing|Setting)[^.]*=',
            # Additional patterns for viXra papers
            r'(?:Given|Assume|Consider|Let)[^.]*\.',
            r'(?:This gives|This yields|We see that|It follows that)[^.]*\.',
            r'(?:Since|Because|As)[^.]*(?:we have|we get|we obtain)[^.]*\.'
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_step = match.strip()
                if (len(clean_step) > 8 and len(clean_step) < 250 and  # More flexible length
                    self._is_meaningful_step(clean_step)):
                    steps.append(clean_step)
        
        # If no clear steps found, try sentence-based extraction with lower threshold
        if len(steps) < 2:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if (len(clean_sentence) > 15 and len(clean_sentence) < 180 and 
                    self._is_meaningful_step(clean_sentence)):
                    steps.append(clean_sentence)
        
        return steps[:10]  # Allow more steps for viXra
    
    def _is_meaningful_step(self, step: str) -> bool:
        """Check if a step contains meaningful physics/math content"""
        meaningful_indicators = ['equation', 'energy', 'force', 'calculate', 'derive', 'obtain', 'result', 'therefore', 'using', 'from']
        # Additional indicators for viXra papers
        vixra_indicators = ['gravity', 'universe', 'theory', 'model', 'space', 'time', 'field', 'particle', 'mass', 'velocity']
        avoid_indicators = ['page', 'figure', 'table', 'reference', 'citation', 'admin', 'note']
        
        step_lower = step.lower()
        
        has_meaningful = any(indicator in step_lower for indicator in meaningful_indicators)
        has_vixra = any(indicator in step_lower for indicator in vixra_indicators)
        has_avoid = any(indicator in step_lower for indicator in avoid_indicators)
        has_common_words = any(word in step_lower for word in ['the', 'and', 'of', 'to', 'we', 'is', 'this', 'that'])
        
        # More lenient: meaningful OR vixra terms, but not avoid terms
        return (has_meaningful or has_vixra) and not has_avoid and has_common_words
    
    def _assess_difficulty(self, text: str) -> str:
        """Assess the difficulty level of the content"""
        advanced_indicators = [
            'tensor', 'manifold', 'lagrangian', 'hamiltonian', 'variational',
            'differential geometry', 'lie group', 'quantum field theory',
            'gauge theory', 'renormalization', 'symmetry breaking'
        ]
        
        intermediate_indicators = [
            'partial derivative', 'vector calculus', 'fourier transform',
            'wave equation', 'maxwell equations', 'quantum mechanics',
            'statistical mechanics', 'thermodynamics'
        ]
        
        text_lower = text.lower()
        
        advanced_count = sum(1 for indicator in advanced_indicators if indicator in text_lower)
        intermediate_count = sum(1 for indicator in intermediate_indicators if indicator in text_lower)
        
        if advanced_count > 0:
            return "advanced"
        elif intermediate_count > 1:
            return "intermediate"
        else:
            return "introductory"
    
    def _categorize_topic(self, text: str, subject: str) -> str:
        """Categorize the physics topic"""
        topic_keywords = {
            "mechanics": ["force", "acceleration", "momentum", "energy", "kinematics"],
            "electromagnetism": ["electric", "magnetic", "field", "charge", "current", "maxwell"],
            "thermodynamics": ["temperature", "entropy", "heat", "thermal", "gas"],
            "quantum": ["quantum", "wave function", "operator", "eigenvalue", "spin"],
            "relativity": ["relativity", "spacetime", "lorentz", "minkowski", "metric"],
            "optics": ["light", "optical", "photon", "interference", "diffraction"],
            "statistical": ["statistical", "distribution", "probability", "ensemble"],
            "astrophysics": ["stellar", "galactic", "cosmology", "black hole", "gravity"]
        }
        
        text_lower = text.lower()
        topic_scores = {}
        
        for topic, keywords in topic_keywords.items():
            score = sum(1 for keyword in keywords if keyword in text_lower)
            topic_scores[topic] = score
        
        if topic_scores:
            best_topic = max(topic_scores, key=topic_scores.get)
            if topic_scores[best_topic] > 0:
                return best_topic
        
        return subject.lower() if subject else "general"
    
    def _separate_problem_solution(self, steps: List[str]) -> Tuple[str, List[str]]:
        """Separate problem statement from solution steps"""
        if not steps:
            return "Problem statement not clearly identified.", []
        
        problem_indicators = ["given", "find", "calculate", "determine", "show", "prove", "consider", "let"]
        solution_indicators = ["solution", "answer", "we start", "beginning", "first step", "step 1"]
        
        problem_parts = []
        solution_parts = []
        
        in_solution = False
        for i, step in enumerate(steps):
            step_lower = step.lower()
            
            if not in_solution and any(indicator in step_lower for indicator in solution_indicators):
                in_solution = True
            
            if i < 2 and not in_solution:
                if any(indicator in step_lower for indicator in problem_indicators):
                    problem_parts.append(step)
                    continue
            
            if in_solution or i >= 2:
                solution_parts.append(step)
            else:
                problem_parts.append(step)
        
        if problem_parts:
            problem = " ".join(problem_parts)
        else:
            problem = "Derivation or proof:"
        
        if not solution_parts and steps:
            solution_parts = steps
            problem = "Problem statement not clearly identified."
        
        return problem, solution_parts
    
    def _identify_prerequisites(self, text: str) -> List[str]:
        """Identify prerequisite knowledge needed"""
        prereq_indicators = {
            "calculus": ["derivative", "integral", "differential", "partial"],
            "linear_algebra": ["vector", "matrix", "eigenvalue", "determinant"],
            "differential_equations": ["differential equation", "laplace", "boundary condition"],
            "complex_analysis": ["complex", "analytic", "residue", "contour"],
            "group_theory": ["group", "symmetry", "representation", "invariant"],
            "probability": ["probability", "random", "stochastic", "distribution"]
        }
        
        text_lower = text.lower()
        prerequisites = []
        
        for prereq, indicators in prereq_indicators.items():
            if any(indicator in text_lower for indicator in indicators):
                prerequisites.append(prereq)
        
        return prerequisites
    
    def _extract_physics_concepts(self, text: str) -> List[str]:
        """Extract key physics concepts mentioned"""
        concept_patterns = [
            r'(?:conservation of|principle of|law of)\s+\w+',
            r'(?:theorem|equation|formula|relation|transformation)\s+\w+',
            r'(?:model|theory|approximation|method)\s+\w+'
        ]
        
        concepts = []
        for pattern in concept_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts.extend(matches)
        
        return list(set(concepts))
    
    def save_training_data(self, training_examples: List[Dict]) -> str:
        """Save training data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"physics_training_data_{timestamp}.json"
        filepath = self.training_dir / filename
        
        training_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_examples": len(training_examples),
                "difficulty_distribution": self._get_difficulty_distribution(training_examples),
                "topic_distribution": self._get_topic_distribution(training_examples),
                "version": "1.0"
            },
            "training_examples": training_examples
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Training data saved to {filepath}")
        return str(filepath)
    
    def _get_difficulty_distribution(self, examples: List[Dict]) -> Dict[str, int]:
        """Get distribution of difficulty levels"""
        distribution = {}
        for example in examples:
            difficulty = example["metadata"]["difficulty"]
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution
    
    def _get_topic_distribution(self, examples: List[Dict]) -> Dict[str, int]:
        """Get distribution of physics topics"""
        distribution = {}
        for example in examples:
            topic = example["metadata"]["topic"]
            distribution[topic] = distribution.get(topic, 0) + 1
        return distribution

class EnhancedTrainingDataBuilder(TrainingDataBuilder):
    """Enhanced training data builder following UGPhysics standards"""
    
    def _find_complete_derivations(self, text: str) -> List[str]:
        """Find complete derivations with clear step-by-step structure following UGPhysics style"""
        
        # Enhanced patterns for physics derivations
        derivation_patterns = [
            # Mathematical derivation patterns (UGPhysics style)
            r'(?:Given|Starting with|Consider|Let)\s+.*?(?:equation|formula|relation).*?(?:\n.*?)*?(?:Therefore|Thus|Hence|We obtain|Solution)\s*:?\s*.*?(?:\n.*?)*?(?=\n\n|\Z)',
            
            # Physics problem solving patterns
            r'(?:Problem|Question)\s*:?\s*.*?(?:\n.*?)*?(?:Solution|Answer)\s*:?\s*.*?(?:\n.*?)*?(?:Therefore|Hence|Final answer)\s*:?\s*.*?(?=\n\n|\Z)',
            
            # Step-by-step mathematical derivations
            r'(?:Step\s+\d+|First|Initially|Next|Then).*?(?:\n(?:Step\s+\d+|Next|Then|Finally|Therefore).*?)*(?:\n.*?)*?(?=\n\n|\Z)',
            
            # Physics law applications (like UGPhysics examples)
            r'(?:Using|Applying|From)\s+(?:conservation|law|principle|theorem)\s+of\s+\w+.*?(?:\n.*?)*?(?:we get|we obtain|this gives)\s*:?\s*.*?(?=\n\n|\Z)',
            
            # Equation manipulations
            r'(?:From\s+)?(?:equation|relation)\s*\(\d+\).*?(?:\n.*?)*?(?:substituting|rearranging|solving).*?(?:\n.*?)*?(?:we get|we obtain)\s*:?\s*.*?(?=\n\n|\Z)'
        ]
        
        derivations = []
        for pattern in derivation_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                cleaned = self._clean_derivation_text(match)
                if self._is_complete_derivation(cleaned):
                    derivations.append(cleaned)
        
        return derivations[:5]  # Limit to best examples
    
    def _clean_derivation_text(self, text: str) -> str:
        """Clean derivation text to ensure coherent reasoning chains"""
        if not text:
            return ""
        
        # Remove obvious OCR artifacts and formatting issues
        text = re.sub(r'[^\x00-\x7F]+', '', text)  # Remove non-ASCII
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = re.sub(r'\.{3,}', '...', text)  # Normalize ellipses
        text = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', text)  # Add spaces between words
        
        # Remove page numbers, references, and citations
        text = re.sub(r'\[\d+\]', '', text)
        text = re.sub(r'Page\s+\d+', '', text)
        text = re.sub(r'Fig\.\s*\d+', '', text)
        text = re.sub(r'Eq\.\s*\(\d+\)', '', text)
        
        # Ensure complete sentences
        sentences = text.split('.')
        complete_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10 and self._is_meaningful_sentence(sentence):
                complete_sentences.append(sentence)
        
        return '. '.join(complete_sentences) + '.' if complete_sentences else ""
    
    def _is_complete_derivation(self, text: str) -> bool:
        """Check if text forms a complete, coherent derivation"""
        if len(text) < 100:  # Too short
            return False
        
        # Must have both starting point and conclusion
        has_start = any(starter in text.lower() for starter in [
            'given', 'starting with', 'consider', 'let', 'assume', 'suppose'
        ])
        has_conclusion = any(conclusion in text.lower() for conclusion in [
            'therefore', 'thus', 'hence', 'we obtain', 'we get', 'solution', 'result'
        ])
        
        if not (has_start and has_conclusion):
            return False
        
        # Check for mathematical content
        has_math = any(indicator in text.lower() for indicator in [
            'equation', 'formula', 'derivative', 'integral', 'solve', 'calculate'
        ])
        
        # Check for physics content
        has_physics = any(concept in text.lower() for concept in [
            'energy', 'force', 'field', 'particle', 'wave', 'quantum', 'mass', 'velocity'
        ])
        
        return has_math or has_physics
    
    def _is_meaningful_sentence(self, sentence: str) -> bool:
        """Check if sentence contains meaningful physics/math content"""
        if len(sentence) < 15:
            return False
        
        # Must contain some meaningful words
        meaningful_words = ['equation', 'energy', 'force', 'calculate', 'derive', 'solve', 
                           'therefore', 'using', 'given', 'find', 'determine']
        word_count = sum(1 for word in meaningful_words if word in sentence.lower())
        
        # Must have reasonable word density
        words = sentence.split()
        if len(words) < 3:
            return False
        
        return word_count > 0
    
    def _create_training_example(self, text_block: str, paper, example_type: str) -> Optional[Dict[str, Any]]:
        """Create UGPhysics-style training example"""
        
        cleaned_text = self._clean_derivation_text(text_block)
        if len(cleaned_text) < 100:
            return None
        
        # Extract problem and solution following UGPhysics format
        problem, solution_steps = self._extract_problem_solution_ugphysics_style(cleaned_text)
        
        if len(solution_steps) < 2:
            return None
        
        # Enhanced difficulty assessment
        difficulty = self._assess_difficulty_enhanced(cleaned_text)
        topic = self._categorize_topic_enhanced(cleaned_text, paper.subject)
        
        training_example = {
            "id": f"{paper.id}_{example_type}_{hash(cleaned_text) % 10000}",
            "paper_metadata": {
                "source_paper": paper.id,
                "title": self._clean_title_for_training(paper.title),
                "subject": paper.subject,
                "authors": paper.authors
            },
            "problem_statement": problem,
            "solution_steps": solution_steps,
            "metadata": {
                "difficulty": difficulty,
                "topic": topic,
                "example_type": example_type,
                "step_count": len(solution_steps),
                "prerequisites": self._identify_prerequisites_enhanced(cleaned_text),
                "concepts": self._extract_physics_concepts_enhanced(cleaned_text),
                "reasoning_type": self._classify_reasoning_type(cleaned_text)
            },
            "raw_text": self._limit_raw_text(cleaned_text),
            "created_at": datetime.now().isoformat()
        }
        
        return training_example
    
    def _extract_problem_solution_ugphysics_style(self, text: str) -> Tuple[str, List[str]]:
        """Extract problem and solution in UGPhysics format"""
        
        # Look for explicit problem statements
        problem_patterns = [
            r'(?:Problem|Question)\s*:?\s*([^.]*\.(?:[^.]*\.)*)',
            r'(?:Given|Consider|Find|Calculate|Determine|Show|Prove)\s+([^.]*\.(?:[^.]*\.)*)',
            r'(?:A|An)\s+[^.]*(?:particle|wave|field|system)[^.]*\.(?:[^.]*\.)*'
        ]
        
        problem = "Problem statement not clearly identified."
        for pattern in problem_patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                candidate = match.group(1).strip()
                if len(candidate) > 30 and self._is_meaningful_sentence(candidate):
                    problem = candidate
                    break
        
        # Extract solution steps
        steps = self._extract_solution_steps_enhanced(text)
        
        return problem, steps
    
    def _extract_solution_steps_enhanced(self, text: str) -> List[str]:
        """Extract solution steps following UGPhysics methodology"""
        
        # Enhanced step patterns for physics problems
        step_patterns = [
            r'(?:Step\s+\d+|First|Initially|Next|Then|Finally)\s*:?\s*([^.]*\.)',
            r'(?:Using|Applying|From|By)\s+(?:equation|formula|law|principle|conservation)\s+[^.]*\.',
            r'(?:Substituting|Setting|With|Given)\s+[^.]*=.*?\.',
            r'(?:Therefore|Thus|Hence|We obtain|We get|This gives)\s+[^.]*\.',
            r'(?:The|A|An)\s+[^.]*(?:energy|force|momentum|velocity|acceleration)[^.]*\.',
            r'(?:Solving|Calculating|Finding|Determining)\s+[^.]*\.'
        ]
        
        steps = []
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                step = match.strip()
                if len(step) > 20 and self._is_meaningful_step_enhanced(step):
                    steps.append(step)
        
        # If no clear steps, try sentence-based extraction
        if len(steps) < 2:
            sentences = re.split(r'[.!?]+', text)
            for sentence in sentences:
                clean_sentence = sentence.strip()
                if (len(clean_sentence) > 25 and 
                    self._is_meaningful_step_enhanced(clean_sentence)):
                    steps.append(clean_sentence)
        
        return steps[:8]  # Limit to reasonable number
    
    def _is_meaningful_step_enhanced(self, step: str) -> bool:
        """Enhanced check for meaningful physics/math steps"""
        step_lower = step.lower()
        
        # Physics/math indicators
        physics_indicators = ['energy', 'force', 'field', 'particle', 'wave', 'momentum', 
                             'velocity', 'acceleration', 'mass', 'charge', 'potential']
        math_indicators = ['equation', 'formula', 'derivative', 'integral', 'solve', 
                          'calculate', 'substitute', 'equal', 'therefore', 'hence']
        
        has_physics = any(indicator in step_lower for indicator in physics_indicators)
        has_math = any(indicator in step_lower for indicator in math_indicators)
        
        # Avoid meaningless fragments
        avoid_terms = ['page', 'figure', 'table', 'section', 'chapter', 'reference']
        has_avoid = any(term in step_lower for term in avoid_terms)
        
        return (has_physics or has_math) and not has_avoid
    
    def _assess_difficulty_enhanced(self, text: str) -> str:
        """Enhanced difficulty assessment for physics content"""
        text_lower = text.lower()
        
        # Advanced physics concepts
        advanced_concepts = [
            'quantum field theory', 'general relativity', 'gauge theory', 
            'renormalization', 'feynman diagram', 'lagrangian', 'hamiltonian',
            'tensor', 'manifold', 'lie group', 'symmetry breaking'
        ]
        
        # Intermediate concepts  
        intermediate_concepts = [
            'quantum mechanics', 'special relativity', 'electromagnetic field',
            'statistical mechanics', 'thermodynamics', 'wave equation',
            'schrodinger equation', 'maxwell equations', 'fourier transform'
        ]
        
        # Mathematical complexity indicators
        advanced_math = ['partial differential', 'tensor calculus', 'group theory', 
                        'complex analysis', 'differential geometry']
        intermediate_math = ['differential equation', 'linear algebra', 'calculus',
                            'vector calculus', 'complex numbers']
        
        advanced_count = sum(1 for concept in advanced_concepts + advanced_math 
                           if concept in text_lower)
        intermediate_count = sum(1 for concept in intermediate_concepts + intermediate_math 
                               if concept in text_lower)
        
        if advanced_count >= 2:
            return "advanced"
        elif intermediate_count >= 2 or advanced_count >= 1:
            return "intermediate" 
        else:
            return "introductory"
    
    def _classify_reasoning_type(self, text: str) -> str:
        """Classify the type of reasoning following UGPhysics categories"""
        text_lower = text.lower()
        
        if any(term in text_lower for term in ['derive', 'derivation', 'proof', 'show that']):
            return "mathematical_derivation"
        elif any(term in text_lower for term in ['conservation', 'law', 'principle', 'theorem']):
            return "law_application"  
        elif any(term in text_lower for term in ['calculate', 'find', 'determine', 'solve']):
            return "problem_solving"
        elif any(term in text_lower for term in ['given', 'known', 'condition']):
            return "knowledge_recall"
        else:
            return "reasoning_chain"

async def run_enhanced_analysis(args):
    """Run the enhanced paper analysis pipeline"""
    
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / f"analysis_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    analysis_dir = output_base / "analysis_results"
    analysis_dir.mkdir(exist_ok=True)
    
    # Initialize builders
    benchmark_builder = BenchmarkBuilder(str(output_base))
    training_builder = TrainingDataBuilder(str(output_base))
    
    # Determine source and create the appropriate scraper
    if args.source.lower() == "arxiv":
        categories = ARXIV_CATEGORIES
        scraper_class = ArXivScraper
        source_name = "arXiv"
    elif args.source.lower() == "vixra":
        categories = VIXRA_CATEGORIES
        scraper_class = ViXraScraper
        source_name = "viXra"
    else:
        logger.error(f"Unknown source: {args.source}")
        return
    
    # Print header info
    print(f"{source_name} Enhanced Paper Quality Filter ({args.provider.upper()})")
    print("Focus: Papers with subtle physics issues + Benchmark & Training Data Creation")
    print("="*80)
    
    # Handle category selection
    if args.category and args.category != "all":
        if args.source.lower() == "arxiv":
            category_name = categories.get(args.category, args.category)
            print(f"Searching in category: {category_name} ({args.category})")
        else:  # viXra
            from utils.categories import ARXIV_TO_VIXRA_CATEGORIES
            vixra_category = ARXIV_TO_VIXRA_CATEGORIES.get(args.category, args.category)
            category_name = categories.get(vixra_category, vixra_category)
            print(f"Searching in category: {category_name} ({vixra_category})")
    else:
        print(f"Searching across multiple physics categories in {source_name}")
        args.category = None
    
    print(f"Output directory: {output_base}")
    print()
    
    # Initialize classifier
    classifier = SubtlePhysicsClassifier(args.provider, args.openai_key, args.gemini_key)
    
    # Get download directory from args or default
    download_dir = args.download_dir or f"./papers_{args.source.lower()}"
    
    # Scrape papers
    print(f"Scraping recent papers from {source_name}...")
    async with scraper_class(download_dir) as scraper:
        papers = await scraper.get_recent_papers(
            category=args.category, 
            days_back=args.days_back,
            max_papers=args.max_papers
        )
        
        if not papers:
            print("No papers found!")
            return
            
        print(f"Found {len(papers)} papers. Processing each through enhanced analysis...\n")
        
        interesting_papers = []
        benchmark_items = []
        all_training_examples = []
        
        for i, paper in enumerate(papers, 1):
            print(f"PAPER {i}: {paper.title}")
            print(f"Authors: {', '.join(paper.authors)}")
            print(f"Subject: {paper.subject}")
            print(f"Abstract: {paper.abstract[:200]}...")
            print(f"PDF URL: {paper.pdf_url}")
            print()
            
            # Try to download and process PDF
            print("Attempting PDF download...")
            pdf_success = await scraper.download_pdf(paper)
            if pdf_success:
                print(f"PDF downloaded successfully: {paper.pdf_path}")
                paper.full_text, paper.metadata = EnhancedPDFProcessor.extract_text(paper.pdf_path)
                print(f"PDF processed: {paper.metadata.get('word_count', 0)} words")
            else:
                print("PDF download failed - using abstract only")
                paper.full_text = paper.abstract
            
            # Classify paper
            print("\nAnalyzing for subtle physics issues...")
            try:
                assessment = await classifier.classify_paper(paper, args.depth)
                
                print(f"\nRESULTS:")
                print(f"  Overall Score: {assessment.overall_score:.2f}")
                print(f"  Physics Sophistication: {assessment.physics_sophistication:.2f}")
                print(f"  Recommendation: {assessment.stage_3_recommendation}")
                
                if assessment.subtle_issues:
                    print(f"  Subtle Issues Found:")
                    for issue in assessment.subtle_issues:
                        print(f"    - {issue}")
                
                if assessment.stage_2_scores:
                    print(f"  Technical Scores:")
                    for aspect, score in assessment.stage_2_scores.items():
                        print(f"    {aspect}: {score}/10")
                
                print(f"  Analysis: {assessment.reasoning[:300]}...")
                
                # Save individual analysis results
                analysis_file = analysis_dir / f"analysis_{paper.id}.json"
                
                # Clean and limit full text sample
                full_text_sample = None
                if paper.full_text:
                    sample_text = paper.full_text[:5000]
                    sample_text = sample_text.encode('utf-8', errors='ignore').decode('utf-8')
                    sample_text = re.sub(r'\s+', ' ', sample_text).strip()
                    full_text_sample = sample_text
                
                analysis_data = {
                    "paper": {
                        "id": paper.id,
                        "title": benchmark_builder._clean_title(paper.title),
                        "authors": paper.authors,
                        "subject": paper.subject,
                        "abstract": benchmark_builder._clean_abstract(paper.abstract),
                        "submission_date": paper.submission_date,
                        "pdf_url": paper.pdf_url,
                        "metadata": paper.metadata
                    },
                    "assessment": {
                        "overall_score": assessment.overall_score,
                        "stage_1_pass": assessment.stage_1_pass,
                        "stage_2_scores": assessment.stage_2_scores,
                        "stage_3_recommendation": assessment.stage_3_recommendation,
                        "subtle_issues": assessment.subtle_issues,
                        "physics_sophistication": assessment.physics_sophistication,
                        "reasoning": assessment.reasoning,
                        "processing_timestamp": assessment.processing_timestamp
                    },
                    "full_text_sample": full_text_sample
                }
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, indent=2, ensure_ascii=False)
                
                print(f"  Analysis saved to: {analysis_file}")
                
                # Track interesting papers
                if assessment.overall_score >= 0.4:
                    interesting_papers.append((paper, assessment))
                
                # Create benchmark items for papers with issues
                if args.create_benchmark and (assessment.subtle_issues or assessment.overall_score >= 0.3):
                    print("  Creating benchmark questions...")
                    benchmark_item = benchmark_builder.create_reasoning_benchmark(
                        paper, assessment, paper.full_text or paper.abstract
                    )
                    if benchmark_item:  # Only add if creation was successful
                        benchmark_items.append(benchmark_item)
                        print(f"  Added {len(benchmark_item['questions'])} benchmark questions")
                    else:
                        print("  Skipped benchmark creation (unsuitable content)")
                
                # Extract training examples for papers with good derivations
                if args.create_training and paper.full_text:
                    print("  Extracting training examples...")
                    training_examples = training_builder.extract_training_examples(paper, paper.full_text)
                    all_training_examples.extend(training_examples)
                    print(f"  Extracted {len(training_examples)} training examples")
                
            except Exception as e:
                logger.error(f"Error analyzing paper {paper.id}: {e}", exc_info=True)
                print(f"  Error analyzing paper: {e}")
                continue
            
            print("\n" + "="*80 + "\n")
        
        # Save benchmark data
        if args.create_benchmark and benchmark_items:
            print("Saving reasoning benchmark...")
            benchmark_file = benchmark_builder.save_benchmark(benchmark_items)
            print(f"Benchmark saved with {len(benchmark_items)} papers and {sum(len(item['questions']) for item in benchmark_items)} questions")
            print(f"Location: {benchmark_file}")
        
        # Save training data
        if args.create_training and all_training_examples:
            print("Saving training data...")
            training_file = training_builder.save_training_data(all_training_examples)
            print(f"Training data saved with {len(all_training_examples)} examples")
            print(f"Location: {training_file}")
        
        # Summary
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*50)
        print(f"Total papers analyzed: {len(papers)}")
        print(f"Papers with interesting issues: {len(interesting_papers)}")
        print(f"Individual analysis files created: {len(papers)}")
        if args.create_benchmark:
            print(f"Benchmark questions created: {sum(len(item['questions']) for item in benchmark_items)}")
        if args.create_training:
            print(f"Training examples extracted: {len(all_training_examples)}")
        print(f"All outputs saved to: {output_base}")
        
        if interesting_papers:
            print("\nMost interesting papers found:")
            interesting_papers.sort(key=lambda x: x[1].overall_score, reverse=True)
            
            for j, (paper, assessment) in enumerate(interesting_papers[:3], 1):
                print(f"\n{j}. {paper.title}")
                print(f"   Score: {assessment.overall_score:.2f}")
                print(f"   Type: {assessment.stage_3_recommendation}")
                print(f"   Key Issues: {', '.join(assessment.subtle_issues[:3])}")
        
        # Create summary report
        summary_file = output_base / "analysis_summary.json"
        summary_data = {
            "analysis_metadata": {
                "timestamp": timestamp,
                "source": args.source,
                "category": args.category,
                "total_papers": len(papers),
                "interesting_papers": len(interesting_papers),
                "provider": args.provider,
                "depth": args.depth
            },
            "interesting_papers": [
                {
                    "paper_id": paper.id,
                    "title": benchmark_builder._clean_title(paper.title),
                    "score": assessment.overall_score,
                    "recommendation": assessment.stage_3_recommendation,
                    "issue_count": len(assessment.subtle_issues)
                }
                for paper, assessment in interesting_papers
            ],
            "benchmark_stats": {
                "items_created": len(benchmark_items),
                "total_questions": sum(len(item['questions']) for item in benchmark_items)
            } if args.create_benchmark else None,
            "training_stats": {
                "examples_created": len(all_training_examples),
                "difficulty_distribution": training_builder._get_difficulty_distribution(all_training_examples),
                "topic_distribution": training_builder._get_topic_distribution(all_training_examples)
            } if args.create_training and all_training_examples else None
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nSummary report saved to: {summary_file}")

def list_categories(args):
    """List available categories for the selected source"""
    categories = ARXIV_CATEGORIES if args.source.lower() == "arxiv" else VIXRA_CATEGORIES
    
    print(f"Available {args.source} categories:")
    print("="*50)
    for code, name in categories.items():
        print(f"  {code:15} - {name}")
    print(f"\nUse --category <code> to filter by category, or --category all for all categories")

def main():
    """Main entry point for the enhanced script"""
    parser = argparse.ArgumentParser(description="Enhanced Paper Quality Filter - Builds reasoning benchmarks and training data")
    parser.add_argument("--source", choices=["arxiv", "vixra"], required=True, help="Source repository to scrape")
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai", help="LLM provider to use")
    parser.add_argument("--openai-key", help="OpenAI API key (required for OpenAI provider)")
    parser.add_argument("--gemini-key", help="Google Gemini API key (required for Gemini provider)")
    parser.add_argument("--category", help="Category to process (use --list-categories to see options)")
    parser.add_argument("--days-back", type=int, default=30, help="Days back to scrape")
    parser.add_argument("--max-papers", type=int, default=5, help="Maximum papers to process")
    parser.add_argument("--download-dir", help="Directory to download papers to")
    parser.add_argument("--output-dir", default="./enhanced_analysis", help="Base directory for all outputs")
    parser.add_argument("--depth", choices=["basic", "technical", "full", "force"], default="full", 
                       help="Analysis depth (basic=stage 1 only, technical=stages 1-2, full=all stages, force=analyze all papers)")
    parser.add_argument("--create-benchmark", action="store_true", default=True, 
                       help="Create reasoning benchmark from analyzed papers")
    parser.add_argument("--create-training", action="store_true", default=True,
                       help="Extract training data for reinforcement learning")
    parser.add_argument("--list-categories", action="store_true", help="List available categories for the selected source and exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    
    args = parser.parse_args()
    
    # Set logging level
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        for handler in logging.getLogger().handlers:
            handler.setLevel(logging.DEBUG)
    
    # List categories if requested
    if args.list_categories:
        list_categories(args)
        return
    
    # Validate API keys
    if args.provider == "openai" and not args.openai_key:
        parser.error("--openai-key is required when using OpenAI provider")
    elif args.provider == "gemini" and not args.gemini_key:
        parser.error("--gemini-key is required when using Gemini provider")
    
    try:
        asyncio.run(run_enhanced_analysis(args))
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()