#!/usr/bin/env python3
"""
COMPLETE Enhanced Paper Quality Filter - Fixed Integration
Creates self-contained benchmarks using ACTUAL paper concepts + RL training data
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

# These imports should work with your existing codebase
from scrapers.arxiv_scraper import ArXivScraper
from scrapers.vixra_scraper import ViXraScraper
from analysis.pdf_processor import EnhancedPDFProcessor
from analysis.classifier import SubtlePhysicsClassifier
from utils.categories import ARXIV_CATEGORIES, VIXRA_CATEGORIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ActualConceptBenchmarkBuilder:
    """Creates benchmarks using ACTUAL concepts from papers (not generic templates)"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.benchmark_dir = self.output_dir / "benchmark"
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
    def create_benchmark_from_paper(self, paper, assessment, full_text: str) -> Optional[Dict[str, Any]]:
        """Create benchmark using ACTUAL concepts from this specific paper"""
        
        if not self._is_suitable_for_benchmark(paper, full_text):
            return None
        
        # Extract ACTUAL concepts from this paper
        extracted_concepts = self._extract_paper_concepts(full_text, paper)
        if not extracted_concepts or not extracted_concepts.get("has_content"):
            return None
        
        # Generate problems using the extracted concepts
        problems = self._generate_problems_from_paper_concepts(extracted_concepts, paper, assessment)
        
        if len(problems) < 1:
            return None
        
        return {
            "metadata": {
                "source_paper": {
                    "id": paper.id,
                    "title": paper.title,
                    "subject": paper.subject,
                    "authors": paper.authors
                },
                "domain": self._classify_domain(paper.subject),
                "problem_count": len(problems),
                "created_at": datetime.now().isoformat(),
                "version": "3.0_actual_concepts"
            },
            "problems": problems
        }
    
    def _extract_paper_concepts(self, text: str, paper) -> Dict[str, Any]:
        """Extract ACTUAL concepts from this specific paper"""
        
        concepts = {
            "equations": [],
            "derivations": [],
            "numerical_examples": [],
            "physics_scenarios": [],
            "has_content": False
        }
        
        # Extract actual equations with their context
        eq_patterns = [
            r'([A-Za-z_]\w*\s*=\s*[^,.\n]{8,60})',
            r'((?:E|F|V|p|m|v|a|g|H|L|T|P|ρ|σ|ω|λ|μ|ε)\s*=\s*[^,.\n]{5,50})',
            r'(\\frac\{[^}]+\}\{[^}]+\}(?:\s*[=+\-]\s*[^.]{0,40})?)',
        ]
        
        for pattern in eq_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = self._clean_equation(match)
                if cleaned and self._is_physics_equation(cleaned):
                    concepts["equations"].append(cleaned)
        
        # Extract derivation sequences
        derivation_patterns = [
            r'(?:Starting with|From|Given)\s+([^.]*?)(?:we get|we obtain|this gives)\s+([^.]*?)(?:\.|\n)',
            r'(?:Using|Substituting)\s+([^.]*?)(?:into|in)\s+([^.]*?)(?:\.|\n)',
        ]
        
        for pattern in derivation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for start, result in matches:
                if len(start.strip()) > 15 and len(result.strip()) > 10:
                    concepts["derivations"].append({
                        "starting_point": start.strip(),
                        "result": result.strip()
                    })
        
        # Extract numerical examples
        numerical_patterns = [
            r'([A-Za-z_]\w*\s*=\s*[0-9]+\.?[0-9]*(?:\s*×\s*10[⁻⁰-⁹]+)?\s*(?:m|kg|s|Hz|eV|K|Pa|N|J|W|V|A|T|rad)?)',
            r'((?:wavelength|frequency|energy|mass|velocity|temperature|pressure)\s*(?:of|=|is)\s*[0-9][^.\n]*)',
        ]
        
        for pattern in numerical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if self._is_meaningful_numerical(match):
                    concepts["numerical_examples"].append(match.strip())
        
        # Extract physics scenarios described in the paper
        scenario_patterns = [
            r'(?:Consider|Suppose|Let)\s+([^.]*?(?:particle|system|field|wave|oscillator)[^.]*?)(?:\.|\n)',
            r'(?:We study|We consider|In this work)\s+([^.]*?)(?:\.|\n)',
        ]
        
        for pattern in scenario_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if len(match.strip()) > 25 and self._contains_physics_content(match):
                    concepts["physics_scenarios"].append(match.strip())
        
        # Check if we have meaningful content
        concepts["has_content"] = (len(concepts["equations"]) > 0 or 
                                 len(concepts["derivations"]) > 0 or 
                                 len(concepts["numerical_examples"]) > 0)
        
        return concepts
    
    def _generate_problems_from_paper_concepts(self, concepts: Dict, paper, assessment) -> List[Dict]:
        """Generate problems using the ACTUAL extracted concepts"""
        
        problems = []
        
        # Problem 1: Equation Analysis (using actual equations)
        if concepts["equations"]:
            eq_problem = self._create_equation_analysis_problem(concepts, paper)
            if eq_problem:
                problems.append(eq_problem)
        
        # Problem 2: Derivation Verification (using actual derivations)
        if concepts["derivations"]:
            deriv_problem = self._create_derivation_problem(concepts, paper)
            if deriv_problem:
                problems.append(deriv_problem)
        
        # Problem 3: Numerical Analysis (using actual numbers)
        if concepts["numerical_examples"]:
            num_problem = self._create_numerical_problem(concepts, paper)
            if num_problem:
                problems.append(num_problem)
        
        # Problem 4: Scenario Analysis (using actual scenarios)
        if concepts["physics_scenarios"]:
            scenario_problem = self._create_scenario_problem(concepts, paper, assessment)
            if scenario_problem:
                problems.append(scenario_problem)
        
        return problems
    
    def _create_equation_analysis_problem(self, concepts: Dict, paper) -> Dict:
        """Create problem analyzing actual equations from the paper"""
        
        primary_eq = concepts["equations"][0]
        additional_eqs = concepts["equations"][1:3] if len(concepts["equations"]) > 1 else []
        
        problem_statement = f"""Analyze the following physical relationship extracted from a {paper.subject.lower()} study:

**Primary Equation**: {primary_eq}
"""
        
        if additional_eqs:
            problem_statement += "\n**Related Equations**:\n"
            for eq in additional_eqs:
                problem_statement += f"• {eq}\n"
        
        problem_statement += f"""
**Analysis Tasks**:
1. **Dimensional Analysis**: Verify that all terms have consistent dimensions
2. **Physical Interpretation**: Explain the physical meaning of each variable
3. **Domain of Validity**: Under what conditions does this relationship apply?
4. **Mathematical Structure**: Identify the mathematical form (linear, quadratic, exponential, etc.)
5. **Limiting Cases**: What happens in extreme limits of the variables?
6. **Experimental Verification**: How could this relationship be tested experimentally?

Provide a comprehensive physics analysis addressing each point."""
        
        return {
            "problem_id": f"equation_analysis_{paper.id}_{hash(primary_eq) % 10000}",
            "type": "equation_analysis",
            "difficulty": "intermediate",
            "domain": self._classify_domain(paper.subject),
            "problem_statement": problem_statement,
            "source_equations": concepts["equations"][:3],
            "evaluation_criteria": [
                "Correct dimensional analysis",
                "Physical understanding of variables",
                "Recognition of validity limits",
                "Mathematical insight"
            ]
        }
    
    def _create_derivation_problem(self, concepts: Dict, paper) -> Dict:
        """Create problem using actual derivation from the paper"""
        
        derivation = concepts["derivations"][0]
        
        problem_statement = f"""The following derivation sequence appears in a {paper.subject.lower()} analysis:

**Starting Point**: {derivation['starting_point']}

**Claimed Result**: {derivation['result']}

**Verification Tasks**:
1. **Mathematical Validity**: Is the mathematical transition correct?
2. **Missing Steps**: What intermediate steps might be omitted?
3. **Physical Assumptions**: What physics assumptions are made?
4. **Alternative Approaches**: Can you derive the same result differently?
5. **Error Analysis**: If there are errors, identify and correct them
6. **Generalization**: Under what broader conditions does this derivation hold?

Show all mathematical steps clearly and justify each physics assumption."""
        
        return {
            "problem_id": f"derivation_check_{paper.id}_{hash(str(derivation)) % 10000}",
            "type": "derivation_verification",
            "difficulty": "advanced",
            "domain": self._classify_domain(paper.subject),
            "problem_statement": problem_statement,
            "source_derivation": derivation,
            "evaluation_criteria": [
                "Mathematical step verification",
                "Physics assumption identification",
                "Alternative derivation methods",
                "Error detection and correction"
            ]
        }
    
    def _create_numerical_problem(self, concepts: Dict, paper) -> Dict:
        """Create problem using actual numerical values from the paper"""
        
        numerical_data = concepts["numerical_examples"][:3]
        
        problem_statement = f"""The following numerical values appear in a {paper.subject.lower()} study:

**Given Data**:
"""
        for value in numerical_data:
            problem_statement += f"• {value}\n"
        
        problem_statement += f"""
**Calculation Tasks**:
1. **Unit Verification**: Check that all units are consistent and correctly specified
2. **Order of Magnitude**: Verify these values are reasonable for {paper.subject.lower()}
3. **Derived Quantities**: Calculate related physical quantities from this data
4. **Uncertainty Analysis**: Estimate reasonable uncertainty bounds
5. **Comparison**: How do these values compare to standard reference values?
6. **Implications**: What do these numerical results tell us about the physical system?

Show all calculations with proper unit handling and uncertainty propagation."""
        
        return {
            "problem_id": f"numerical_analysis_{paper.id}_{hash(str(numerical_data)) % 10000}",
            "type": "numerical_analysis",
            "difficulty": "intermediate",
            "domain": self._classify_domain(paper.subject),
            "problem_statement": problem_statement,
            "source_data": numerical_data,
            "evaluation_criteria": [
                "Correct unit analysis",
                "Order of magnitude reasoning",
                "Accurate calculations",
                "Physical interpretation"
            ]
        }
    
    def _create_scenario_problem(self, concepts: Dict, paper, assessment) -> Dict:
        """Create problem using actual physics scenario from the paper"""
        
        scenario = concepts["physics_scenarios"][0]
        
        problem_statement = f"""Consider the physical scenario described in a {paper.subject.lower()} study:

**Scenario**: {scenario}

**Physics Analysis Tasks**:
1. **System Identification**: What is the physical system being described?
2. **Relevant Physics**: Which fundamental principles govern this system?
3. **Mathematical Model**: How would you model this system mathematically?
4. **Key Variables**: What are the important physical quantities?
5. **Experimental Setup**: How could this scenario be realized experimentally?
6. **Predictions**: What measurable effects would you expect?
7. **Limitations**: What are the boundaries of this description?

"""
        
        if assessment.subtle_issues:
            problem_statement += f"""**Critical Evaluation**: This work has been identified with potential issues including: {', '.join(assessment.subtle_issues[:2])}. 
Evaluate whether these concerns affect the validity of the scenario description.

"""
        
        problem_statement += """Provide a comprehensive physics analysis that demonstrates deep understanding of the underlying principles."""
        
        return {
            "problem_id": f"scenario_analysis_{paper.id}_{hash(scenario) % 10000}",
            "type": "scenario_analysis",
            "difficulty": "advanced" if assessment.subtle_issues else "intermediate",
            "domain": self._classify_domain(paper.subject),
            "problem_statement": problem_statement,
            "source_scenario": scenario,
            "evaluation_criteria": [
                "System understanding",
                "Physics principle application",
                "Mathematical modeling",
                "Critical evaluation"
            ]
        }
    
    def _is_physics_equation(self, equation: str) -> bool:
        """Check if equation contains meaningful physics content"""
        
        if '=' not in equation or len(equation.strip()) < 5:
            return False
        
        # Check for physics variables
        physics_vars = ['E', 'F', 'V', 'p', 'm', 'v', 'a', 'g', 'H', 'L', 'T', 'P', 'ρ', 'σ', 'ω', 'λ', 'μ', 'ε', 'ψ', 'φ']
        has_physics = any(var in equation for var in physics_vars)
        
        # Check for mathematical operators
        has_math = any(op in equation for op in ['+', '-', '*', '/', '^', '∇', '∂', '∫', '√'])
        
        return has_physics and has_math
    
    def _is_meaningful_numerical(self, value: str) -> bool:
        """Check if numerical value is meaningful"""
        
        if not re.search(r'\d', value):
            return False
        
        # Check for physics units or contexts
        physics_contexts = [
            'm', 'kg', 's', 'Hz', 'eV', 'K', 'Pa', 'N', 'J', 'W', 'V', 'A', 'T', 'rad',
            'wavelength', 'frequency', 'energy', 'mass', 'velocity', 'temperature', 'pressure'
        ]
        
        return any(context in value.lower() for context in physics_contexts) and len(value.strip()) > 4
    
    def _contains_physics_content(self, text: str) -> bool:
        """Check if text contains physics content"""
        
        physics_terms = [
            'energy', 'force', 'field', 'particle', 'wave', 'mass', 'velocity',
            'momentum', 'charge', 'potential', 'frequency', 'wavelength',
            'temperature', 'pressure', 'current', 'magnetic', 'electric'
        ]
        
        text_lower = text.lower()
        return any(term in text_lower for term in physics_terms)
    
    def _clean_equation(self, equation: str) -> str:
        """Clean equation text"""
        cleaned = re.sub(r'\s+', ' ', equation.strip())
        cleaned = re.sub(r'[^\w\s=+\-*/(){}\\.,∇∂∫√α-ωΑ-Ω]', '', cleaned)
        return cleaned
    
    def _classify_domain(self, subject: str) -> str:
        """Classify physics domain"""
        subject_lower = subject.lower()
        
        if any(term in subject_lower for term in ["mechanics", "classical"]):
            return "mechanics"
        elif any(term in subject_lower for term in ["electro", "magnetic", "field"]):
            return "electromagnetism"
        elif any(term in subject_lower for term in ["quantum", "atomic"]):
            return "quantum"
        elif any(term in subject_lower for term in ["thermo", "statistical"]):
            return "thermodynamics"
        elif any(term in subject_lower for term in ["relativity", "gravity"]):
            return "relativity"
        else:
            return "general_physics"
    
    def _is_suitable_for_benchmark(self, paper, full_text: str) -> bool:
        """Check suitability for benchmark creation"""
        
        if not full_text or len(full_text.strip()) < 600:
            return False
        
        # Must have equations or mathematical content
        has_equations = '=' in full_text and len(re.findall(r'[A-Za-z]\s*=', full_text)) >= 1
        
        # Must have physics content
        physics_terms = ['energy', 'force', 'field', 'particle', 'wave', 'mass', 'equation']
        text_sample = full_text[:1500].lower()
        physics_count = sum(1 for term in physics_terms if term in text_sample)
        
        return has_equations or physics_count >= 3
    
    def save_benchmark(self, benchmark_items: List[Dict]) -> str:
        """Save benchmark to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"actual_concepts_benchmark_{timestamp}.json"
        filepath = self.benchmark_dir / filename
        
        benchmark_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "benchmark_type": "actual_paper_concepts_physics_reasoning",
                "total_problems": sum(len(item.get("problems", [])) for item in benchmark_items),
                "total_sets": len(benchmark_items),
                "format_version": "3.0",
                "description": "Physics problems using actual concepts extracted from research papers"
            },
            "problem_sets": benchmark_items
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Actual concepts benchmark saved to {filepath}")
        return str(filepath)


class RLTrainingDataBuilder:
    """Creates RL training data from actual paper content"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.training_dir = self.output_dir / "training_data"
        self.training_dir.mkdir(parents=True, exist_ok=True)
    
    def create_rl_training_examples(self, paper, full_text: str) -> List[Dict[str, Any]]:
        """Create RL training examples from actual paper content"""
        
        if not self._is_suitable_for_training(paper, full_text):
            return []
        
        # Extract actual examples from the paper
        extracted_examples = self._extract_training_examples(full_text, paper)
        
        training_examples = []
        for example in extracted_examples[:2]:  # Limit to best examples
            rl_example = self._create_rl_format(example, paper)
            if rl_example and self._meets_quality_standards(rl_example):
                training_examples.append(rl_example)
        
        return training_examples
    
    def _extract_training_examples(self, text: str, paper) -> List[Dict[str, Any]]:
        """Extract actual problem-solution pairs from paper"""
        
        examples = []
        
        # Pattern 1: Explicit problem/solution
        explicit_patterns = [
            r'(?:Problem|Example|Exercise)\s*:?\s*(.*?)(?:Solution|Answer)\s*:?\s*(.*?)(?=(?:Problem|Example|\n\n|\Z))',
            r'(?:Find|Calculate|Determine)\s+(.*?)(?:\.|\n)\s*(?:We have|We get|We find)\s*(.*?)(?=(?:Find|Calculate|\n\n|\Z))',
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for problem_text, solution_text in matches:
                if len(problem_text.strip()) > 20 and len(solution_text.strip()) > 30:
                    example = {
                        "type": "explicit_problem",
                        "problem": self._clean_text(problem_text),
                        "solution": self._clean_text(solution_text),
                        "quality": self._assess_quality(problem_text, solution_text)
                    }
                    if example["quality"] > 0.4:
                        examples.append(example)
        
        # Pattern 2: Derivation sequences
        derivation_patterns = [
            r'(?:Starting with|From|Given)\s+(.*?)(?:we get|we obtain|this gives)\s+(.*?)(?:Therefore|Thus)\s+(.*?)(?=\n\n|\Z)',
        ]
        
        for pattern in derivation_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for given, process, result in matches:
                if len(given.strip()) > 15 and len(process.strip()) > 20:
                    example = {
                        "type": "derivation",
                        "problem": f"Given: {given.strip()}\nDerive the result and show all steps.",
                        "solution": f"Starting with: {given.strip()}\nProcess: {process.strip()}\nResult: {result.strip()}",
                        "quality": self._assess_quality(given, process + result)
                    }
                    if example["quality"] > 0.4:
                        examples.append(example)
        
        return examples
    
    def _create_rl_format(self, example: Dict, paper) -> Dict[str, Any]:
        """Create RL training format"""
        
        # Create thinking section
        thinking = self._create_thinking_section(example)
        
        # Extract final answer
        final_answer = self._extract_final_answer(example["solution"])
        
        return {
            "id": f"rl_train_{paper.id}_{hash(str(example)) % 100000}",
            "source_metadata": {
                "paper_id": paper.id,
                "paper_title": paper.title,
                "paper_subject": paper.subject,
                "extraction_type": example["type"]
            },
            "prompt": example["problem"],
            "completion": f"<think>\n{thinking}\n</think>\n\n{final_answer}",
            "metadata": {
                "reasoning_quality": example["quality"],
                "difficulty": self._assess_difficulty(example),
                "format_version": "rl_v3.0"
            },
            "created_at": datetime.now().isoformat()
        }
    
    def _create_thinking_section(self, example: Dict) -> str:
        """Create thinking section for RL training"""
        
        solution_parts = example["solution"].split(". ")
        
        thinking_parts = [
            "Let me work through this physics problem step by step.",
            ""
        ]
        
        for i, part in enumerate(solution_parts[:4]):  # Limit to avoid too long
            if len(part.strip()) > 10:
                if i == 0:
                    thinking_parts.append(f"First, {part.strip()}.")
                else:
                    thinking_parts.append(f"Next, {part.strip()}.")
                thinking_parts.append("")
        
        thinking_parts.append("This gives us the solution following standard physics principles.")
        
        return "\n".join(thinking_parts)
    
    def _extract_final_answer(self, solution: str) -> str:
        """Extract final answer from solution"""
        
        # Look for explicit results
        answer_patterns = [
            r'(?:Therefore|Thus|Hence|Result|Answer)\s*:?\s*([^.]+)',
            r'([^.]*=\s*[0-9][^.]*)',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, solution, re.IGNORECASE)
            if matches:
                return f"Therefore: {matches[-1].strip()}"
        
        # Use last meaningful sentence
        sentences = solution.split('.')
        for sentence in reversed(sentences):
            if len(sentence.strip()) > 10 and ('=' in sentence or any(word in sentence.lower() for word in ['result', 'answer', 'solution'])):
                return sentence.strip()
        
        return "The solution follows from the steps shown above."
    
    def _clean_text(self, text: str) -> str:
        """Clean text for training"""
        cleaned = re.sub(r'\s+', ' ', text.strip())
        cleaned = re.sub(r'[^\w\s\.,;:()=+\-*/\[\]{}\\]', '', cleaned)
        return cleaned
    
    def _assess_quality(self, problem: str, solution: str) -> float:
        """Assess quality of training example"""
        
        # Check for physics content
        physics_terms = ['energy', 'force', 'field', 'mass', 'velocity', 'equation']
        problem_physics = sum(1 for term in physics_terms if term in problem.lower())
        solution_physics = sum(1 for term in physics_terms if term in solution.lower())
        
        # Check for mathematical content
        has_math = '=' in solution or any(op in solution for op in ['+', '-', '*', '/'])
        
        # Check length appropriateness
        problem_length = len(problem.split())
        solution_length = len(solution.split())
        
        length_score = 1.0 if 10 <= problem_length <= 100 and 20 <= solution_length <= 200 else 0.5
        physics_score = min(1.0, (problem_physics + solution_physics) / 4)
        math_score = 1.0 if has_math else 0.5
        
        return (length_score + physics_score + math_score) / 3
    
    def _assess_difficulty(self, example: Dict) -> str:
        """Assess difficulty level"""
        
        content = example["problem"] + " " + example["solution"]
        content_lower = content.lower()
        
        advanced_terms = ['differential', 'integral', 'quantum', 'relativistic']
        intermediate_terms = ['derivative', 'vector', 'conservation', 'electromagnetic']
        
        advanced_count = sum(1 for term in advanced_terms if term in content_lower)
        intermediate_count = sum(1 for term in intermediate_terms if term in content_lower)
        
        if advanced_count >= 1:
            return "advanced"
        elif intermediate_count >= 1:
            return "intermediate"
        else:
            return "introductory"
    
    def _meets_quality_standards(self, rl_example: Dict) -> bool:
        """Check quality standards"""
        
        return (rl_example["metadata"]["reasoning_quality"] > 0.4 and
                len(rl_example["prompt"]) > 25 and
                len(rl_example["completion"]) > 80)
    
    def _is_suitable_for_training(self, paper, full_text: str) -> bool:
        """Check suitability for training data"""
        
        if not full_text or len(full_text.strip()) < 800:
            return False
        
        # Must have problem-solving indicators
        indicators = ['problem', 'solution', 'example', 'calculate', 'find', 'derive']
        text_sample = full_text[:1500].lower()
        indicator_count = sum(1 for indicator in indicators if indicator in text_sample)
        
        return indicator_count >= 3
    
    def save_training_data(self, training_examples: List[Dict]) -> str:
        """Save training data"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rl_physics_training_{timestamp}.json"
        filepath = self.training_dir / filename
        
        training_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "total_examples": len(training_examples),
                "format_version": "3.0",
                "description": "RL training data from actual physics papers"
            },
            "training_examples": training_examples
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"RL training data saved to {filepath}")
        return str(filepath)
    
    def _get_difficulty_distribution(self, examples: List[Dict]) -> Dict[str, int]:
        """Get difficulty distribution"""
        distribution = {}
        for example in examples:
            difficulty = example["metadata"]["difficulty"]
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution


async def run_fixed_enhanced_analysis(args):
    """Run the FIXED enhanced analysis pipeline"""
    
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / f"fixed_enhanced_analysis_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    analysis_dir = output_base / "analysis_results"
    analysis_dir.mkdir(exist_ok=True)
    
    # Initialize FIXED builders
    benchmark_builder = ActualConceptBenchmarkBuilder(str(output_base))
    training_builder = RLTrainingDataBuilder(str(output_base))
    
    # Determine source and create scraper
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
    
    # Print header
    print(f"{source_name} FIXED Enhanced Paper Quality Filter ({args.provider.upper()})")
    print("Focus: ACTUAL paper concepts → Self-contained benchmarks + RL training")
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
    
    # Get download directory
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
        
        print(f"Found {len(papers)} papers. Processing with FIXED enhanced analysis...\n")
        
        interesting_papers = []
        actual_concept_benchmarks = []
        all_rl_training_examples = []
        
        for i, paper in enumerate(papers, 1):
            print(f"PAPER {i}: {paper.title}")
            print(f"Authors: {', '.join(paper.authors)}")
            print(f"Subject: {paper.subject}")
            print(f"Abstract: {paper.abstract[:200]}...")
            print(f"PDF URL: {paper.pdf_url}")
            print()
            
            # Download and process PDF
            print("Attempting enhanced PDF download and processing...")
            pdf_success = await scraper.download_pdf(paper)
            if pdf_success:
                print(f"PDF downloaded successfully: {paper.pdf_path}")
                paper.full_text, paper.metadata = EnhancedPDFProcessor.extract_text_enhanced(paper.pdf_path)
                print(f"Enhanced PDF processed: {paper.metadata.get('word_count', 0)} words")
            else:
                print("PDF download failed - using abstract only")
                paper.full_text = paper.abstract
            
            # Classify paper
            print("\nAnalyzing for subtle physics issues...")
            try:
                assessment = await classifier.classify_paper(paper, args.depth)
                
                print(f"\nASSESSMENT RESULTS:")
                print(f"  Overall Score: {assessment.overall_score:.2f}")
                print(f"  Physics Sophistication: {assessment.physics_sophistication:.2f}")
                print(f"  Recommendation: {assessment.stage_3_recommendation}")
                
                if assessment.subtle_issues:
                    print(f"  Subtle Issues Found:")
                    for issue in assessment.subtle_issues:
                        print(f"    - {issue}")
                
                # Save individual analysis
                analysis_file = analysis_dir / f"analysis_{paper.id}.json"
                analysis_data = {
                    "paper": {
                        "id": paper.id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "subject": paper.subject,
                        "abstract": paper.abstract,
                        "pdf_url": paper.pdf_url
                    },
                    "assessment": {
                        "overall_score": assessment.overall_score,
                        "subtle_issues": assessment.subtle_issues,
                        "physics_sophistication": assessment.physics_sophistication,
                        "reasoning": assessment.reasoning
                    }
                }
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, indent=2, ensure_ascii=False)
                
                # Track interesting papers
                if assessment.overall_score >= 0.4:
                    interesting_papers.append((paper, assessment))
                
                # Create ACTUAL CONCEPT benchmarks
                if args.create_benchmark and paper.full_text:
                    print("  Creating benchmark using ACTUAL paper concepts...")
                    benchmark_item = benchmark_builder.create_benchmark_from_paper(
                        paper, assessment, paper.full_text
                    )
                    if benchmark_item:
                        actual_concept_benchmarks.append(benchmark_item)
                        problem_count = len(benchmark_item.get("problems", []))
                        print(f"  ✓ Created {problem_count} problems using actual concepts from paper")
                    else:
                        print("  ✗ Insufficient extractable concepts for benchmark")
                
                # Extract RL training examples
                if args.create_training and paper.full_text:
                    print("  Extracting RL training examples from paper content...")
                    rl_examples = training_builder.create_rl_training_examples(paper, paper.full_text)
                    if rl_examples:
                        all_rl_training_examples.extend(rl_examples)
                        avg_quality = sum(ex["metadata"]["reasoning_quality"] for ex in rl_examples) / len(rl_examples)
                        print(f"  ✓ Extracted {len(rl_examples)} examples (avg quality: {avg_quality:.2f})")
                    else:
                        print("  ✗ No suitable training examples found")
                
            except Exception as e:
                logger.error(f"Error analyzing paper {paper.id}: {e}", exc_info=True)
                print(f"  ✗ Error analyzing paper: {e}")
                continue
            
            print("\n" + "="*80 + "\n")
        
        # Save enhanced benchmark data
        if args.create_benchmark and actual_concept_benchmarks:
            print("Saving benchmarks using ACTUAL paper concepts...")
            benchmark_file = benchmark_builder.save_benchmark(actual_concept_benchmarks)
            total_problems = sum(len(item.get("problems", [])) for item in actual_concept_benchmarks)
            print(f"✓ Benchmark saved with {len(actual_concept_benchmarks)} paper concept sets")
            print(f"✓ Total problems using actual concepts: {total_problems}")
            print(f"Location: {benchmark_file}")
        
        # Save RL training data
        if args.create_training and all_rl_training_examples:
            print("Saving RL training data from actual paper content...")
            training_file = training_builder.save_training_data(all_rl_training_examples)
            high_quality_count = sum(1 for ex in all_rl_training_examples 
                                   if ex["metadata"]["reasoning_quality"] > 0.6)
            print(f"✓ Training data saved with {len(all_rl_training_examples)} examples")
            print(f"✓ High quality examples (>0.6): {high_quality_count}")
            print(f"Location: {training_file}")
        
        # Final Summary
        print("\n" + "="*80)
        print("FIXED ENHANCED ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total papers processed: {len(papers)}")
        print(f"Papers with interesting issues: {len(interesting_papers)}")
        
        if args.create_benchmark:
            total_problems = sum(len(item.get("problems", [])) for item in actual_concept_benchmarks)
            print(f"Benchmarks using actual paper concepts: {len(actual_concept_benchmarks)}")
            print(f"Total problems created: {total_problems}")
        
        if args.create_training:
            if all_rl_training_examples:
                avg_quality = sum(ex["metadata"]["reasoning_quality"] for ex in all_rl_training_examples) / len(all_rl_training_examples)
                print(f"RL training examples: {len(all_rl_training_examples)}")
                print(f"Average quality score: {avg_quality:.2f}")
        
        print(f"All outputs saved to: {output_base}")
        
        # Show interesting papers
        if interesting_papers:
            print("\nMost interesting papers:")
            interesting_papers.sort(key=lambda x: x[1].overall_score, reverse=True)
            
            for j, (paper, assessment) in enumerate(interesting_papers[:3], 1):
                clean_title = paper.title[:70] + "..." if len(paper.title) > 70 else paper.title
                print(f"\n{j}. {clean_title}")
                print(f"   Score: {assessment.overall_score:.2f}")
                print(f"   Type: {assessment.stage_3_recommendation}")


def list_categories(args):
    """List available categories"""
    categories = ARXIV_CATEGORIES if args.source.lower() == "arxiv" else VIXRA_CATEGORIES
    
    print(f"Available {args.source} categories:")
    print("="*50)
    for code, name in categories.items():
        print(f"  {code:15} - {name}")
    print(f"\nUse --category <code> to filter by category")


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="FIXED Enhanced Paper Quality Filter - Uses ACTUAL paper concepts")
    parser.add_argument("--source", choices=["arxiv", "vixra"], required=True, help="Source repository")
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai", help="LLM provider")
    parser.add_argument("--openai-key", help="OpenAI API key")
    parser.add_argument("--gemini-key", help="Google Gemini API key")
    parser.add_argument("--category", help="Category to process")
    parser.add_argument("--days-back", type=int, default=30, help="Days back to scrape")
    parser.add_argument("--max-papers", type=int, default=5, help="Maximum papers to process")
    parser.add_argument("--download-dir", help="Directory to download papers")
    parser.add_argument("--output-dir", default="./fixed_enhanced_analysis", help="Output directory")
    parser.add_argument("--depth", choices=["basic", "technical", "full", "force"], default="full", help="Analysis depth")
    parser.add_argument("--create-benchmark", action="store_true", default=True, help="Create benchmarks")
    parser.add_argument("--create-training", action="store_true", default=True, help="Create RL training data")
    parser.add_argument("--list-categories", action="store_true", help="List categories")
    parser.add_argument("--debug", action="store_true", help="Debug logging")
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    if args.list_categories:
        list_categories(args)
        return
    
    # Validate API keys
    if args.provider == "openai" and not args.openai_key:
        parser.error("--openai-key is required when using OpenAI provider")
    elif args.provider == "gemini" and not args.gemini_key:
        parser.error("--gemini-key is required when using Gemini provider")
    
    try:
        asyncio.run(run_fixed_enhanced_analysis(args))
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()