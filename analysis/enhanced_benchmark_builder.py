"""
Enhanced Benchmark Builder - Creates self-contained physics reasoning problems
Key improvement: Extract ACTUAL concepts from papers while keeping problems self-contained
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class SelfContainedBenchmarkBuilder:
    """Creates self-contained physics reasoning benchmarks using actual paper concepts"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.benchmark_dir = self.output_dir / "benchmark"
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
    def create_self_contained_benchmark(self, paper, assessment, full_text: str) -> Optional[Dict[str, Any]]:
        """Create benchmark using ACTUAL concepts from the paper while keeping problems self-contained"""
        
        if not self._is_suitable_for_benchmark(paper, full_text):
            return None
        
        # Extract ACTUAL physics content from the paper
        extracted_concepts = self._extract_actual_concepts_from_paper(full_text, paper.subject)
        if not extracted_concepts:
            return None
        
        # Generate problems based on ACTUAL extracted concepts
        problems = self._generate_problems_from_extracted_concepts(
            extracted_concepts, paper, assessment
        )
        
        if len(problems) < 1:
            return None
        
        benchmark_item = {
            "metadata": {
                "source_paper": {
                    "id": paper.id,
                    "title": paper.title,
                    "subject": paper.subject,
                    "authors": paper.authors
                },
                "domain": self._classify_physics_domain(paper.subject),
                "difficulty_level": self._assess_problem_difficulty(extracted_concepts),
                "problem_count": len(problems),
                "created_at": datetime.now().isoformat(),
                "version": "2.0",
                "extraction_method": "actual_paper_concepts"
            },
            "problems": problems
        }
        
        return benchmark_item
    
    def _extract_actual_concepts_from_paper(self, text: str, subject: str) -> Dict[str, Any]:
        """Extract ACTUAL physics concepts from the paper text"""
        
        concepts = {
            "equations": [],
            "derivations": [],
            "physical_scenarios": [],
            "mathematical_expressions": [],
            "physics_principles": [],
            "numerical_values": [],
            "problem_setups": []
        }
        
        # Extract actual equations from the paper
        equation_patterns = [
            r'([A-Za-z_]\w*\s*=\s*[^,.\n]{10,80})',  # Variable definitions
            r'(\\frac\{[^}]+\}\{[^}]+\}(?:\s*[=+\-]\s*[^.]{0,50})?)',  # Fraction expressions
            r'((?:E|F|V|p|m|v|a|g|H|L|T|P)\s*=\s*[^,.\n]{5,60})',  # Physics variables
            r'(∇[^.]{5,50})',  # Gradient expressions
            r'(∂[^.]{5,50})',  # Partial derivatives
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = self._clean_equation(match)
                if cleaned and self._is_meaningful_equation(cleaned):
                    concepts["equations"].append(cleaned)
        
        # Extract derivation segments
        derivation_patterns = [
            r'(?:Starting with|Beginning with|From|Given)\s+([^.]*?(?:equation|formula|relation)[^.]*?)(?:\.|\n)',
            r'(?:Substituting|Using|Applying)\s+([^.]*?(?:=|into|yields?)[^.]*?)(?:\.|\n)',
            r'(?:Therefore|Thus|Hence)\s+([^.]*?=\s*[^.]*?)(?:\.|\n)',
        ]
        
        for pattern in derivation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = match.strip()
                if len(cleaned) > 20 and self._contains_physics_content(cleaned):
                    concepts["derivations"].append(cleaned)
        
        # Extract physical scenarios described in the paper
        scenario_patterns = [
            r'(?:Consider|Suppose|Let|Assume)\s+([^.]*?(?:particle|system|field|wave|oscillator|potential)[^.]*?)(?:\.|\n)',
            r'(?:In this|Our|The)\s+([^.]*?(?:experiment|setup|system|model|approach)[^.]*?)(?:\.|\n)',
            r'(?:We study|We consider|We analyze)\s+([^.]*?)(?:\.|\n)',
        ]
        
        for pattern in scenario_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                cleaned = match.strip()
                if len(cleaned) > 30 and self._contains_physics_content(cleaned):
                    concepts["physical_scenarios"].append(cleaned)
        
        # Extract numerical values with context
        numerical_patterns = [
            r'([A-Za-z_]\w*\s*=\s*[0-9]+\.?[0-9]*(?:\s*×\s*10[⁻⁰-⁹]+)?\s*(?:m|kg|s|Hz|eV|K|Pa|N|J|W|V|A|T|rad)?\b)',
            r'((?:wavelength|frequency|energy|mass|velocity|temperature|pressure)\s*(?:of|=|is)\s*[0-9]+[^.\n]*)',
        ]
        
        for pattern in numerical_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if self._is_meaningful_numerical_value(match):
                    concepts["numerical_values"].append(match.strip())
        
        # Extract physics principles mentioned
        principles_patterns = [
            r'(conservation of \w+[^.]*)',
            r'(Newton\'?s? \w+ law[^.]*)',
            r'(Maxwell\'?s? equations?[^.]*)',
            r'(Schr[öo]dinger equation[^.]*)',
            r'(uncertainty principle[^.]*)',
            r'(thermodynamic \w+ law[^.]*)',
        ]
        
        for pattern in principles_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts["physics_principles"].extend([m.strip() for m in matches])
        
        return concepts if any(len(v) > 0 for v in concepts.values()) else None
    
    def _generate_problems_from_extracted_concepts(self, concepts: Dict, paper, assessment) -> List[Dict]:
        """Generate problems using ACTUAL extracted concepts from the paper"""
        
        problems = []
        
        # Problem Type 1: Mathematical Derivation using actual equations
        if concepts["equations"] or concepts["derivations"]:
            derivation_problem = self._create_derivation_from_actual_content(concepts, paper)
            if derivation_problem:
                problems.append(derivation_problem)
        
        # Problem Type 2: Physical Scenario Analysis using actual scenarios
        if concepts["physical_scenarios"]:
            scenario_problem = self._create_scenario_analysis_problem(concepts, paper)
            if scenario_problem:
                problems.append(scenario_problem)
        
        # Problem Type 3: Error Detection based on assessment issues
        if assessment.subtle_issues and concepts["equations"]:
            error_problem = self._create_error_detection_from_concepts(concepts, paper, assessment)
            if error_problem:
                problems.append(error_problem)
        
        # Problem Type 4: Numerical Calculation using actual values
        if concepts["numerical_values"]:
            numerical_problem = self._create_numerical_problem_from_paper(concepts, paper)
            if numerical_problem:
                problems.append(numerical_problem)
        
        return problems
    
    def _create_derivation_from_actual_content(self, concepts: Dict, paper) -> Optional[Dict]:
        """Create derivation problem using actual equations/derivations from the paper"""
        
        # Use actual equations from the paper
        primary_equation = concepts["equations"][0] if concepts["equations"] else None
        derivation_steps = concepts["derivations"][:3] if concepts["derivations"] else []
        
        if not primary_equation and not derivation_steps:
            return None
        
        # Create problem statement using actual content
        problem_statement = f"""Consider the physical system described by the following relationship:

{primary_equation if primary_equation else "Mathematical relationship from the given context"}

"""
        
        if derivation_steps:
            problem_statement += "The derivation proceeds through these steps:\n"
            for i, step in enumerate(derivation_steps, 1):
                problem_statement += f"{i}. {step}\n"
            problem_statement += "\n"
        
        problem_statement += """Analyze this derivation and:
1. Verify the mathematical consistency of each step
2. Check the physical reasoning behind the approach
3. Identify any assumptions or approximations made
4. Determine if the final result is dimensionally correct
5. Suggest alternative approaches if applicable

Provide a complete analysis of the mathematical and physical reasoning."""
        
        return {
            "problem_id": f"derivation_from_paper_{paper.id}_{hash(primary_equation or str(derivation_steps)) % 10000}",
            "type": "mathematical_derivation_analysis",
            "difficulty": "intermediate",
            "domain": self._classify_physics_domain(paper.subject),
            "problem_statement": problem_statement,
            "source_content": {
                "primary_equation": primary_equation,
                "derivation_steps": derivation_steps,
                "paper_context": f"From {paper.subject} research"
            },
            "evaluation_criteria": [
                "Mathematical rigor in step verification",
                "Understanding of physical principles",
                "Dimensional analysis accuracy",
                "Recognition of assumptions and limitations"
            ]
        }
    
    def _create_scenario_analysis_problem(self, concepts: Dict, paper) -> Optional[Dict]:
        """Create problem analyzing actual physical scenarios from the paper"""
        
        if not concepts["physical_scenarios"]:
            return None
        
        primary_scenario = concepts["physical_scenarios"][0]
        related_principles = concepts["physics_principles"][:2] if concepts["physics_principles"] else []
        
        problem_statement = f"""Consider the following physical scenario:

{primary_scenario}

"""
        
        if related_principles:
            problem_statement += "This system involves the following physics principles:\n"
            for principle in related_principles:
                problem_statement += f"• {principle}\n"
            problem_statement += "\n"
        
        problem_statement += """Analyze this physical system by addressing:

1. **System Setup**: What are the key physical quantities and their relationships?
2. **Governing Principles**: Which fundamental laws of physics apply to this system?
3. **Mathematical Model**: How would you set up equations to describe this system?
4. **Approximations**: What simplifying assumptions might be reasonable?
5. **Predictions**: What physical behavior would you expect to observe?
6. **Experimental Considerations**: How could this system be studied experimentally?

Provide a comprehensive physics analysis of this scenario."""
        
        return {
            "problem_id": f"scenario_analysis_{paper.id}_{hash(primary_scenario) % 10000}",
            "type": "physical_scenario_analysis",
            "difficulty": "intermediate",
            "domain": self._classify_physics_domain(paper.subject),
            "problem_statement": problem_statement,
            "source_content": {
                "primary_scenario": primary_scenario,
                "related_principles": related_principles
            },
            "evaluation_criteria": [
                "Understanding of physical setup",
                "Correct application of physics principles",
                "Mathematical modeling ability",
                "Recognition of approximations and limitations"
            ]
        }
    
    def _create_error_detection_from_concepts(self, concepts: Dict, paper, assessment) -> Optional[Dict]:
        """Create error detection problem using actual content and identified issues"""
        
        if not concepts["equations"] or not assessment.subtle_issues:
            return None
        
        primary_equation = concepts["equations"][0]
        key_issues = assessment.subtle_issues[:2]
        
        # Create a problem with potential errors based on the assessment
        problem_statement = f"""The following analysis presents a physical relationship and its derivation:

**Key Equation**: {primary_equation}

**Potential Issues Identified**:
"""
        
        for i, issue in enumerate(key_issues, 1):
            problem_statement += f"{i}. {issue}\n"
        
        problem_statement += f"""
**Assessment Context**: This work has been identified as having sophistication level {assessment.physics_sophistication:.2f} with recommendation: {assessment.stage_3_recommendation}

**Your Task**:
1. Analyze the equation and derivation for mathematical errors
2. Check for physics principle violations
3. Verify dimensional consistency
4. Identify any logical inconsistencies
5. Assess whether the identified issues are valid concerns
6. Propose corrections where necessary

Focus particularly on subtle errors that might not be immediately obvious but could invalidate the physics reasoning."""
        
        return {
            "problem_id": f"error_detection_{paper.id}_{hash(str(key_issues)) % 10000}",
            "type": "error_detection_analysis",
            "difficulty": "advanced",
            "domain": self._classify_physics_domain(paper.subject),
            "problem_statement": problem_statement,
            "source_content": {
                "primary_equation": primary_equation,
                "identified_issues": key_issues,
                "sophistication_level": assessment.physics_sophistication
            },
            "evaluation_criteria": [
                "Identification of mathematical errors",
                "Recognition of physics principle violations",
                "Dimensional analysis accuracy",
                "Critical evaluation of identified issues"
            ]
        }
    
    def _create_numerical_problem_from_paper(self, concepts: Dict, paper) -> Optional[Dict]:
        """Create numerical calculation problem using actual values from the paper"""
        
        if not concepts["numerical_values"]:
            return None
        
        numerical_data = concepts["numerical_values"][:3]  # Use up to 3 values
        related_equations = concepts["equations"][:2] if concepts["equations"] else []
        
        problem_statement = f"""Using the numerical data and relationships from a {paper.subject} analysis:

**Given Data**:
"""
        
        for value in numerical_data:
            problem_statement += f"• {value}\n"
        
        if related_equations:
            problem_statement += "\n**Relevant Relationships**:\n"
            for eq in related_equations:
                problem_statement += f"• {eq}\n"
        
        problem_statement += f"""
**Calculation Tasks**:
1. Verify that all given quantities have consistent units
2. Calculate any derived quantities that can be determined from the given data
3. Estimate the order of magnitude for key physical parameters
4. Determine which quantities might be measurable experimentally
5. Assess the physical reasonableness of the numerical values

**Analysis Requirements**:
- Show all unit conversions explicitly
- Provide uncertainty estimates where appropriate  
- Explain the physical significance of calculated results
- Compare values to typical scales in {paper.subject.lower()}"""
        
        return {
            "problem_id": f"numerical_analysis_{paper.id}_{hash(str(numerical_data)) % 10000}",
            "type": "numerical_analysis",
            "difficulty": "intermediate",
            "domain": self._classify_physics_domain(paper.subject),
            "problem_statement": problem_statement,
            "source_content": {
                "numerical_data": numerical_data,
                "related_equations": related_equations
            },
            "evaluation_criteria": [
                "Correct unit analysis and conversions",
                "Accurate numerical calculations",
                "Physical interpretation of results",
                "Order of magnitude reasoning"
            ]
        }
    
    def _is_meaningful_equation(self, equation: str) -> bool:
        """Check if equation contains meaningful physics content"""
        
        # Must have mathematical content
        if '=' not in equation:
            return False
        
        # Check for physics variables
        physics_vars = ['E', 'F', 'V', 'p', 'm', 'v', 'a', 'g', 'H', 'L', 'T', 'P', 'ρ', 'σ', 'ω', 'λ', 'μ', 'ε']
        has_physics_vars = any(var in equation for var in physics_vars)
        
        # Check for mathematical operators
        has_math_ops = any(op in equation for op in ['+', '-', '*', '/', '^', '∇', '∂', '∫'])
        
        # Must be substantial enough
        is_substantial = len(equation.strip()) > 8
        
        return has_physics_vars and has_math_ops and is_substantial
    
    def _contains_physics_content(self, text: str) -> bool:
        """Check if text contains actual physics content"""
        
        physics_terms = [
            'energy', 'force', 'field', 'particle', 'wave', 'mass', 'velocity',
            'momentum', 'charge', 'potential', 'frequency', 'wavelength',
            'temperature', 'pressure', 'density', 'current', 'voltage',
            'magnetic', 'electric', 'quantum', 'classical', 'relativistic'
        ]
        
        text_lower = text.lower()
        physics_count = sum(1 for term in physics_terms if term in text_lower)
        
        # Also check for mathematical content
        has_math = any(char in text for char in ['=', '+', '-', '*', '/', '(', ')', '^'])
        
        return physics_count >= 1 or has_math
    
    def _is_meaningful_numerical_value(self, value: str) -> bool:
        """Check if numerical value is meaningful for physics"""
        
        # Must contain actual numbers
        if not re.search(r'\d', value):
            return False
        
        # Check for physics units or contexts
        physics_contexts = [
            'm', 'kg', 's', 'Hz', 'eV', 'K', 'Pa', 'N', 'J', 'W', 'V', 'A', 'T',
            'wavelength', 'frequency', 'energy', 'mass', 'velocity', 'temperature'
        ]
        
        has_physics_context = any(context in value.lower() for context in physics_contexts)
        
        # Must be substantial
        is_substantial = len(value.strip()) > 5
        
        return has_physics_context and is_substantial
    
    def _classify_physics_domain(self, subject: str) -> str:
        """Classify physics domain from subject"""
        subject_lower = subject.lower()
        
        if any(term in subject_lower for term in ["classical mechanics", "mechanics"]):
            return "mechanics"
        elif any(term in subject_lower for term in ["electro", "magnetic", "field"]):
            return "electromagnetism"
        elif any(term in subject_lower for term in ["quantum", "atomic", "molecular"]):
            return "quantum"
        elif any(term in subject_lower for term in ["thermo", "statistical", "kinetic"]):
            return "thermodynamics"
        elif any(term in subject_lower for term in ["relativity", "gravity", "cosmol"]):
            return "relativity"
        elif any(term in subject_lower for term in ["optics", "photon", "light"]):
            return "optics"
        else:
            return "general_physics"
    
    def _assess_problem_difficulty(self, concepts: Dict) -> str:
        """Assess difficulty based on actual extracted concepts"""
        
        advanced_indicators = 0
        
        # Check for advanced mathematical content
        for equation in concepts.get("equations", []):
            if any(term in equation.lower() for term in ["tensor", "∇", "∂", "∫", "eigenvalue"]):
                advanced_indicators += 1
        
        # Check for advanced physics concepts
        for principle in concepts.get("physics_principles", []):
            if any(term in principle.lower() for term in ["quantum field", "relativity", "symmetry"]):
                advanced_indicators += 1
        
        if advanced_indicators >= 3:
            return "advanced"
        elif advanced_indicators >= 1:
            return "intermediate"
        else:
            return "introductory"
    
    def _is_suitable_for_benchmark(self, paper, full_text: str) -> bool:
        """Check if paper is suitable for benchmark creation"""
        
        if not full_text or len(full_text.strip()) < 800:
            return False
        
        # Must have substantial physics/mathematical content
        has_equations = '=' in full_text and len(re.findall(r'[A-Za-z]\s*=', full_text)) >= 2
        
        physics_indicators = [
            'energy', 'force', 'field', 'particle', 'wave', 'quantum',
            'equation', 'formula', 'derivation', 'calculation'
        ]
        
        text_sample = full_text[:2000].lower()
        physics_count = sum(1 for indicator in physics_indicators if indicator in text_sample)
        
        return has_equations or physics_count >= 4
    
    def _clean_equation(self, equation: str) -> str:
        """Clean and format equation"""
        # Remove extra whitespace and format
        cleaned = re.sub(r'\s+', ' ', equation.strip())
        # Remove common artifacts but keep mathematical symbols
        cleaned = re.sub(r'[^\w\s=+\-*/(){}\\.,∇∂∫α-ωΑ-Ω]', '', cleaned)
        return cleaned
    
    def save_benchmark(self, benchmark_items: List[Dict]) -> str:
        """Save self-contained benchmark to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"self_contained_physics_benchmark_{timestamp}.json"
        filepath = self.benchmark_dir / filename
        
        benchmark_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "benchmark_type": "self_contained_physics_reasoning_from_papers",
                "total_problems": sum(len(item.get("problems", [])) for item in benchmark_items),
                "total_sets": len(benchmark_items),
                "format_version": "2.0",
                "description": "Self-contained physics problems using actual concepts from research papers"
            },
            "problem_sets": benchmark_items
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Self-contained benchmark saved to {filepath}")
        return str(filepath)