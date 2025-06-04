"""
Enhanced RL Training Data Builder - Creates clear chain-of-thought training examples
Following DeepSeek-R1 and UGPhysics standards for RL training data
Key improvements: Clear problem statements + High-quality reasoning chains + Better language filtering
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class ChainOfThoughtTrainingBuilder:
    """Creates high-quality chain-of-thought training data for RL"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.training_dir = self.output_dir / "training_data"
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard format templates following DeepSeek-R1 style
        self.thinking_templates = self._load_thinking_templates()
        
    def create_rl_training_examples(self, paper, full_text: str) -> List[Dict[str, Any]]:
        """Create high-quality RL training examples with clear chain-of-thought"""
        
        if not self._is_suitable_for_training(paper, full_text):
            return []
        
        # Extract complete problem-solution pairs with high quality standards
        problem_solutions = self._extract_complete_problem_solutions(full_text)
        
        training_examples = []
        for problem_solution in problem_solutions[:3]:  # Limit to best examples
            
            # Create structured training example in RL format
            example = self._create_structured_rl_example(problem_solution, paper)
            if example:
                training_examples.append(example)
        
        return training_examples
    
    def _extract_complete_problem_solutions(self, text: str) -> List[Dict[str, Any]]:
        """Extract complete problem-solution pairs with clear structure"""
        
        solutions = []
        
        # Pattern 1: Explicit Problem/Solution structure (highest priority)
        explicit_patterns = [
            r'(?:Problem|Question|Exercise)\s*:?\s*(.*?)(?:Solution|Answer)\s*:?\s*(.*?)(?=(?:Problem|Question|Exercise|\Z))',
            r'(?:Find|Calculate|Determine|Show|Prove)\s+(.*?)(?:Solution|We have|We start|We begin)\s*:?\s*(.*?)(?=(?:Find|Calculate|Determine|\Z))',
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for problem_text, solution_text in matches:
                processed = self._process_problem_solution_pair(problem_text.strip(), solution_text.strip())
                if processed and self._is_high_quality_example(processed):
                    solutions.append(processed)
        
        # Pattern 2: Derivation structure (Given → Steps → Result)
        derivation_patterns = [
            r'(?:Given|Consider|Let)\s+(.*?)(?:To find|To derive|To show)\s+(.*?)(?:We have|We start|We begin|Step|First)\s+(.*?)(?:Therefore|Thus|Hence|Finally)\s+(.*?)(?=\n\n|\Z)',
        ]
        
        for pattern in derivation_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for given, goal, steps, result in matches:
                derivation = self._create_derivation_example(given.strip(), goal.strip(), steps.strip(), result.strip())
                if derivation and self._is_high_quality_example(derivation):
                    solutions.append(derivation)
        
        # Pattern 3: Step-by-step solutions with clear progression
        step_patterns = [
            r'(.*?)(?:Step\s+1|First)\s*:?\s*(.*?)(?:Step\s+2|Next|Then)\s*:?\s*(.*?)(?:Step\s+3|Finally|Therefore)\s*:?\s*(.*?)(?=\n\n|\Z)',
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for setup, step1, step2, step3 in matches:
                if len(setup.strip()) > 30:  # Has substantial problem setup
                    step_solution = self._create_step_by_step_example(setup.strip(), [step1.strip(), step2.strip(), step3.strip()])
                    if step_solution and self._is_high_quality_example(step_solution):
                        solutions.append(step_solution)
        
        return solutions[:5]  # Return best examples only
    
    def _process_problem_solution_pair(self, problem_text: str, solution_text: str) -> Optional[Dict[str, Any]]:
        """Process a problem-solution pair into structured format"""
        
        if len(problem_text) < 30 or len(solution_text) < 50:
            return None
        
        # Clean and structure the problem statement
        clean_problem = self._clean_problem_statement(problem_text)
        if not clean_problem:
            return None
        
        # Extract and structure solution steps
        solution_steps = self._extract_solution_steps(solution_text)
        if len(solution_steps) < 2:
            return None
        
        # Extract final answer if present
        final_answer = self._extract_final_answer(solution_text)
        
        return {
            "type": "problem_solution",
            "problem_statement": clean_problem,
            "solution_steps": solution_steps,
            "final_answer": final_answer,
            "reasoning_quality": self._assess_reasoning_quality(solution_steps)
        }
    
    def _create_derivation_example(self, given: str, goal: str, steps: str, result: str) -> Optional[Dict[str, Any]]:
        """Create structured derivation example"""
        
        if len(given) < 20 or len(steps) < 50:
            return None
        
        # Structure the derivation as a clear problem
        problem_statement = f"Given: {given}\nDerive: {goal}"
        
        # Parse derivation steps with better validation
        derivation_steps = self._parse_derivation_steps(steps)
        if len(derivation_steps) < 2:
            return None
        
        # Add final result as conclusion
        derivation_steps.append(f"Therefore: {result}")
        
        return {
            "type": "mathematical_derivation",
            "problem_statement": problem_statement,
            "solution_steps": derivation_steps,
            "final_answer": result,
            "reasoning_quality": self._assess_reasoning_quality(derivation_steps)
        }
    
    def _create_step_by_step_example(self, setup: str, steps: List[str]) -> Optional[Dict[str, Any]]:
        """Create step-by-step solution example"""
        
        valid_steps = [step for step in steps if len(step.strip()) > 15 and self._is_meaningful_step(step)]
        if len(valid_steps) < 2:
            return None
        
        return {
            "type": "step_by_step_solution",
            "problem_statement": setup,
            "solution_steps": valid_steps,
            "final_answer": valid_steps[-1] if valid_steps else "",
            "reasoning_quality": self._assess_reasoning_quality(valid_steps)
        }
    
    def _clean_problem_statement(self, problem_text: str) -> Optional[str]:
        """Clean and validate problem statement with strict standards"""
        
        # Remove artifacts and clean up
        cleaned = re.sub(r'[^\w\s\.,;:()=+\-*/\[\]{}\\]', '', problem_text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Must be substantial
        if len(cleaned) < 30:
            return None
        
        # Check for essential physics content
        physics_indicators = [
            'mass', 'energy', 'force', 'velocity', 'acceleration', 'momentum',
            'field', 'charge', 'current', 'voltage', 'potential', 'wave',
            'frequency', 'amplitude', 'temperature', 'pressure', 'volume',
            'particle', 'quantum', 'electron', 'proton', 'atom'
        ]
        
        has_physics = any(indicator in cleaned.lower() for indicator in physics_indicators)
        has_math = any(char in cleaned for char in ['=', '+', '-', '*', '/', '(', ')'])
        
        if not has_physics and not has_math:
            return None
        
        # Ensure it's actually a problem (not just a statement)
        problem_indicators = ['find', 'calculate', 'determine', 'derive', 'show', 'prove', 'what', 'how']
        is_problem = any(indicator in cleaned.lower() for indicator in problem_indicators)
        
        if not is_problem and '?' not in cleaned:
            return None
        
        # Check language quality (must be primarily English)
        if not self._check_english_quality(cleaned):
            return None
        
        return cleaned
    
    def _extract_solution_steps(self, solution_text: str) -> List[str]:
        """Extract clear solution steps from text with high quality standards"""
        
        steps = []
        
        # Pattern 1: Numbered steps (highest priority)
        numbered_steps = re.findall(r'(?:Step\s*)?(\d+)[.\)]\s*([^.]+\.)', solution_text, re.IGNORECASE)
        if numbered_steps:
            steps = [f"Step {num}: {step.strip()}" for num, step in numbered_steps if len(step.strip()) > 15]
        
        # Pattern 2: Sequential indicators
        if not steps:
            sequential_patterns = [
                r'(?:First|Initially|To start)\s*:?\s*([^.]+\.)',
                r'(?:Next|Then|Subsequently)\s*:?\s*([^.]+\.)',
                r'(?:Finally|Therefore|Thus|Hence)\s*:?\s*([^.]+\.)',
            ]
            
            for pattern in sequential_patterns:
                matches = re.findall(pattern, solution_text, re.IGNORECASE)
                steps.extend([match.strip() for match in matches if len(match.strip()) > 15])
        
        # Pattern 3: Physics reasoning steps
        if not steps:
            physics_steps = re.findall(
                r'(?:Using|From|By|Since|Given|We have|We get|We find|We obtain|This gives)\s+([^.]+\.)',
                solution_text, re.IGNORECASE
            )
            steps = [step.strip() for step in physics_steps if len(step.strip()) > 15 and self._is_meaningful_step(step)]
        
        # Pattern 4: Equation-based steps
        if not steps:
            equation_steps = re.findall(
                r'([^.]*[=][^.]*\.)',
                solution_text
            )
            steps = [step.strip() for step in equation_steps if len(step.strip()) > 15 and self._is_meaningful_step(step)]
        
        # Clean and validate steps
        cleaned_steps = []
        for step in steps:
            cleaned = self._clean_solution_step(step)
            if cleaned and self._is_meaningful_step(cleaned) and self._check_english_quality(cleaned):
                cleaned_steps.append(cleaned)
        
        return cleaned_steps[:8]  # Limit to reasonable number
    
    def _clean_solution_step(self, step: str) -> Optional[str]:
        """Clean individual solution step with strict quality control"""
        
        # Remove artifacts and normalize
        cleaned = re.sub(r'[^\w\s\.,;:()=+\-*/\[\]{}\\α-ωΑ-Ω]', '', step)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Must have minimum length and substance
        if len(cleaned) < 20:
            return None
        
        # Must contain meaningful content (math or physics)
        has_math = any(char in cleaned for char in ['=', '>', '<', '+', '-', '*', '/'])
        has_physics_words = any(word in cleaned.lower() for word in [
            'energy', 'force', 'mass', 'velocity', 'field', 'charge', 'potential',
            'equation', 'solve', 'calculate', 'substitute', 'derive'
        ])
        
        if not has_math and not has_physics_words:
            return None
        
        # Must be a complete thought
        if not cleaned.endswith('.') and '=' not in cleaned:
            cleaned += '.'
        
        return cleaned
    
    def _is_meaningful_step(self, step: str) -> bool:
        """Check if step contains meaningful physics/math content with strict standards"""
        
        step_lower = step.lower()
        
        # Physics/math content indicators
        content_indicators = [
            'equation', 'formula', 'energy', 'force', 'mass', 'velocity',
            'calculate', 'solve', 'substitute', 'derive', 'obtain',
            'therefore', 'hence', 'thus', 'using', 'from', 'given',
            'field', 'charge', 'potential', 'momentum', 'acceleration'
        ]
        
        has_content = any(indicator in step_lower for indicator in content_indicators)
        has_math = any(char in step for char in ['=', '+', '-', '*', '/', '(', ')'])
        
        # Avoid meaningless fragments
        avoid_terms = ['figure', 'table', 'page', 'section', 'reference', 'see', 'above', 'below', 'paper', 'author']
        has_avoid = any(term in step_lower for term in avoid_terms)
        
        # Must have reasonable word count
        word_count = len(step.split())
        
        return (has_content or has_math) and not has_avoid and word_count >= 5
    
    def _check_english_quality(self, text: str) -> bool:
        """Check if text is high-quality English with strict standards"""
        
        # Check for minimum English words
        english_words = [
            'the', 'and', 'of', 'to', 'a', 'in', 'that', 'is', 'for', 'with',
            'we', 'have', 'can', 'will', 'this', 'from', 'by', 'at', 'are',
            'equation', 'energy', 'force', 'field', 'particle', 'wave'
        ]
        
        text_lower = text.lower()
        english_count = sum(1 for word in english_words if f' {word} ' in f' {text_lower} ')
        
        # Must have substantial English content
        word_count = len(text.split())
        english_ratio = english_count / max(word_count, 1)
        
        # Strict English requirement
        if english_ratio < 0.3 or english_count < 5:
            return False
        
        # Check for physics/technical terms (indicates proper technical content)
        technical_terms = [
            'equation', 'energy', 'force', 'mass', 'velocity', 'field',
            'calculate', 'solve', 'derive', 'find', 'given', 'therefore',
            'momentum', 'acceleration', 'potential', 'charge'
        ]
        
        technical_count = sum(1 for term in technical_terms if term in text_lower)
        
        # Must have some technical content
        return technical_count >= 2
    
    def _parse_derivation_steps(self, steps_text: str) -> List[str]:
        """Parse mathematical derivation into clear steps"""
        
        # Look for mathematical progression with equations
        math_steps = re.findall(
            r'([^.]*(?:=|→|⇒|∴)[^.]*\.?)',
            steps_text
        )
        
        if math_steps:
            valid_steps = []
            for step in math_steps:
                cleaned = step.strip()
                if len(cleaned) > 15 and self._is_meaningful_step(cleaned):
                    if not cleaned.endswith('.'):
                        cleaned += '.'
                    valid_steps.append(cleaned)
            return valid_steps
        
        # Fallback to sentence-based parsing
        sentences = re.split(r'[.!?]+', steps_text)
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 20 and self._is_meaningful_step(sentence + '.'):
                meaningful_sentences.append(sentence + '.')
        
        return meaningful_sentences
    
    def _extract_final_answer(self, solution_text: str) -> str:
        """Extract the final answer from solution"""
        
        # Look for explicit answer indicators
        answer_patterns = [
            r'(?:Answer|Result|Therefore|Thus|Hence|Finally)\s*:?\s*([^.]+)',
            r'([^.]*=\s*[^.]*\s*(?:units?|m|kg|s|N|J|W|V|A|Hz|Pa).*?)(?:\.|$)',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, solution_text, re.IGNORECASE)
            if matches:
                answer = matches[-1].strip()  # Take last (most likely final) answer
                if len(answer) > 5 and self._check_english_quality(answer):
                    return answer
        
        # If no explicit answer found, try to extract from last meaningful step
        sentences = solution_text.split('.')
        for sentence in reversed(sentences):
            if '=' in sentence and len(sentence.strip()) > 10:
                return sentence.strip()
        
        return ""
    
    def _assess_reasoning_quality(self, steps: List[str]) -> float:
        """Assess the quality of reasoning in solution steps with strict standards"""
        
        if not steps:
            return 0.0
        
        quality_score = 0.0
        
        # Check for logical progression (0.3 weight)
        has_logical_flow = self._check_logical_flow(steps)
        if has_logical_flow:
            quality_score += 0.3
        
        # Check for mathematical rigor (0.3 weight)
        math_rigor = self._check_mathematical_rigor(steps)
        quality_score += math_rigor * 0.3
        
        # Check for physics content (0.2 weight)
        physics_content = self._check_physics_content(steps)
        quality_score += physics_content * 0.2
        
        # Check for clarity and completeness (0.2 weight)
        clarity = self._check_clarity(steps)
        quality_score += clarity * 0.2
        
        return min(1.0, quality_score)
    
    def _check_logical_flow(self, steps: List[str]) -> bool:
        """Check if steps follow logical progression"""
        
        # Look for progression indicators
        start_indicators = ['given', 'start', 'first', 'initial', 'consider']
        middle_indicators = ['then', 'next', 'using', 'from', 'substitute', 'applying']
        end_indicators = ['therefore', 'thus', 'finally', 'hence', 'result']
        
        steps_text = ' '.join(steps).lower()
        
        has_start = any(indicator in steps_text for indicator in start_indicators)
        has_middle = any(indicator in steps_text for indicator in middle_indicators)
        has_end = any(indicator in steps_text for indicator in end_indicators)
        
        return has_start and has_middle and has_end
    
    def _check_mathematical_rigor(self, steps: List[str]) -> float:
        """Check mathematical rigor (0-1 score)"""
        
        total_steps = len(steps)
        if total_steps == 0:
            return 0.0
        
        math_steps = sum(1 for step in steps if any(char in step for char in ['=', '+', '-', '*', '/', '(', ')']))
        equation_steps = sum(1 for step in steps if '=' in step)
        
        math_ratio = math_steps / total_steps
        equation_ratio = equation_steps / total_steps
        
        return min(1.0, (math_ratio + equation_ratio) / 2)
    
    def _check_physics_content(self, steps: List[str]) -> float:
        """Check physics content quality (0-1 score)"""
        
        physics_terms = [
            'energy', 'force', 'momentum', 'mass', 'velocity', 'acceleration',
            'field', 'potential', 'charge', 'current', 'frequency', 'wavelength',
            'conservation', 'newton', 'maxwell', 'schrodinger', 'einstein',
            'pressure', 'temperature', 'volume', 'entropy', 'quantum'
        ]
        
        total_steps = len(steps)
        if total_steps == 0:
            return 0.0
        
        physics_steps = sum(1 for step in steps 
                          if any(term in step.lower() for term in physics_terms))
        
        return min(1.0, physics_steps / total_steps)
    
    def _check_clarity(self, steps: List[str]) -> float:
        """Check clarity and completeness (0-1 score)"""
        
        # Check average step length (should be substantial but not too long)
        avg_length = sum(len(step) for step in steps) / len(steps) if steps else 0
        length_score = 1.0 if 40 <= avg_length <= 150 else 0.5
        
        # Check for complete sentences
        complete_sentences = sum(1 for step in steps if step.strip().endswith('.'))
        completeness_score = complete_sentences / len(steps) if steps else 0
        
        # Check for English quality
        english_quality = sum(1 for step in steps if self._check_english_quality(step))
        english_score = english_quality / len(steps) if steps else 0
        
        return (length_score + completeness_score + english_score) / 3
    
    def _is_high_quality_example(self, example: Dict[str, Any]) -> bool:
        """Check if example meets strict quality standards"""
        
        # Must have clear problem statement
        if len(example.get("problem_statement", "")) < 40:
            return False
        
        # Must have multiple solution steps
        if len(example.get("solution_steps", [])) < 2:
            return False
        
        # Must have high quality score
        if example.get("reasoning_quality", 0) < 0.5:  # Raised threshold
            return False
        
        # Check language quality for both problem and solution
        problem_text = example["problem_statement"]
        solution_text = " ".join(example["solution_steps"])
        
        if not self._check_english_quality(problem_text) or not self._check_english_quality(solution_text):
            return False
        
        # Check for substantial physics/math content
        full_text = problem_text + " " + solution_text
        if not self._has_substantial_content(full_text):
            return False
        
        return True
    
    def _has_substantial_content(self, text: str) -> bool:
        """Check if text has substantial physics/mathematical content"""
        
        # Must have equations or calculations
        has_equations = '=' in text and any(char in text for char in ['+', '-', '*', '/'])
        
        # Must have physics concepts
        physics_concepts = [
            'energy', 'force', 'field', 'mass', 'velocity', 'momentum',
            'charge', 'potential', 'current', 'wave', 'frequency',
            'temperature', 'pressure', 'entropy', 'quantum'
        ]
        
        text_lower = text.lower()
        physics_count = sum(1 for concept in physics_concepts if concept in text_lower)
        
        return has_equations or physics_count >= 3
    
    def _create_structured_rl_example(self, problem_solution: Dict[str, Any], paper) -> Optional[Dict[str, Any]]:
        """Create final structured training example in RL format (DeepSeek-R1 style)"""
        
        if not self._is_high_quality_example(problem_solution):
            return None
        
        # Create thinking section following DeepSeek-R1 format
        thinking_section = self._create_thinking_section(problem_solution)
        
        # Create final answer section
        answer_section = problem_solution.get("final_answer", "").strip()
        if not answer_section:
            # Extract from last step if no explicit answer
            last_step = problem_solution["solution_steps"][-1] if problem_solution["solution_steps"] else ""
            answer_section = self._extract_answer_from_step(last_step)
        
        training_example = {
            "id": f"rl_training_{hash(str(problem_solution)) % 100000}",
            "source_metadata": {
                "paper_id": paper.id,
                "subject": paper.subject,
                "extraction_method": "enhanced_chain_of_thought_v2"
            },
            "prompt": problem_solution["problem_statement"],
            "completion": f"<think>\n{thinking_section}\n</think>\n\n{answer_section}",
            "metadata": {
                "type": problem_solution["type"],
                "reasoning_quality": problem_solution["reasoning_quality"],
                "step_count": len(problem_solution["solution_steps"]),
                "difficulty": self._assess_difficulty(problem_solution),
                "format_version": "rl_v2.0",
                "language_quality": "high_english"
            },
            "created_at": datetime.now().isoformat()
        }
        
        return training_example
    
    def _create_thinking_section(self, problem_solution: Dict[str, Any]) -> str:
        """Create structured thinking section for RL training (DeepSeek-R1 style)"""
        
        thinking_parts = []
        
        # Start with problem analysis
        thinking_parts.append("Let me analyze this physics problem step by step.")
        thinking_parts.append("")
        
        # Add solution steps with proper reasoning flow
        for i, step in enumerate(problem_solution["solution_steps"], 1):
            # Format step with reasoning context
            if i == 1:
                thinking_parts.append(f"First, {step}")
            elif i == len(problem_solution["solution_steps"]):
                thinking_parts.append(f"Finally, {step}")
            else:
                thinking_parts.append(f"Next, {step}")
            thinking_parts.append("")
        
        # Add verification if quality is high
        if problem_solution.get("reasoning_quality", 0) > 0.7:
            thinking_parts.append("Let me verify this result makes physical sense...")
            thinking_parts.append("The units are consistent and the magnitude is reasonable.")
            thinking_parts.append("")
        
        return "\n".join(thinking_parts)
    
    def _extract_answer_from_step(self, step: str) -> str:
        """Extract clean answer from a solution step"""
        
        # Look for equations or numerical results
        if '=' in step:
            parts = step.split('=')
            if len(parts) >= 2:
                return f"The answer is: {parts[-1].strip()}"
        
        # Look for conclusion statements
        conclusion_indicators = ['therefore', 'thus', 'hence', 'so', 'result']
        step_lower = step.lower()
        
        for indicator in conclusion_indicators:
            if indicator in step_lower:
                return f"Therefore: {step.strip()}"
        
        return f"The result is: {step.strip()}"
    
    def _assess_difficulty(self, problem_solution: Dict[str, Any]) -> str:
        """Assess difficulty level of the problem"""
        
        text = problem_solution["problem_statement"] + " " + " ".join(problem_solution["solution_steps"])
        text_lower = text.lower()
        
        # Advanced indicators
        advanced_terms = [
            'differential', 'integral', 'eigenvalue', 'tensor', 'lagrangian',
            'hamiltonian', 'fourier', 'perturbation', 'quantum field',
            'relativistic', 'schrodinger', 'maxwell equations'
        ]
        
        # Intermediate indicators
        intermediate_terms = [
            'derivative', 'vector', 'matrix', 'complex', 'wave equation',
            'conservation', 'momentum', 'energy conservation', 'field theory'
        ]
        
        advanced_count = sum(1 for term in advanced_terms if term in text_lower)
        intermediate_count = sum(1 for term in intermediate_terms if term in text_lower)
        
        if advanced_count >= 2:
            return "advanced"
        elif intermediate_count >= 2 or advanced_count >= 1:
            return "intermediate"
        else:
            return "introductory"
    
    def _is_suitable_for_training(self, paper, full_text: str) -> bool:
        """Check if paper is suitable for RL training data extraction with strict standards"""
        
        if not full_text or len(full_text.strip()) < 1500:  # Increased minimum
            return False
        
        # Check for high-quality English content
        if not self._check_english_quality(full_text[:1000]):
            return False
        
        # Check for problem-solution structure
        structure_indicators = [
            'problem', 'solution', 'find', 'calculate', 'derive', 'show',
            'step', 'first', 'next', 'therefore', 'thus', 'hence'
        ]
        
        text_sample = full_text[:2000].lower()
        structure_count = sum(1 for indicator in structure_indicators if indicator in text_sample)
        
        # Check for substantial physics content
        physics_indicators = [
            'energy', 'force', 'field', 'particle', 'wave', 'mass',
            'velocity', 'momentum', 'charge', 'potential', 'equation'
        ]
        
        physics_count = sum(1 for indicator in physics_indicators if indicator in text_sample)
        
        return structure_count >= 6 and physics_count >= 4
    
    def _load_thinking_templates(self) -> Dict[str, str]:
        """Load templates for thinking sections"""
        return {
            "analysis_start": "Let me analyze this physics problem step by step.",
            "verification": "Let me verify this result makes physical sense...",
            "conclusion": "Therefore, the final answer is:"
        }
    
    def save_training_data(self, training_examples: List[Dict]) -> str:
        """Save RL training data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rl_chain_of_thought_training_{timestamp}.json"
        filepath = self.training_dir / filename
        
        training_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "format_type": "chain_of_thought_rl_training",
                "total_examples": len(training_examples),
                "format_version": "2.0",
                "description": "High-quality chain-of-thought training data for physics reasoning RL",
                "quality_standards": "Strict English and physics content requirements",
                "difficulty_distribution": self._get_difficulty_distribution(training_examples),
                "quality_stats": self._get_quality_statistics(training_examples)
            },
            "training_examples": training_examples
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"High-quality RL training data saved to {filepath}")
        return str(filepath)
    
    def _get_difficulty_distribution(self, examples: List[Dict]) -> Dict[str, int]:
        """Get distribution of difficulty levels"""
        distribution = {}
        for example in examples:
            difficulty = example["metadata"]["difficulty"]
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution
    
    def _get_quality_statistics(self, examples: List[Dict]) -> Dict[str, float]:
        """Get quality statistics for the dataset"""
        if not examples:
            return {}
        
        qualities = [ex["metadata"]["reasoning_quality"] for ex in examples]
        step_counts = [ex["metadata"]["step_count"] for ex in examples]
        
        return {
            "avg_reasoning_quality": sum(qualities) / len(qualities),
            "min_reasoning_quality": min(qualities),
            "max_reasoning_quality": max(qualities),
            "avg_step_count": sum(step_counts) / len(step_counts),
            "total_high_quality": sum(1 for q in qualities if q > 0.7),
            "total_very_high_quality": sum(1 for q in qualities if q > 0.8)
        }