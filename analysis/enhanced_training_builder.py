"""
Chain of Thought Training Builder - Creates high-quality RL training data
Following DeepSeek-R1 standards for reasoning chain training
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class ChainOfThoughtTrainingBuilder:
    """Creates high-quality chain-of-thought training data for RL from actual paper content"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.training_dir = self.output_dir / "training_data"
        self.training_dir.mkdir(parents=True, exist_ok=True)
        
    def create_rl_training_examples(self, paper, full_text: str) -> List[Dict[str, Any]]:
        """Create high-quality RL training examples using actual paper content"""
        
        if not self._is_suitable_for_training(paper, full_text):
            return []
        
        # Extract actual problem-solution pairs from the paper
        extracted_examples = self._extract_actual_problem_solutions(full_text, paper)
        
        training_examples = []
        for example in extracted_examples[:3]:  # Limit to best examples
            
            # Create structured RL training example
            rl_example = self._create_rl_training_format(example, paper)
            if rl_example and self._meets_quality_standards(rl_example):
                training_examples.append(rl_example)
        
        return training_examples
    
    def _extract_actual_problem_solutions(self, text: str, paper) -> List[Dict[str, Any]]:
        """Extract actual problem-solution pairs from paper text"""
        
        examples = []
        
        # Pattern 1: Explicit Problem/Solution structure
        explicit_patterns = [
            r'(?:Problem|Question|Example)\s*:?\s*(.*?)(?:Solution|Answer|Result)\s*:?\s*(.*?)(?=(?:Problem|Question|Example|\n\n|\Z))',
            r'(?:Find|Calculate|Determine|Show|Prove)\s+(.*?)(?:\.|\n)\s*(?:Solution|We have|We start|We get)\s*:?\s*(.*?)(?=(?:Find|Calculate|\n\n|\Z))',
        ]
        
        for pattern in explicit_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for problem_text, solution_text in matches:
                example = self._process_explicit_example(problem_text.strip(), solution_text.strip(), paper)
                if example:
                    examples.append(example)
        
        # Pattern 2: Derivation sequences (Given → Steps → Result)
        derivation_patterns = [
            r'(?:Given|Starting with|Consider)\s+(.*?)(?:\.|\n)\s*(?:We derive|We show|We find)\s+(.*?)(?:Therefore|Thus|Hence)\s+(.*?)(?=\n\n|\Z)',
            r'(?:From|Using)\s+([^.]*equation[^.]*)\s*(?:\.|\n)\s*(.*?)(?:we get|we obtain|this gives)\s+(.*?)(?=\n\n|\Z)',
        ]
        
        for pattern in derivation_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for given, steps, result in matches:
                example = self._process_derivation_example(given.strip(), steps.strip(), result.strip(), paper)
                if example:
                    examples.append(example)
        
        # Pattern 3: Calculation sequences with numerical results
        calculation_patterns = [
            r'(?:Calculate|Computing|To find)\s+(.*?)(?:\.|\n)\s*(.*?)(?:=\s*[0-9][^.\n]*)',
            r'(?:The value of|We have)\s+([^=]*=\s*[^.\n]*)\s*(.*?)(?:Therefore|Thus)\s*(.*?)(?=\n\n|\Z)',
        ]
        
        for pattern in calculation_patterns:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for setup, calculation, result in matches:
                example = self._process_calculation_example(setup.strip(), calculation.strip(), result.strip(), paper)
                if example:
                    examples.append(example)
        
        return examples[:5]  # Return best examples
    
    def _process_explicit_example(self, problem_text: str, solution_text: str, paper) -> Optional[Dict[str, Any]]:
        """Process explicit problem-solution pair"""
        
        if len(problem_text) < 25 or len(solution_text) < 40:
            return None
        
        # Clean and validate content
        clean_problem = self._clean_and_validate_text(problem_text)
        clean_solution = self._clean_and_validate_text(solution_text)
        
        if not clean_problem or not clean_solution:
            return None
        
        # Extract reasoning steps from solution
        reasoning_steps = self._extract_reasoning_steps(clean_solution)
        if len(reasoning_steps) < 2:
            return None
        
        return {
            "type": "explicit_problem_solution",
            "problem_statement": clean_problem,
            "solution_steps": reasoning_steps,
            "final_answer": self._extract_final_answer(clean_solution),
            "source_context": f"From {paper.subject} paper: {paper.title[:60]}...",
            "quality_score": self._assess_quality(clean_problem, reasoning_steps)
        }
    
    def _process_derivation_example(self, given: str, steps: str, result: str, paper) -> Optional[Dict[str, Any]]:
        """Process derivation example"""
        
        if len(given) < 15 or len(steps) < 30:
            return None
        
        # Create structured problem statement
        problem_statement = f"Given: {given.strip()}\nDerive the following result and show all steps."
        
        # Parse derivation steps
        derivation_steps = self._parse_derivation_steps(steps, result)
        if len(derivation_steps) < 2:
            return None
        
        return {
            "type": "mathematical_derivation",
            "problem_statement": problem_statement,
            "solution_steps": derivation_steps,
            "final_answer": result.strip(),
            "source_context": f"From {paper.subject} derivation",
            "quality_score": self._assess_quality(problem_statement, derivation_steps)
        }
    
    def _process_calculation_example(self, setup: str, calculation: str, result: str, paper) -> Optional[Dict[str, Any]]:
        """Process numerical calculation example"""
        
        if len(setup) < 15 or len(calculation) < 20:
            return None
        
        # Create problem statement
        problem_statement = f"Calculate: {setup.strip()}"
        
        # Create calculation steps
        calc_steps = [calculation.strip()]
        if result.strip():
            calc_steps.append(f"Therefore: {result.strip()}")
        
        return {
            "type": "numerical_calculation",
            "problem_statement": problem_statement,
            "solution_steps": calc_steps,
            "final_answer": result.strip() if result.strip() else calc_steps[-1],
            "source_context": f"From {paper.subject} numerical analysis",
            "quality_score": self._assess_quality(problem_statement, calc_steps)
        }
    
    def _clean_and_validate_text(self, text: str) -> Optional[str]:
        """Clean and validate text for training quality"""
        
        if not text:
            return None
        
        # Remove artifacts and normalize
        cleaned = re.sub(r'[^\w\s\.,;:()=+\-*/\[\]{}\\α-ωΑ-Ω]', '', text)
        cleaned = re.sub(r'\s+', ' ', cleaned).strip()
        
        # Must be substantial
        if len(cleaned) < 15:
            return None
        
        # Must have physics/math content
        has_physics = any(term in cleaned.lower() for term in [
            'energy', 'force', 'mass', 'velocity', 'field', 'charge', 'potential',
            'particle', 'wave', 'frequency', 'momentum', 'acceleration'
        ])
        
        has_math = any(char in cleaned for char in ['=', '+', '-', '*', '/', '(', ')'])
        
        if not has_physics and not has_math:
            return None
        
        # Check language quality (primarily English)
        english_words = ['the', 'and', 'of', 'to', 'a', 'in', 'we', 'is', 'this', 'that']
        english_count = sum(1 for word in english_words if word in cleaned.lower())
        
        if english_count < 2:
            return None
        
        return cleaned
    
    def _extract_reasoning_steps(self, solution_text: str) -> List[str]:
        """Extract reasoning steps from solution text"""
        
        steps = []
        
        # Pattern 1: Numbered or sequential steps
        step_patterns = [
            r'(?:Step\s*\d+|First|Next|Then|Finally)\s*:?\s*([^.]+\.)',
            r'(?:We have|We get|We find|We obtain|We use)\s*:?\s*([^.]+\.)',
            r'(?:From|Using|By|Since)\s+([^.]+\.)',
            r'(?:Therefore|Thus|Hence)\s+([^.]+\.)',
        ]
        
        for pattern in step_patterns:
            matches = re.findall(pattern, solution_text, re.IGNORECASE)
            for match in matches:
                step = match.strip()
                if len(step) > 10 and self._is_meaningful_step(step):
                    steps.append(step)
        
        # Pattern 2: Equation-based steps
        if not steps:
            equation_steps = re.findall(r'([^.]*=\s*[^.]*\.)', solution_text)
            for step in equation_steps:
                if len(step.strip()) > 10 and '=' in step:
                    steps.append(step.strip())
        
        # Pattern 3: Sentence-based extraction as fallback
        if not steps:
            sentences = re.split(r'[.!?]+', solution_text)
            for sentence in sentences:
                sentence = sentence.strip()
                if len(sentence) > 15 and self._is_meaningful_step(sentence):
                    steps.append(sentence + '.')
        
        return steps[:6]  # Limit to reasonable number
    
    def _parse_derivation_steps(self, steps_text: str, result: str) -> List[str]:
        """Parse derivation steps into clear sequence"""
        
        # Look for mathematical progression
        math_steps = re.findall(r'([^.]*(?:=|→|⇒)[^.]*)', steps_text)
        
        if math_steps:
            valid_steps = []
            for step in math_steps:
                cleaned = step.strip()
                if len(cleaned) > 10 and self._is_meaningful_step(cleaned):
                    if not cleaned.endswith('.'):
                        cleaned += '.'
                    valid_steps.append(cleaned)
            
            # Add result as final step
            if result and not any(result.strip() in step for step in valid_steps):
                valid_steps.append(f"Therefore: {result.strip()}")
            
            return valid_steps
        
        # Fallback to sentence parsing
        sentences = re.split(r'[.!?]+', steps_text)
        meaningful_sentences = []
        
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 15 and self._is_meaningful_step(sentence):
                meaningful_sentences.append(sentence + '.')
        
        if result:
            meaningful_sentences.append(f"Therefore: {result.strip()}")
        
        return meaningful_sentences
    
    def _is_meaningful_step(self, step: str) -> bool:
        """Check if step contains meaningful content"""
        
        step_lower = step.lower()
        
        # Physics/math indicators
        meaningful_indicators = [
            'equation', 'energy', 'force', 'mass', 'velocity', 'field',
            'calculate', 'derive', 'solve', 'substitute', 'obtain',
            'therefore', 'using', 'from', 'given', 'hence', 'thus'
        ]
        
        has_meaningful = any(indicator in step_lower for indicator in meaningful_indicators)
        has_math = any(char in step for char in ['=', '+', '-', '*', '/', '(', ')'])
        
        # Avoid meaningless content
        avoid_terms = ['figure', 'table', 'page', 'section', 'paper', 'author', 'reference']
        has_avoid = any(term in step_lower for term in avoid_terms)
        
        # Must have reasonable word count
        word_count = len(step.split())
        
        return (has_meaningful or has_math) and not has_avoid and word_count >= 4
    
    def _extract_final_answer(self, solution_text: str) -> str:
        """Extract final answer from solution"""
        
        # Look for explicit answer indicators
        answer_patterns = [
            r'(?:Answer|Result|Therefore|Thus|Hence|Finally)\s*:?\s*([^.]+)',
            r'([^.]*=\s*[0-9][^.]*(?:m|kg|s|Hz|eV|K|Pa|N|J|W|V|A|T)?[^.]*)',
        ]
        
        for pattern in answer_patterns:
            matches = re.findall(pattern, solution_text, re.IGNORECASE)
            if matches:
                answer = matches[-1].strip()
                if len(answer) > 3:
                    return answer
        
        # Extract from last meaningful sentence
        sentences = solution_text.split('.')
        for sentence in reversed(sentences):
            if '=' in sentence and len(sentence.strip()) > 8:
                return sentence.strip()
        
        return "Result derived as shown above"
    
    def _assess_quality(self, problem: str, steps: List[str]) -> float:
        """Assess quality of the training example"""
        
        if not steps:
            return 0.0
        
        quality_score = 0.0
        
        # Check problem clarity (0.3 weight)
        problem_score = 0.3 if len(problem) > 20 and any(word in problem.lower() for word in ['find', 'calculate', 'derive', 'show']) else 0.1
        quality_score += problem_score
        
        # Check step quality (0.4 weight)
        good_steps = sum(1 for step in steps if self._is_high_quality_step(step))
        step_quality = (good_steps / len(steps)) * 0.4
        quality_score += step_quality
        
        # Check mathematical content (0.2 weight)
        math_content = sum(1 for step in steps if '=' in step) / len(steps) * 0.2
        quality_score += math_content
        
        # Check logical flow (0.1 weight)
        has_progression = any(word in ' '.join(steps).lower() for word in ['therefore', 'thus', 'hence', 'so'])
        flow_score = 0.1 if has_progression else 0.05
        quality_score += flow_score
        
        return min(1.0, quality_score)
    
    def _is_high_quality_step(self, step: str) -> bool:
        """Check if step is high quality"""
        
        # Must be substantial
        if len(step) < 15:
            return False
        
        # Should contain meaningful physics/math
        step_lower = step.lower()
        quality_indicators = [
            'equation', 'energy', 'force', 'velocity', 'field', 'mass',
            'substituting', 'using', 'from', 'therefore', 'hence', 'thus',
            'calculate', 'derive', 'solve', 'obtain', 'find'
        ]
        
        has_quality = any(indicator in step_lower for indicator in quality_indicators)
        has_math = '=' in step or any(op in step for op in ['+', '-', '*', '/', '(', ')'])
        
        return has_quality or has_math
    
    def _create_rl_training_format(self, example: Dict[str, Any], paper) -> Optional[Dict[str, Any]]:
        """Create training example in RL format (DeepSeek-R1 style)"""
        
        # Create thinking section
        thinking_section = self._create_thinking_section(example)
        
        # Create answer section
        answer_section = example.get("final_answer", "").strip()
        if not answer_section:
            answer_section = "The solution follows from the steps shown above."
        
        return {
            "id": f"rl_physics_{paper.id}_{hash(str(example)) % 100000}",
            "source_metadata": {
                "paper_id": paper.id,
                "paper_title": paper.title,
                "paper_subject": paper.subject,
                "extraction_type": example["type"]
            },
            "prompt": example["problem_statement"],
            "completion": f"<think>\n{thinking_section}\n</think>\n\n{answer_section}",
            "metadata": {
                "reasoning_quality": example["quality_score"],
                "step_count": len(example["solution_steps"]),
                "difficulty": self._assess_difficulty(example),
                "format_version": "rl_physics_v2.0",
                "source_context": example.get("source_context", "")
            },
            "created_at": datetime.now().isoformat()
        }
    
    def _create_thinking_section(self, example: Dict[str, Any]) -> str:
        """Create thinking section following DeepSeek-R1 format"""
        
        thinking_parts = []
        
        # Start with problem analysis
        thinking_parts.append("Let me work through this physics problem step by step.")
        thinking_parts.append("")
        
        # Add solution steps with reasoning context
        for i, step in enumerate(example["solution_steps"], 1):
            if i == 1:
                thinking_parts.append(f"First, {step}")
            elif i == len(example["solution_steps"]):
                thinking_parts.append(f"Finally, {step}")
            else:
                thinking_parts.append(f"Next, {step}")
            thinking_parts.append("")
        
        # Add verification for high quality examples
        if example.get("quality_score", 0) > 0.7:
            thinking_parts.append("Let me verify this makes physical sense...")
            thinking_parts.append("The approach is consistent with known physics principles.")
            thinking_parts.append("")
        
        return "\n".join(thinking_parts)
    
    def _assess_difficulty(self, example: Dict[str, Any]) -> str:
        """Assess difficulty level"""
        
        content = example["problem_statement"] + " " + " ".join(example["solution_steps"])
        content_lower = content.lower()
        
        # Advanced indicators
        advanced_terms = [
            'differential', 'integral', 'eigenvalue', 'tensor', 'lagrangian',
            'quantum field', 'relativistic', 'perturbation'
        ]
        
        # Intermediate indicators
        intermediate_terms = [
            'derivative', 'vector', 'matrix', 'conservation', 'momentum',
            'electromagnetic', 'thermodynamic', 'wave equation'
        ]
        
        advanced_count = sum(1 for term in advanced_terms if term in content_lower)
        intermediate_count = sum(1 for term in intermediate_terms if term in content_lower)
        
        if advanced_count >= 2:
            return "advanced"
        elif intermediate_count >= 2 or advanced_count >= 1:
            return "intermediate"
        else:
            return "introductory"
    
    def _meets_quality_standards(self, rl_example: Dict[str, Any]) -> bool:
        """Check if example meets quality standards"""
        
        # Must have reasonable quality score
        if rl_example["metadata"]["reasoning_quality"] < 0.4:
            return False
        
        # Must have multiple reasoning steps
        if rl_example["metadata"]["step_count"] < 2:
            return False
        
        # Must have substantial content
        prompt_length = len(rl_example["prompt"])
        completion_length = len(rl_example["completion"])
        
        if prompt_length < 30 or completion_length < 100:
            return False
        
        return True
    
    def _is_suitable_for_training(self, paper, full_text: str) -> bool:
        """Check if paper is suitable for training data extraction"""
        
        if not full_text or len(full_text.strip()) < 1000:
            return False
        
        # Must have problem-solving content
        problem_indicators = [
            'problem', 'solution', 'find', 'calculate', 'derive', 'show',
            'example', 'exercise', 'question', 'answer'
        ]
        
        # Must have reasoning content
        reasoning_indicators = [
            'step', 'first', 'next', 'then', 'therefore', 'thus', 'hence',
            'using', 'from', 'given', 'we have', 'we get'
        ]
        
        text_sample = full_text[:2000].lower()
        problem_count = sum(1 for indicator in problem_indicators if indicator in text_sample)
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in text_sample)
        
        # Must have substantial physics content
        physics_count = sum(1 for term in ['energy', 'force', 'field', 'mass', 'equation'] if term in text_sample)
        
        return problem_count >= 2 and reasoning_count >= 3 and physics_count >= 2
    
    def save_training_data(self, training_examples: List[Dict]) -> str:
        """Save RL training data to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"rl_physics_training_{timestamp}.json"
        filepath = self.training_dir / filename
        
        training_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "format_type": "rl_physics_reasoning_training",
                "total_examples": len(training_examples),
                "format_version": "2.0",
                "description": "RL training data extracted from physics papers",
                "quality_stats": self._get_quality_statistics(training_examples)
            },
            "training_examples": training_examples
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(training_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"RL training data saved to {filepath}")
        return str(filepath)
    
    def _get_difficulty_distribution(self, examples: List[Dict]) -> Dict[str, int]:
        """Get distribution of difficulty levels"""
        distribution = {}
        for example in examples:
            difficulty = example["metadata"]["difficulty"]
            distribution[difficulty] = distribution.get(difficulty, 0) + 1
        return distribution
    
    def _get_quality_statistics(self, examples: List[Dict]) -> Dict[str, float]:
        """Get quality statistics"""
        if not examples:
            return {}
        
        qualities = [ex["metadata"]["reasoning_quality"] for ex in examples]
        step_counts = [ex["metadata"]["step_count"] for ex in examples]
        
        return {
            "avg_reasoning_quality": sum(qualities) / len(qualities),
            "min_reasoning_quality": min(qualities),
            "max_reasoning_quality": max(qualities),
            "avg_step_count": sum(step_counts) / len(step_counts),
            "high_quality_count": sum(1 for q in qualities if q > 0.7),
            "total_examples": len(examples)
        }