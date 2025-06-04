"""
Enhanced Benchmark Builder - Creates self-contained physics reasoning problems
Following UGPhysics standards for rigorous benchmark creation
Key improvement: NO references to original papers - fully self-contained problems
"""

import re
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime

logger = logging.getLogger(__name__)

class SelfContainedBenchmarkBuilder:
    """Creates self-contained physics reasoning benchmarks without paper references"""
    
    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.benchmark_dir = self.output_dir / "benchmark"
        self.benchmark_dir.mkdir(parents=True, exist_ok=True)
        
        # Physics problem templates for different domains
        self.problem_templates = self._load_problem_templates()
        
    def create_self_contained_benchmark(self, paper, assessment, full_text: str) -> Optional[Dict[str, Any]]:
        """Create benchmark with self-contained physics problems (NO paper references)"""
        
        if not self._is_suitable_for_benchmark(paper, full_text):
            return None
        
        # Extract physics content and concepts (NOT paper details)
        physics_content = self._extract_physics_concepts_detailed(full_text)
        if not physics_content:
            return None
        
        # Generate self-contained problems based on extracted concepts
        problems = self._generate_self_contained_problems(
            physics_content, paper.subject, assessment
        )
        
        if len(problems) < 2:
            return None
        
        benchmark_item = {
            "metadata": {
                "domain": self._classify_physics_domain(paper.subject),
                "difficulty_level": self._assess_problem_difficulty(physics_content),
                "problem_count": len(problems),
                "created_at": datetime.now().isoformat(),
                "version": "2.0",
                "source_type": "physics_concept_extraction"  # NOT paper reference
            },
            "problems": problems
        }
        
        return benchmark_item
    
    def _extract_physics_concepts_detailed(self, text: str) -> Dict[str, Any]:
        """Extract detailed physics concepts and mathematical structures"""
        
        concepts = {
            "equations": [],
            "physical_laws": [],
            "mathematical_methods": [],
            "problem_scenarios": [],
            "key_variables": [],
            "physics_principles": []
        }
        
        # Extract equations with context (clean, no paper references)
        equation_patterns = [
            r'([A-Za-z]\s*=\s*[^,.\n]{10,100})',  # Variable equations
            r'(\\frac\{[^}]+\}\{[^}]+\}[^.]{0,50})',  # Fraction expressions
            r'((?:F|E|V|p|m|v|a|g)\s*=\s*[^,.\n]{5,80})',  # Physics variables
        ]
        
        for pattern in equation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                clean_eq = self._clean_equation(match)
                if clean_eq and len(clean_eq) > 5:
                    concepts["equations"].append(clean_eq)
        
        # Extract physical laws and principles
        law_patterns = [
            r'(conservation of \w+)',
            r'(Newton\'?s? \w+ law)',
            r'(Maxwell\'?s? equations?)',
            r'(Schr[öo]dinger equation)',
            r'(Einstein\'?s? \w+ relativity)',
            r'(thermodynamic \w+ law)',
        ]
        
        for pattern in law_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts["physical_laws"].extend([m.lower() for m in matches])
        
        # Extract mathematical methods
        method_patterns = [
            r'(differential equation)',
            r'(integration by parts)',
            r'(Taylor expansion)',
            r'(Fourier transform)',
            r'(eigenvalue problem)',
            r'(perturbation theory)',
        ]
        
        for pattern in method_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts["mathematical_methods"].extend([m.lower() for m in matches])
        
        # Extract physical scenarios (for problem generation)
        scenario_patterns = [
            r'(particle in a \w+ potential)',
            r'(\w+ oscillator)',
            r'(electromagnetic field in \w+)',
            r'(wave propagation in \w+)',
            r'(quantum system with \w+)',
        ]
        
        for pattern in scenario_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            concepts["problem_scenarios"].extend([m.lower() for m in matches])
        
        return concepts if any(concepts.values()) else None
    
    def _generate_self_contained_problems(self, physics_content: Dict, subject: str, assessment) -> List[Dict]:
        """Generate self-contained physics problems based on extracted concepts"""
        
        problems = []
        domain = self._classify_physics_domain(subject)
        
        # Problem Type 1: Mathematical Derivation (most rigorous)
        if physics_content["equations"]:
            derivation_problem = self._create_derivation_problem(
                physics_content, domain, assessment
            )
            if derivation_problem:
                problems.append(derivation_problem)
        
        # Problem Type 2: Conceptual Analysis
        if physics_content["physical_laws"]:
            conceptual_problem = self._create_conceptual_problem(
                physics_content, domain, assessment
            )
            if conceptual_problem:
                problems.append(conceptual_problem)
        
        # Problem Type 3: Error Detection (based on assessment issues)
        if assessment.subtle_issues:
            error_problem = self._create_error_detection_problem(
                physics_content, domain, assessment
            )
            if error_problem:
                problems.append(error_problem)
        
        # Problem Type 4: Application Problem
        if physics_content["problem_scenarios"]:
            application_problem = self._create_application_problem(
                physics_content, domain, assessment
            )
            if application_problem:
                problems.append(application_problem)
        
        return problems
    
    def _create_derivation_problem(self, physics_content: Dict, domain: str, assessment) -> Optional[Dict]:
        """Create mathematical derivation problem (UGPhysics style)"""
        
        if not physics_content["equations"]:
            return None
        
        # Generate problem based on domain
        if domain == "mechanics":
            problem = self._create_mechanics_derivation()
        elif domain == "electromagnetism":
            problem = self._create_em_derivation()
        elif domain == "quantum":
            problem = self._create_quantum_derivation()
        elif domain == "thermodynamics":
            problem = self._create_thermo_derivation()
        else:
            problem = self._create_general_derivation()
        
        if not problem:
            return None
        
        return {
            "problem_id": f"derivation_{domain}_{hash(str(physics_content['equations'])) % 10000}",
            "type": "mathematical_derivation",
            "difficulty": "intermediate",
            "domain": domain,
            "problem_statement": problem["statement"],
            "solution_outline": problem["solution"],
            "evaluation_criteria": [
                "Mathematical rigor and correctness",
                "Proper application of physics principles",
                "Dimensional consistency",
                "Logical flow of derivation"
            ],
            "common_errors": problem.get("common_errors", [
                "Sign errors in mathematical steps",
                "Incorrect application of boundary conditions", 
                "Dimensional inconsistencies",
                "Missing physical assumptions"
            ])
        }
    
    def _create_mechanics_derivation(self) -> Dict:
        """Create mechanics derivation problem (completely self-contained)"""
        
        return {
            "statement": """Consider a particle of mass m moving under the influence of a conservative force F = -dV/dx, where V(x) is the potential energy.

Starting from Newton's second law and the definition of potential energy, derive the equation for the total mechanical energy E = K + V, where K is kinetic energy.

Show that for conservative forces, the total mechanical energy is conserved (dE/dt = 0).

Provide a complete mathematical derivation including:
1. Statement of Newton's second law
2. Definition of kinetic energy
3. Work-energy theorem application
4. Proof of energy conservation""",
            
            "solution": """1. Start with Newton's second law: F = ma = m(dv/dt)
2. Use chain rule: dv/dt = (dv/dx)(dx/dt) = v(dv/dx)
3. Therefore: F = mv(dv/dx)
4. For conservative force: F = -dV/dx
5. Multiply both sides by dx: F dx = mv dv
6. Integrate: ∫F dx = ∫mv dv
7. This gives: -ΔV = ½mv² - ½mv₀²
8. Rearranging: ½mv² + V = ½mv₀² + V₀ = constant
9. Therefore: E = K + V = constant (energy conservation)""",
            
            "common_errors": [
                "Forgetting the negative sign in F = -dV/dx",
                "Incorrect application of chain rule for dv/dt",
                "Sign errors when integrating the work integral",
                "Not recognizing that the constant represents total energy"
            ]
        }
    
    def _create_em_derivation(self) -> Dict:
        """Create electromagnetism derivation problem"""
        
        return {
            "statement": """Consider a point charge q moving with velocity v in uniform electric and magnetic fields E and B.

Derive the Lorentz force equation F = q(E + v × B) from first principles.

Your derivation should include:
1. The electric force component and its physical origin
2. The magnetic force component and why it's perpendicular to velocity
3. The significance of the cross product v × B
4. Discussion of when each component dominates

Explain the physical meaning of each term and the conditions under which this equation applies.""",
            
            "solution": """1. Electric force on charge: F_E = qE (Coulomb's law)
   - Force is parallel/antiparallel to E field
   - Does work on charge (can change kinetic energy)
   
2. Magnetic force arises from Lorentz transformation of fields
   - Moving charge experiences force perpendicular to both v and B
   - Magnetic force: F_B = q(v × B)
   - Magnitude: |F_B| = qvB sin θ where θ is angle between v and B
   
3. Total electromagnetic force: F = F_E + F_B = q(E + v × B)

4. Physical significance:
   - Electric term: can do work, changes particle energy
   - Magnetic term: always perpendicular to v, does no work
   - For v << c: magnetic term often negligible
   - Cross product ensures F_B ⊥ v and F_B ⊥ B"""
        }
    
    def _create_quantum_derivation(self) -> Dict:
        """Create quantum mechanics derivation problem"""
        
        return {
            "statement": """For a quantum mechanical particle in a one-dimensional infinite square well of width L (0 ≤ x ≤ L), derive the allowed energy levels.

Starting from the time-independent Schrödinger equation, show that:
1. The wave function must be zero at the boundaries
2. The allowed wave functions are ψₙ(x) = √(2/L) sin(nπx/L)
3. The energy eigenvalues are Eₙ = n²π²ℏ²/(2mL²)

Include proper normalization and explain the physical significance of the quantum number n.""",
            
            "solution": """1. Time-independent Schrödinger equation: -ℏ²/(2m) d²ψ/dx² = Eψ
2. Inside well (V=0): d²ψ/dx² = -k²ψ where k² = 2mE/ℏ²
3. General solution: ψ(x) = A sin(kx) + B cos(kx)
4. Boundary conditions: ψ(0) = 0 and ψ(L) = 0
5. From ψ(0) = 0: B = 0, so ψ(x) = A sin(kx)
6. From ψ(L) = 0: sin(kL) = 0, therefore kL = nπ (n = 1,2,3,...)
7. This gives: k = nπ/L, so E = n²π²ℏ²/(2mL²)
8. Normalization: ∫₀ᴸ |ψ|² dx = 1 gives A = √(2/L)
9. Physical meaning: n determines energy level, larger n = higher energy"""
        }
    
    def _create_thermo_derivation(self) -> Dict:
        """Create thermodynamics derivation problem"""
        
        return {
            "statement": """For an ideal gas undergoing a reversible adiabatic process, derive the relationship PVᵞ = constant, where γ = Cp/Cv.

Start with the first law of thermodynamics and the ideal gas law. Show that:
1. For an adiabatic process, dQ = 0
2. The relationship between temperature and volume: TVᵞ⁻¹ = constant
3. The pressure-volume relationship: PVᵞ = constant

Explain the physical significance of the adiabatic index γ.""",
            
            "solution": """1. First law: dU = dQ - dW, for adiabatic process dQ = 0
2. For ideal gas: dU = nCᵥdT, dW = PdV
3. Therefore: nCᵥdT = -PdV
4. From ideal gas law: PV = nRT, so P = nRT/V
5. Substituting: nCᵥdT = -(nRT/V)dV
6. Simplifying: CᵥdT = -RT(dV/V)
7. Rearranging: dT/T = -(R/Cᵥ)(dV/V)
8. Since γ = Cp/Cᵥ and Cp - Cᵥ = R: γ-1 = R/Cᵥ
9. Therefore: dT/T = -(γ-1)(dV/V)
10. Integrating: ln(T) = -(γ-1)ln(V) + constant
11. This gives: TVᵞ⁻¹ = constant
12. Using PV = nRT: PVᵞ = constant"""
        }
    
    def _create_general_derivation(self) -> Dict:
        """Create general physics derivation problem"""
        
        return {
            "statement": """Using dimensional analysis, derive the general form of the relationship between physical quantities in a system characterized by:
- A characteristic length scale L
- A characteristic time scale T  
- A characteristic energy scale E
- A characteristic mass scale M

Show how dimensional consistency constrains the possible functional relationships between these quantities. Apply the Buckingham π theorem to find dimensionless combinations.""",
            
            "solution": """1. Identify dimensions: [L], [T], [E] = ML²T⁻², [M]
2. Any physical relationship must be dimensionally consistent
3. For relationship f(L,T,E,M) = 0, use Buckingham π theorem
4. Number of dimensionless groups = 4 variables - 3 dimensions = 1
5. Form dimensionless combination: π = E T² / (M L²)
6. This must be a dimensionless constant
7. Therefore: E ∝ ML²/T² (consistent with kinetic energy form)
8. Alternative combinations: vT/L, Et²/(ML²), etc."""
        }
    
    def _create_conceptual_problem(self, physics_content: Dict, domain: str, assessment) -> Optional[Dict]:
        """Create conceptual understanding problem"""
        
        laws = physics_content.get("physical_laws", [])
        if not laws:
            return None
        
        primary_law = laws[0]
        
        return {
            "problem_id": f"conceptual_{domain}_{hash(primary_law) % 10000}",
            "type": "conceptual_analysis",
            "difficulty": "intermediate",
            "domain": domain,
            "problem_statement": self._generate_conceptual_statement(primary_law, domain),
            "evaluation_criteria": [
                "Understanding of underlying physics principles",
                "Ability to connect different concepts",
                "Recognition of assumptions and limitations",
                "Clear physical reasoning"
            ],
            "common_misconceptions": [
                "Confusing necessary vs sufficient conditions",
                "Misunderstanding the scope of applicability",
                "Incorrect causal relationships",
                "Oversimplification of complex phenomena"
            ]
        }
    
    def _generate_conceptual_statement(self, law: str, domain: str) -> str:
        """Generate conceptual problem statement"""
        
        if "conservation" in law:
            return """Analyze the principle of conservation laws in physics.

Consider a physical system where energy appears to be "lost" due to friction or other dissipative processes.

Explain:
1. How conservation of energy is maintained at the microscopic level
2. The role of entropy in understanding energy "loss"
3. The difference between conserved and non-conserved quantities
4. When conservation laws can be violated and under what circumstances

Provide specific examples where apparent violations of conservation laws led to the discovery of new physics principles."""
        
        elif "newton" in law:
            return """Examine Newton's laws of motion and their domain of applicability.

Consider scenarios where Newton's laws might appear to fail or require modification:
1. Motion at very high speeds (approaching speed of light)
2. Motion of very small particles (quantum scale)
3. Motion in strong gravitational fields
4. Non-inertial reference frames

For each scenario, explain:
- Why classical mechanics breaks down
- What new physics principles are needed
- How the classical laws emerge as limiting cases
- The experimental evidence for these modifications"""
        
        else:
            return f"""Analyze the physical principle of {law} and its applications.

Discuss:
1. The fundamental assumptions underlying this principle
2. The mathematical formulation and its physical meaning
3. Situations where this principle applies vs where it breaks down
4. How this principle connects to other areas of physics
5. Experimental tests and confirmations of this principle

Provide examples of how violations or extensions of this principle have led to new discoveries in physics."""
    
    def _create_error_detection_problem(self, physics_content: Dict, domain: str, assessment) -> Optional[Dict]:
        """Create error detection and analysis problem"""
        
        if not assessment.subtle_issues:
            return None
        
        # Create a problem with deliberate errors based on the assessment
        error_types = self._categorize_errors(assessment.subtle_issues)
        
        return {
            "problem_id": f"error_detection_{domain}_{hash(str(assessment.subtle_issues)) % 10000}",
            "type": "error_detection",
            "difficulty": "advanced",
            "domain": domain,
            "problem_statement": self._generate_error_detection_statement(error_types, domain),
            "error_categories": error_types,
            "evaluation_criteria": [
                "Identification of specific errors",
                "Understanding of correct physics principles",
                "Ability to distinguish subtle vs obvious mistakes",
                "Provision of correct solutions"
            ]
        }
    
    def _generate_error_detection_statement(self, error_types: List[str], domain: str) -> str:
        """Generate error detection problem with embedded mistakes"""
        
        if "dimensional" in error_types:
            return """The following derivation attempts to find the period of a simple pendulum:

Given: A pendulum of length L and mass m in gravitational field g
Starting from energy conservation: ½mω²A² = mgL(1 - cos θ)
For small angles: cos θ ≈ 1 - θ²/2
Therefore: ½mω²A² = mgLθ²/2
Since A = Lθ for small oscillations: ½mω²L²θ² = mgLθ²/2
Simplifying: ω²L = g
Therefore: ω = √(g/L)
The period is: T = 2π/ω = 2π√(L/g)

However, dimensional analysis gives: T ~ √(L·m/g)

Identify and correct all errors in this derivation. Explain why the dimensional analysis result is incorrect and provide the correct dimensional reasoning."""
        
        elif "mathematical" in error_types:
            return """A student derives the kinetic energy of a relativistic particle:

Starting with: E² = (pc)² + (mc²)²
The total energy is: E = γmc² where γ = 1/√(1-v²/c²)
The momentum is: p = γmv
Substituting: (γmc²)² = (γmvc)² + (mc²)²
Expanding: γ²m²c⁴ = γ²m²v²c² + m²c⁴
Simplifying: γ²c⁴ = γ²v²c² + c⁴
Dividing by c⁴: γ² = γ²v²/c² + 1
Therefore: γ²(1 - v²/c²) = 1
This gives: γ = 1/√(1-v²/c²) ✓

The kinetic energy is: K = E - mc² = γmc² - mc² = mc²(γ - 1)
For small velocities: γ ≈ 1 + v²/(2c²)
Therefore: K ≈ mc²(v²/(2c²)) = mv²/2 ✓

Find and explain the error in the reasoning that leads to an apparently correct result."""
        
        else:
            return """Analyze the following physics argument for errors in reasoning:

"Since quantum mechanics is probabilistic, it means that electrons don't have definite positions until measured. This implies that the electron can be anywhere in the universe with some probability. However, we know electrons are confined to atoms, which means there must be some force keeping them there even when not observed. This force must be non-quantum in nature since quantum mechanics only gives probabilities."

Identify the conceptual errors and misconceptions in this reasoning. Provide a correct explanation of:
1. What quantum mechanical probability means
2. How confinement works in quantum systems
3. The relationship between measurement and physical reality in quantum mechanics"""
    
    def _categorize_errors(self, subtle_issues: List[str]) -> List[str]:
        """Categorize types of errors found"""
        categories = []
        
        for issue in subtle_issues:
            issue_lower = issue.lower()
            if any(term in issue_lower for term in ["dimensional", "dimension", "unit"]):
                categories.append("dimensional")
            elif any(term in issue_lower for term in ["math", "equation", "calculation"]):
                categories.append("mathematical")
            elif any(term in issue_lower for term in ["assumption", "approximation"]):
                categories.append("physics_assumptions")
            elif any(term in issue_lower for term in ["logic", "reasoning", "contradiction"]):
                categories.append("logical")
            else:
                categories.append("conceptual")
        
        return list(set(categories))
    
    def _create_application_problem(self, physics_content: Dict, domain: str, assessment) -> Optional[Dict]:
        """Create practical application problem"""
        
        scenarios = physics_content.get("problem_scenarios", [])
        if not scenarios:
            return None
        
        return {
            "problem_id": f"application_{domain}_{hash(str(scenarios)) % 10000}",
            "type": "practical_application",
            "difficulty": "intermediate",
            "domain": domain,
            "problem_statement": self._generate_application_statement(scenarios[0], domain),
            "evaluation_criteria": [
                "Identification of relevant physics principles",
                "Appropriate mathematical modeling",
                "Realistic assumptions and approximations",
                "Correct order-of-magnitude results"
            ]
        }
    
    def _generate_application_statement(self, scenario: str, domain: str) -> str:
        """Generate practical application problem"""
        
        if "oscillator" in scenario:
            return """Design a mechanical oscillator system for a precision timing application.

Requirements:
- Period stability of 1 part in 10⁶
- Operating temperature range: 0°C to 50°C
- Amplitude variations less than ±5%

Consider:
1. What type of oscillator would be most suitable (simple pendulum, torsional, spring-mass)?
2. How do temperature variations affect the period?
3. What materials would minimize temperature sensitivity?
4. How do you account for air resistance and other damping effects?
5. What are the fundamental physical limits to timing precision?

Provide quantitative estimates and justify your design choices with physics principles."""
        
        elif "field" in scenario:
            return """Calculate the electromagnetic field configuration for a particle accelerator design.

Specifications:
- Accelerate protons from rest to 10% speed of light
- Acceleration distance: 1 meter
- Uniform electric field region

Determine:
1. Required electric field strength
2. Energy gained by the proton
3. Time required for acceleration
4. Power requirements (assuming 1000 protons/second)
5. Relativistic corrections to the motion

Consider the engineering constraints and safety requirements for such high-voltage systems."""
        
        else:
            return """Analyze the physics of a real-world system that demonstrates the principles under discussion.

Your analysis should include:
1. Identification of the key physics principles involved
2. Mathematical modeling of the system behavior
3. Reasonable approximations and their justifications
4. Comparison with experimental or observational data
5. Discussion of limitations and potential improvements

Choose a system that illustrates both the power and limitations of the theoretical framework."""
    
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
    
    def _assess_problem_difficulty(self, physics_content: Dict) -> str:
        """Assess overall difficulty level"""
        
        # Count advanced concepts
        advanced_indicators = 0
        
        for equation in physics_content.get("equations", []):
            if any(term in equation.lower() for term in ["tensor", "differential", "integral", "eigenvalue"]):
                advanced_indicators += 1
        
        for method in physics_content.get("mathematical_methods", []):
            if any(term in method for term in ["perturbation", "fourier", "eigenvalue"]):
                advanced_indicators += 1
        
        if advanced_indicators >= 3:
            return "advanced"
        elif advanced_indicators >= 1:
            return "intermediate"
        else:
            return "introductory"
    
    def _is_suitable_for_benchmark(self, paper, full_text: str) -> bool:
        """Check if paper is suitable for benchmark creation"""
        
        if not full_text or len(full_text.strip()) < 1000:
            return False
        
        # Check for substantial physics content
        physics_indicators = [
            'equation', 'energy', 'force', 'field', 'particle', 'wave',
            'quantum', 'relativity', 'conservation', 'momentum', 'mass'
        ]
        
        text_lower = full_text[:2000].lower()
        physics_count = sum(1 for indicator in physics_indicators if indicator in text_lower)
        
        return physics_count >= 5
    
    def _clean_equation(self, equation: str) -> str:
        """Clean and format equation"""
        # Remove extra whitespace and format
        cleaned = re.sub(r'\s+', ' ', equation.strip())
        # Remove common artifacts
        cleaned = re.sub(r'[^\w\s=+\-*/(){}\\.,]', '', cleaned)
        return cleaned
    
    def _load_problem_templates(self) -> Dict:
        """Load problem templates for different physics domains"""
        return {
            "mechanics": ["force_derivation", "energy_conservation", "momentum_conservation"],
            "electromagnetism": ["field_calculation", "wave_propagation", "circuit_analysis"],
            "quantum": ["schrodinger_equation", "eigenvalue_problems", "measurement_theory"],
            "thermodynamics": ["entropy_calculation", "cycle_efficiency", "phase_transitions"]
        }
    
    def save_benchmark(self, benchmark_items: List[Dict]) -> str:
        """Save self-contained benchmark to file"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"self_contained_physics_benchmark_{timestamp}.json"
        filepath = self.benchmark_dir / filename
        
        benchmark_data = {
            "metadata": {
                "created_at": datetime.now().isoformat(),
                "benchmark_type": "self_contained_physics_reasoning",
                "total_problems": sum(len(item.get("problems", [])) for item in benchmark_items),
                "total_sets": len(benchmark_items),
                "format_version": "2.0",
                "description": "Self-contained physics reasoning problems without paper references"
            },
            "problem_sets": benchmark_items
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(benchmark_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Self-contained benchmark saved to {filepath}")
        return str(filepath)