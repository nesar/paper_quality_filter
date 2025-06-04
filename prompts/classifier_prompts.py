"""
Prompts for the LLM-based classification of physics papers
"""

def stage_1_sophistication_prompt(title, authors, subject, abstract, text_sample):
    """
    Generate prompt for stage 1: Sophistication filter
    This filters out obvious crackpot papers but keeps sophisticated attempts
    """
    return f"""
    Analyze this physics paper to determine if it shows sufficient sophistication to warrant further analysis.
    We want to EXCLUDE obvious crackpot papers but INCLUDE papers that attempt serious physics even if they contain errors.
    
    Title: {title}
    Authors: {', '.join(authors)}
    Subject: {subject}
    Abstract: {abstract}
    
    Text sample: {text_sample}
    
    EXCLUDE (sophistication_score < 0.4) if paper contains:
    - Conspiracy theories about physics establishment
    - Claims of "disproving Einstein" or "disproving Maxwell" without mathematical rigor
    - Perpetual motion or free energy claims
    - Pure numerology without physical reasoning
    - Obvious word salad or incoherent arguments
    - Claims of "theory of everything" with no mathematical framework
    
    INCLUDE (sophistication_score >= 0.4) if paper shows:
    - Mathematical derivations (even if incorrect)
    - References to established physics literature
    - Proper use of physics terminology
    - Structured logical arguments
    - Specific calculations or numerical work
    - Attempts at experimental methodology
    
    We're looking for papers that are WRONG but show intellectual effort and physics knowledge.
    
    Respond in JSON format:
    {{
        "pass": true/false,
        "sophistication_score": 0.0-1.0,
        "reasoning": "explanation of sophistication level",
        "physics_knowledge_indicators": ["list", "of", "positive", "indicators"],
        "crackpot_indicators": ["list", "of", "negative", "indicators"]
    }}
    """

def stage_2_subtle_analysis_prompt(title, subject, full_text):
    """
    Generate prompt for stage 2: Analysis for subtle physics issues
    This analyzes for subtle errors in mathematics, physics, logic, and literature integration
    """
    return f"""
    Analyze this physics paper for SUBTLE errors and issues. Focus on technical mistakes that require physics knowledge to identify.
    
    Title: {title}
    Subject: {subject}
    Full text: {full_text}
    
    Look for SUBTLE issues like:
    
    Mathematical/Derivation Issues (score 1-10):
    - Sign errors in equations
    - Incorrect limiting cases
    - Misapplied mathematical techniques
    - Dimensional inconsistencies
    - Invalid approximations
    
    Physics Assumptions (score 1-10):
    - Inappropriate gauge choices
    - Neglected physical effects
    - Misunderstood symmetries
    - Incorrect boundary conditions
    - Wrong reference frames
    
    Logical Consistency (score 1-10):
    - Internal contradictions
    - Circular reasoning
    - Non-sequitur conclusions
    - Misinterpretation of results
    - Overgeneralization
    
    Literature Integration (score 1-10):
    - Misrepresentation of established work
    - Overlooked relevant research
    - Incorrect context placement
    - Cherry-picked references
    
    DO NOT penalize for:
    - Novel approaches or interpretations
    - Challenging established paradigms
    - Speculative extensions of known physics
    - Philosophical discussions about physics
    
    Respond in JSON format:
    {{
        "scores": {{
            "mathematical_errors": 1-10,
            "physics_assumptions": 1-10, 
            "logical_consistency": 1-10,
            "literature_integration": 1-10
        }},
        "subtle_issues": ["specific", "technical", "problems", "found"],
        "technical_analysis": "detailed explanation of issues found",
        "redeeming_aspects": ["positive", "aspects", "if", "any"]
    }}
    """

def stage_3_interesting_failures_prompt(title, subject, abstract, full_text):
    """
    Generate prompt for stage 3: Evaluation of interesting failures
    This evaluates if the errors/failures are interesting or educational
    """
    return f"""
    This physics paper has been identified as having technical issues. Evaluate whether these failures are INTERESTING or educational.
    
    Title: {title}
    Subject: {subject}
    Abstract: {abstract}
    Full text: {full_text}
    
    Consider:
    
    Educational Value:
    - Are the mistakes instructive?
    - Do they highlight common misconceptions?
    - Could correcting them lead to better understanding?
    - Are they subtle enough to fool others?
    
    Technical Depth:
    - Does the author demonstrate real physics knowledge?
    - Are the calculations non-trivial?
    - Is the mathematical framework sophisticated?
    - Are the errors at an advanced level?
    
    Interesting Failures:
    - Novel but flawed approaches
    - Creative but incorrect interpretations
    - Sophisticated errors that require expertise to spot
    - Well-executed work with subtle foundational flaws
    
    Categories:
    - EDUCATIONAL_FAILURE: Wrong but instructive, highlights common errors
    - SOPHISTICATED_ERROR: Advanced mistake requiring expertise to identify
    - CREATIVE_APPROACH: Novel method with flawed execution
    - BORING_MISTAKE: Common error, not particularly interesting
    - REJECT: Too flawed or unsophisticated
    
    Respond in JSON format:
    {{
        "recommendation": "EDUCATIONAL_FAILURE/SOPHISTICATED_ERROR/CREATIVE_APPROACH/BORING_MISTAKE/REJECT",
        "reasoning": "detailed explanation of why this failure is or isn't interesting",
        "educational_aspects": ["what", "could", "be", "learned"],
        "technical_depth": "assessment of sophistication level",
        "error_subtlety": "how subtle/advanced are the errors",
        "confidence": 0.0-1.0
    }}
    """