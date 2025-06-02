"""
LLM-based classifier for analyzing physics papers
"""
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional

import openai
import google.generativeai as genai

from models.paper import Paper, QualityAssessment
from prompts.classifier_prompts import (
    stage_1_sophistication_prompt,
    stage_2_subtle_analysis_prompt,
    stage_3_interesting_failures_prompt
)

logger = logging.getLogger(__name__)

class SubtlePhysicsClassifier:
    """LLM-based classifier focused on subtle physics issues"""
    
    def __init__(self, provider: str = "openai", openai_api_key: Optional[str] = None, gemini_api_key: Optional[str] = None):
        self.provider = provider.lower()
        
        if self.provider == "openai":
            if not openai_api_key:
                raise ValueError("OpenAI API key required for OpenAI provider")
            self.client = openai.AsyncOpenAI(api_key=openai_api_key)
            self.model = "gpt-4"
        
        elif self.provider == "gemini":
            if not gemini_api_key:
                raise ValueError("Gemini API key required for Gemini provider")
            genai.configure(api_key=gemini_api_key)
            self.client = genai.GenerativeModel('gemini-1.5-flash')
            self.model = "gemini-1.5-flash"
        
        else:
            raise ValueError("Provider must be 'openai' or 'gemini'")
        
        logger.info(f"Initialized SubtlePhysicsClassifier with {self.provider} provider using model {self.model}")
    
    async def _call_llm(self, prompt: str) -> str:
        """Call the configured LLM provider"""
        try:
            if self.provider == "openai":
                response = await self.client.chat.completions.create(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=0.1
                )
                return response.choices[0].message.content
            
            elif self.provider == "gemini":
                loop = asyncio.get_event_loop()
                
                def _sync_generate():
                    try:
                        response = self.client.generate_content(
                            prompt,
                            generation_config=genai.types.GenerationConfig(
                                temperature=0.1,
                                candidate_count=1,
                            )
                        )
                        return response.text
                    except Exception as e:
                        logger.error(f"Gemini generation error: {e}")
                        # Return a simple JSON response to avoid cascading errors
                        return '{"error": "Failed to generate content"}'
                
                return await loop.run_in_executor(None, _sync_generate)
        
        except Exception as e:
            logger.error(f"LLM API call failed: {e}")
            # Return a simple JSON response to avoid cascading errors
            return '{"error": "LLM API call failed"}'
    
    async def classify_paper(self, paper: Paper, depth: str = "full") -> QualityAssessment:
        """Classify a paper focusing on subtle physics issues"""
        
        assessment = QualityAssessment(
            paper_id=paper.id,
            overall_score=0.0,
            stage_1_pass=False,
            stage_2_scores={},
            stage_3_recommendation="REJECT",
            subtle_issues=[],
            physics_sophistication=0.0,
            reasoning="",
            processing_timestamp=datetime.now().isoformat()
        )
        
        # Stage 1: Filter out obvious crackpot but keep sophisticated attempts
        stage_1_result = await self.stage_1_sophistication_filter(paper)
        assessment.stage_1_pass = stage_1_result['pass']
        assessment.physics_sophistication = stage_1_result['sophistication_score']
        
        if not assessment.stage_1_pass and depth != "force":
            assessment.reasoning = "Failed sophistication filter: " + stage_1_result['reasoning']
            return assessment
        
        # Stage 2: Analyze for subtle physics issues
        if depth in ["technical", "full", "force"]:
            stage_2_result = await self.stage_2_subtle_analysis(paper)
            assessment.stage_2_scores = stage_2_result['scores']
            assessment.subtle_issues = stage_2_result['subtle_issues']
        
        # Stage 3: Final evaluation for interesting failures
        if depth in ["full", "force"]:
            stage_3_result = await self.stage_3_interesting_failures(paper)
            assessment.stage_3_recommendation = stage_3_result['recommendation']
            assessment.reasoning = stage_3_result['reasoning']
        
        # Calculate overall score
        assessment.overall_score = self._calculate_overall_score(assessment)
        
        return assessment
    
    async def stage_1_sophistication_filter(self, paper: Paper) -> Dict[str, Any]:
        """Stage 1: Filter for physics sophistication, exclude obvious crackpot"""
        
        # Get text sample for analysis
        text_sample = paper.full_text[:3000] if paper.full_text else "Not available"
        
        # Generate prompt using the template
        prompt = stage_1_sophistication_prompt(
            paper.title,
            paper.authors,
            paper.subject,
            paper.abstract,
            text_sample
        )
        
        try:
            response_text = await self._call_llm(prompt)
            response_text = response_text.strip()
            
            # Extract JSON from the response if needed
            if response_text.startswith('```json'):
                response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
            
            # Find JSON content in the text if it's not properly formatted
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                response_text = response_text[json_start:json_end+1]
            
            result = json.loads(response_text)
            
            # Ensure all required fields are present
            required_fields = ['pass', 'sophistication_score', 'reasoning']
            for field in required_fields:
                if field not in result:
                    if field == 'pass':
                        result[field] = False
                    elif field == 'sophistication_score':
                        result[field] = 0.0
                    else:
                        result[field] = f"Missing {field} in response"
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in stage 1: {e}")
            logger.error(f"Response text: {response_text[:200]}...")
            return {
                "pass": False,
                "sophistication_score": 0.0,
                "reasoning": f"Failed to parse JSON response: {e}",
                "physics_knowledge_indicators": [],
                "crackpot_indicators": ["JSON parsing error"]
            }
        except Exception as e:
            logger.error(f"Stage 1 classification error: {e}")
            return {
                "pass": False,
                "sophistication_score": 0.0,
                "reasoning": f"Classification error: {e}",
                "physics_knowledge_indicators": [],
                "crackpot_indicators": ["Classification error"]
            }
    
    async def stage_2_subtle_analysis(self, paper: Paper) -> Dict[str, Any]:
        """Stage 2: Analyze for subtle physics issues and errors"""
        
        # Get text for analysis (limited to 5000 chars to fit in context window)
        text_for_analysis = paper.full_text[:5000] if paper.full_text else paper.abstract
        
        # Generate prompt using the template
        prompt = stage_2_subtle_analysis_prompt(
            paper.title,
            paper.subject,
            text_for_analysis
        )
        
        try:
            response_text = await self._call_llm(prompt)
            response_text = response_text.strip()
            
            # Extract JSON from the response if needed
            if response_text.startswith('```json'):
                response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
            
            # Find JSON content in the text if it's not properly formatted
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                response_text = response_text[json_start:json_end+1]
            
            result = json.loads(response_text)
            
            # Ensure all required fields are present
            if 'scores' not in result:
                result['scores'] = {
                    "mathematical_errors": 5,
                    "physics_assumptions": 5,
                    "logical_consistency": 5,
                    "literature_integration": 5
                }
            
            if 'subtle_issues' not in result:
                result['subtle_issues'] = ["Could not determine specific issues"]
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in stage 2: {e}")
            logger.error(f"Response text: {response_text[:200]}...")
            return {
                "scores": {"mathematical_errors": 5, "physics_assumptions": 5, "logical_consistency": 5, "literature_integration": 5},
                "subtle_issues": [f"Failed to parse JSON response: {e}"],
                "technical_analysis": f"JSON parsing error: {e}",
                "redeeming_aspects": []
            }
        except Exception as e:
            logger.error(f"Stage 2 classification error: {e}")
            return {
                "scores": {"mathematical_errors": 5, "physics_assumptions": 5, "logical_consistency": 5, "literature_integration": 5},
                "subtle_issues": [f"Classification error: {e}"],
                "technical_analysis": f"Classification error: {e}",
                "redeeming_aspects": []
            }
    
    async def stage_3_interesting_failures(self, paper: Paper) -> Dict[str, Any]:
        """Stage 3: Evaluate if the errors/failures are interesting or educational"""
        
        # Get text for analysis (limited to 6000 chars to fit in context window)
        text_for_analysis = paper.full_text[:6000] if paper.full_text else "Limited text available"
        
        # Generate prompt using the template
        prompt = stage_3_interesting_failures_prompt(
            paper.title,
            paper.subject,
            paper.abstract,
            text_for_analysis
        )
        
        try:
            response_text = await self._call_llm(prompt)
            response_text = response_text.strip()
            
            # Extract JSON from the response if needed
            if response_text.startswith('```json'):
                response_text = response_text[7:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
            elif response_text.startswith('```'):
                response_text = response_text[3:]
                if response_text.endswith('```'):
                    response_text = response_text[:-3]
            
            # Find JSON content in the text if it's not properly formatted
            json_start = response_text.find('{')
            json_end = response_text.rfind('}')
            if json_start != -1 and json_end != -1 and json_end > json_start:
                response_text = response_text[json_start:json_end+1]
            
            result = json.loads(response_text)
            
            # Ensure all required fields are present
            if 'recommendation' not in result:
                result['recommendation'] = "EDUCATIONAL_FAILURE"
            
            if 'reasoning' not in result:
                result['reasoning'] = "Could not determine specific reasoning"
            
            return result
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error in stage 3: {e}")
            logger.error(f"Response text: {response_text[:200]}...")
            return {
                "recommendation": "EDUCATIONAL_FAILURE",
                "reasoning": f"Failed to parse JSON response: {e}",
                "educational_aspects": [],
                "technical_depth": "Unknown due to JSON error",
                "error_subtlety": "Cannot assess due to JSON error",
                "confidence": 0.5
            }
        except Exception as e:
            logger.error(f"Stage 3 classification error: {e}")
            return {
                "recommendation": "EDUCATIONAL_FAILURE",
                "reasoning": f"Classification error: {e}",
                "educational_aspects": [],
                "technical_depth": "Unknown due to error",
                "error_subtlety": "Cannot assess",
                "confidence": 0.5
            }
    
    def _calculate_overall_score(self, assessment: QualityAssessment) -> float:
        """Calculate overall score focusing on interesting failures"""
        if not assessment.stage_1_pass:
            return 0.0
        
        score = assessment.physics_sophistication * 0.3
        
        if assessment.stage_2_scores:
            # Higher scores for more subtle/interesting errors
            avg_technical = sum(assessment.stage_2_scores.values()) / len(assessment.stage_2_scores)
            # Invert scoring - we want interesting failures, not perfect papers
            technical_score = max(0, 10 - avg_technical) / 10.0
            score += technical_score * 0.4
        
        recommendation_scores = {
            "REJECT": 0.0,
            "BORING_MISTAKE": 0.2,
            "EDUCATIONAL_FAILURE": 0.6,
            "CREATIVE_APPROACH": 0.8,
            "SOPHISTICATED_ERROR": 1.0
        }
        
        rec_score = recommendation_scores.get(assessment.stage_3_recommendation, 0.0)
        score += rec_score * 0.3
        
        return min(1.0, score)