#!/usr/bin/env python3
"""
Enhanced Paper Quality Filter - Updated Main Entry Point
Integrates self-contained benchmark creation and high-quality RL training data
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
from analysis.pdf_processor import EnhancedPDFProcessor
from analysis.classifier import SubtlePhysicsClassifier
from utils.categories import ARXIV_CATEGORIES, VIXRA_CATEGORIES

# Import enhanced builders
from analysis.enhanced_benchmark_builder import SelfContainedBenchmarkBuilder
from analysis.enhanced_training_builder import ChainOfThoughtTrainingBuilder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_enhanced_analysis(args):
    """Run the enhanced paper analysis pipeline with improved benchmark and training data"""
    
    # Setup output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = Path(args.output_dir) / f"enhanced_analysis_{timestamp}"
    output_base.mkdir(parents=True, exist_ok=True)
    
    analysis_dir = output_base / "analysis_results"
    analysis_dir.mkdir(exist_ok=True)
    
    # Initialize enhanced builders
    benchmark_builder = SelfContainedBenchmarkBuilder(str(output_base))
    training_builder = ChainOfThoughtTrainingBuilder(str(output_base))
    
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
    print("Focus: Self-contained benchmarks + High-quality RL training data")
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
            
        print(f"Found {len(papers)} papers. Processing with enhanced analysis...\n")
        
        interesting_papers = []
        self_contained_benchmarks = []
        all_rl_training_examples = []
        
        for i, paper in enumerate(papers, 1):
            print(f"PAPER {i}: {paper.title}")
            print(f"Authors: {', '.join(paper.authors)}")
            print(f"Subject: {paper.subject}")
            print(f"Abstract: {paper.abstract[:200]}...")
            print(f"PDF URL: {paper.pdf_url}")
            print()
            
            # Try to download and process PDF with enhanced extraction
            print("Attempting enhanced PDF download and processing...")
            pdf_success = await scraper.download_pdf(paper)
            if pdf_success:
                print(f"PDF downloaded successfully: {paper.pdf_path}")
                paper.full_text, paper.metadata = EnhancedPDFProcessor.extract_text_enhanced(paper.pdf_path)
                print(f"Enhanced PDF processed: {paper.metadata.get('word_count', 0)} words")
                print(f"Physics density: {paper.metadata.get('physics_density', 0):.1f}%")
            else:
                print("PDF download failed - using abstract only")
                paper.full_text = paper.abstract
            
            # Classify paper for subtle physics issues
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
                
                if assessment.stage_2_scores:
                    print(f"  Technical Scores:")
                    for aspect, score in assessment.stage_2_scores.items():
                        print(f"    {aspect}: {score}/10")
                
                print(f"  Analysis: {assessment.reasoning[:300]}...")
                
                # Save individual analysis results
                analysis_file = analysis_dir / f"analysis_{paper.id}.json"
                analysis_data = {
                    "paper": {
                        "id": paper.id,
                        "title": paper.title,
                        "authors": paper.authors,
                        "subject": paper.subject,
                        "abstract": paper.abstract,
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
                    "full_text_sample": paper.full_text[:5000] if paper.full_text else None
                }
                
                with open(analysis_file, 'w', encoding='utf-8') as f:
                    json.dump(analysis_data, f, indent=2, ensure_ascii=False)
                
                print(f"  Analysis saved to: {analysis_file}")
                
                # Track interesting papers
                if assessment.overall_score >= 0.4:
                    interesting_papers.append((paper, assessment))
                
                # Create SELF-CONTAINED benchmark items (key improvement)
                if args.create_benchmark and paper.full_text:
                    print("  Creating self-contained benchmark problems...")
                    benchmark_item = benchmark_builder.create_self_contained_benchmark(
                        paper, assessment, paper.full_text
                    )
                    if benchmark_item:
                        self_contained_benchmarks.append(benchmark_item)
                        problem_count = len(benchmark_item.get("problems", []))
                        print(f"  ✓ Created {problem_count} self-contained physics problems")
                    else:
                        print("  ✗ Insufficient content for benchmark creation")
                
                # Extract HIGH-QUALITY RL training examples (key improvement)
                if args.create_training and paper.full_text:
                    print("  Extracting high-quality RL training examples...")
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
        if args.create_benchmark and self_contained_benchmarks:
            print("Saving self-contained reasoning benchmarks...")
            benchmark_file = benchmark_builder.save_benchmark(self_contained_benchmarks)
            total_problems = sum(len(item.get("problems", [])) for item in self_contained_benchmarks)
            print(f"✓ Benchmark saved with {len(self_contained_benchmarks)} problem sets")
            print(f"✓ Total self-contained problems: {total_problems}")
            print(f"Location: {benchmark_file}")
        
        # Save enhanced RL training data
        if args.create_training and all_rl_training_examples:
            print("Saving high-quality RL training data...")
            training_file = training_builder.save_training_data(all_rl_training_examples)
            high_quality_count = sum(1 for ex in all_rl_training_examples 
                                   if ex["metadata"]["reasoning_quality"] > 0.7)
            print(f"✓ Training data saved with {len(all_rl_training_examples)} examples")
            print(f"✓ High quality examples (>0.7): {high_quality_count}")
            print(f"Location: {training_file}")
        
        # Enhanced Summary Report
        print("\n" + "="*80)
        print("ENHANCED ANALYSIS SUMMARY")
        print("="*50)
        print(f"Total papers processed: {len(papers)}")
        print(f"Papers with interesting issues: {len(interesting_papers)}")
        print(f"Individual analysis files: {len(papers)}")
        
        if args.create_benchmark:
            total_problems = sum(len(item.get("problems", [])) for item in self_contained_benchmarks)
            print(f"Self-contained benchmark sets: {len(self_contained_benchmarks)}")
            print(f"Total benchmark problems: {total_problems}")
        
        if args.create_training:
            if all_rl_training_examples:
                avg_quality = sum(ex["metadata"]["reasoning_quality"] for ex in all_rl_training_examples) / len(all_rl_training_examples)
                high_quality = sum(1 for ex in all_rl_training_examples if ex["metadata"]["reasoning_quality"] > 0.7)
                print(f"RL training examples: {len(all_rl_training_examples)}")
                print(f"Average quality score: {avg_quality:.2f}")
                print(f"High quality examples: {high_quality}")
        
        print(f"All outputs saved to: {output_base}")
        
        # Show most interesting papers
        if interesting_papers:
            print("\nMost interesting papers found:")
            interesting_papers.sort(key=lambda x: x[1].overall_score, reverse=True)
            
            for j, (paper, assessment) in enumerate(interesting_papers[:3], 1):
                clean_title = paper.title[:80] + "..." if len(paper.title) > 80 else paper.title
                print(f"\n{j}. {clean_title}")
                print(f"   Score: {assessment.overall_score:.2f}")
                print(f"   Type: {assessment.stage_3_recommendation}")
                key_issues = assessment.subtle_issues[:2] if assessment.subtle_issues else ["No specific issues"]
                print(f"   Key Issues: {', '.join(key_issues)}")
        
        # Create consolidated summary file
        summary_file = output_base / "enhanced_analysis_summary.json"
        summary_data = {
            "analysis_metadata": {
                "timestamp": timestamp,
                "source": args.source,
                "category": args.category,
                "total_papers": len(papers),
                "interesting_papers": len(interesting_papers),
                "provider": args.provider,
                "depth": args.depth,
                "enhancement_version": "2.0"
            },
            "interesting_papers": [
                {
                    "paper_id": paper.id,
                    "title": paper.title[:100],
                    "score": assessment.overall_score,
                    "recommendation": assessment.stage_3_recommendation,
                    "issue_count": len(assessment.subtle_issues)
                }
                for paper, assessment in interesting_papers
            ],
            "self_contained_benchmarks": {
                "total_sets": len(self_contained_benchmarks),
                "total_problems": sum(len(item.get("problems", [])) for item in self_contained_benchmarks),
                "domains_covered": list(set(item["metadata"]["domain"] for item in self_contained_benchmarks if "metadata" in item))
            } if args.create_benchmark else None,
            "rl_training_data": {
                "total_examples": len(all_rl_training_examples),
                "avg_quality": sum(ex["metadata"]["reasoning_quality"] for ex in all_rl_training_examples) / len(all_rl_training_examples) if all_rl_training_examples else 0,
                "high_quality_count": sum(1 for ex in all_rl_training_examples if ex["metadata"]["reasoning_quality"] > 0.7),
                "difficulty_distribution": training_builder._get_difficulty_distribution(all_rl_training_examples) if all_rl_training_examples else {}
            } if args.create_training and all_rl_training_examples else None
        }
        
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nEnhanced summary report: {summary_file}")

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
    parser = argparse.ArgumentParser(description="Enhanced Paper Quality Filter - Self-contained benchmarks and RL training")
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
                       help="Create self-contained reasoning benchmarks")
    parser.add_argument("--create-training", action="store_true", default=True,
                       help="Extract high-quality RL training data")
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