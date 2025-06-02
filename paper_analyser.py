#!/usr/bin/env python3
"""
Paper Quality Filter - Main Entry Point
A system for identifying papers with subtle physics issues from arXiv or viXra
"""

import asyncio
import argparse
import logging
import os
from pathlib import Path

# Local imports
from scrapers.arxiv_scraper import ArXivScraper
from scrapers.vixra_scraper import ViXraScraper
from analysis.pdf_processor import PDFProcessor
from analysis.classifier import SubtlePhysicsClassifier
from utils.categories import ARXIV_CATEGORIES, VIXRA_CATEGORIES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

async def run_analysis(args):
    """Run the paper analysis pipeline with the given arguments"""
    
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
    print(f"{source_name} Paper Quality Filter - ENHANCED MODE ({args.provider.upper()})")
    print("Focus: Papers with subtle physics issues (not obvious crackpot)")
    print("="*70)
    
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
            
        print(f"Found {len(papers)} papers. Processing each through the subtle physics filter...\n")
        
        interesting_papers = []
        
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
                paper.full_text, paper.metadata = PDFProcessor.extract_text(paper.pdf_path)
                print(f"PDF processed: {paper.metadata.get('word_count', 0)} words")
            else:
                print("PDF download failed - using abstract only")
            
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
                
                # Track interesting papers
                if assessment.overall_score >= 0.4:
                    interesting_papers.append((paper, assessment))
            except Exception as e:
                logger.error(f"Error analyzing paper {paper.id}: {e}", exc_info=True)
                print(f"  Error analyzing paper: {e}")
                continue
            
            print("\n" + "="*80 + "\n")
        
        # Summary
        print("SUMMARY")
        print("="*50)
        print(f"Total papers analyzed: {len(papers)}")
        print(f"Papers with interesting issues: {len(interesting_papers)}")
        
        if interesting_papers:
            print("\nMost interesting papers found:")
            interesting_papers.sort(key=lambda x: x[1].overall_score, reverse=True)
            
            for j, (paper, assessment) in enumerate(interesting_papers[:3], 1):
                print(f"\n{j}. {paper.title}")
                print(f"   Score: {assessment.overall_score:.2f}")
                print(f"   Type: {assessment.stage_3_recommendation}")
                print(f"   Key Issues: {', '.join(assessment.subtle_issues[:3])}")

def list_categories(args):
    """List available categories for the selected source"""
    categories = ARXIV_CATEGORIES if args.source.lower() == "arxiv" else VIXRA_CATEGORIES
    
    print(f"Available {args.source} categories:")
    print("="*50)
    for code, name in categories.items():
        print(f"  {code:15} - {name}")
    print(f"\nUse --category <code> to filter by category, or --category all for all categories")

def main():
    """Main entry point for the script"""
    parser = argparse.ArgumentParser(description="Paper Quality Filter - Identifies papers with subtle physics issues")
    parser.add_argument("--source", choices=["arxiv", "vixra"], required=True, help="Source repository to scrape")
    parser.add_argument("--provider", choices=["openai", "gemini"], default="openai", help="LLM provider to use")
    parser.add_argument("--openai-key", help="OpenAI API key (required for OpenAI provider)")
    parser.add_argument("--gemini-key", help="Google Gemini API key (required for Gemini provider)")
    parser.add_argument("--category", help="Category to process (use --list-categories to see options)")
    parser.add_argument("--days-back", type=int, default=30, help="Days back to scrape")
    parser.add_argument("--max-papers", type=int, default=5, help="Maximum papers to process")
    parser.add_argument("--download-dir", help="Directory to download papers to")
    parser.add_argument("--depth", choices=["basic", "technical", "full", "force"], default="full", 
                       help="Analysis depth (basic=stage 1 only, technical=stages 1-2, full=all stages, force=analyze all papers)")
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
        asyncio.run(run_analysis(args))
    except KeyboardInterrupt:
        print("\nShutdown requested by user")
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)  # Include traceback
        raise

if __name__ == "__main__":
    main()