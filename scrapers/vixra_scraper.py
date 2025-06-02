"""
Scraper for viXra papers
"""
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiohttp
from bs4 import BeautifulSoup

from models.paper import Paper
from utils.categories import VIXRA_CATEGORIES

logger = logging.getLogger(__name__)

class ViXraScraper:
    """Handles scraping and downloading papers from viXra with category support"""
    
    BASE_URL = "https://vixra.org"
    
    def __init__(self, download_dir: str = "./papers"):
        self.download_dir = Path(download_dir)
        self.download_dir.mkdir(exist_ok=True)
        self.session = None
    
    async def __aenter__(self):
        self.session = aiohttp.ClientSession()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
    
    def get_category_urls(self, category: Optional[str] = None) -> List[str]:
        """Get URLs to scrape based on category selection"""
        if category and category in VIXRA_CATEGORIES:
            # viXra uses different URL patterns
            category_urls = [
                f"{self.BASE_URL}/{category}/",
                f"{self.BASE_URL}/astro/",  # Try general astrophysics page
                f"{self.BASE_URL}/all/{category}",
            ]
            return category_urls
        elif category:
            # Try direct category input
            return [
                f"{self.BASE_URL}/{category}/",
                f"{self.BASE_URL}/all/{category}",
                f"{self.BASE_URL}/all/",  # Fallback to all
            ]
        else:
            # Default to main pages and recent submissions
            return [
                f"{self.BASE_URL}/all/",
                f"{self.BASE_URL}/",
                "https://vixra.org/all/2025",  # Try year-based listing
                "https://vixra.org/all/2024",
            ]
    
    async def get_recent_papers(self, 
                              category: Optional[str] = None,
                              days_back: int = 30,
                              max_papers: int = 100) -> List[Paper]:
        """Scrape recent papers from viXra with category filtering"""
        papers = []
        
        # Map arXiv category code to viXra category code if needed
        from utils.categories import ARXIV_TO_VIXRA_CATEGORIES
        
        vixra_category = category
        if category and category in ARXIV_TO_VIXRA_CATEGORIES:
            vixra_category = ARXIV_TO_VIXRA_CATEGORIES[category]
            logger.info(f"Mapping arXiv category '{category}' to viXra category '{vixra_category}'")
        
        if vixra_category:
            logger.info(f"Searching in category: {VIXRA_CATEGORIES.get(vixra_category, vixra_category)}")
        
        urls_to_try = self.get_category_urls(vixra_category)  # Pass the mapped category
        
        for scrape_url in urls_to_try:
            logger.info(f"Trying URL: {scrape_url}")
            
            try:
                async with self.session.get(scrape_url) as response:
                    logger.info(f"Response status: {response.status}")
                    
                    if response.status != 200:
                        logger.warning(f"Failed: HTTP {response.status}")
                        continue
                    
                    content = await response.text()
                    logger.info(f"Content length: {len(content)} characters")
                    
                    soup = BeautifulSoup(content, 'html.parser')
                    
                    papers_found = await self._extract_papers_from_html(soup, max_papers, category)
                    
                    if papers_found:
                        papers.extend(papers_found)
                        logger.info(f"Found {len(papers_found)} papers from {scrape_url}")
                        break
                    else:
                        logger.warning(f"No papers found at {scrape_url}")
                        
            except Exception as e:
                logger.error(f"Error with {scrape_url}: {e}")
                continue
        
        logger.info(f"Total papers collected: {len(papers)}")
        return papers[:max_papers]
    
    async def _extract_papers_from_html(self, soup, max_papers: int, category: Optional[str] = None) -> List[Paper]:
        """Extract papers from HTML using viXra's specific structure"""
        papers = []
        
        # viXra has a specific structure with paper entries
        # Look for patterns like [3003] viXra:2505.0145 [pdf] submitted on...
        text_content = soup.get_text()
        
        # Use regex to find viXra paper entries
        paper_pattern = r'\[(\d+)\]\s+viXra:(\d{4}\.\d{4})\s+.*?submitted on ([\d\-\s:]+).*?Authors:\s+([^\n]+).*?Comments:\s+([^\n]+).*?((?:(?!Category:).)*)Category:\s+([^\n]+)'
        
        matches = re.findall(paper_pattern, text_content, re.DOTALL)
        
        logger.info(f"Found {len(matches)} paper pattern matches")
        
        for match in matches[:max_papers]:
            try:
                paper_num, paper_id, submission_date, authors_text, comments, abstract_text, category_text = match
                
                # Clean up the data
                authors = [author.strip() for author in authors_text.split(',') if author.strip()]
                if not authors:
                    authors = ["Unknown"]
                
                # Extract title from comments or use paper ID
                title = comments.strip() if comments.strip() else f"viXra Paper {paper_id}"
                
                # Clean abstract
                abstract = abstract_text.strip()
                if not abstract or len(abstract) < 10:
                    abstract = "Abstract not available"
                
                # Clean category
                subject = category_text.strip() if category_text else category or "unknown"
                
                # viXra PDFs have version numbers - extract from the full URL pattern
                pdf_url = f"{self.BASE_URL}/pdf/{paper_id}v1.pdf"  # Most viXra papers are v1
                
                paper = Paper(
                    id=paper_id,
                    title=title,
                    authors=authors,
                    subject=subject,
                    abstract=abstract,
                    submission_date=submission_date.strip(),
                    pdf_url=pdf_url
                )
                
                papers.append(paper)
                logger.info(f"Extracted paper: {paper_id} - {title[:50]}...")
                
            except Exception as e:
                logger.error(f"Error parsing paper match: {e}")
                continue
        
        # If regex didn't work well, try simpler approach
        if not papers:
            logger.info("Regex parsing failed, trying simpler approach...")
            
            # Look for viXra paper IDs in the text
            vixra_ids = re.findall(r'viXra:(\d{4}\.\d{4})', text_content)
            logger.info(f"Found {len(vixra_ids)} viXra IDs")
            
            for paper_id in vixra_ids[:max_papers]:
                try:
                    # Create basic paper object
                    paper = Paper(
                        id=paper_id,
                        title=f"viXra Paper {paper_id}",
                        authors=["Unknown"],
                        subject=category or "unknown",
                        abstract="Abstract extracted from viXra listing",
                        submission_date=datetime.now().strftime('%Y-%m-%d'),
                        pdf_url=f"{self.BASE_URL}/pdf/{paper_id}v1.pdf"
                    )
                    
                    papers.append(paper)
                    
                except Exception as e:
                    logger.error(f"Error creating basic paper {paper_id}: {e}")
                    continue
        
        return papers
    
    async def download_pdf(self, paper: Paper) -> bool:
        """Download PDF for a paper"""
        try:
            pdf_path = self.download_dir / f"{paper.id}.pdf"
            
            if pdf_path.exists():
                paper.pdf_path = str(pdf_path)
                logger.info(f"PDF already exists for {paper.id}")
                return True
            
            logger.info(f"Attempting to download PDF: {paper.pdf_url}")
            
            async with self.session.get(paper.pdf_url) as response:
                logger.info(f"PDF download response status: {response.status}")
                
                if response.status == 200:
                    content = await response.read()
                    logger.info(f"Downloaded {len(content)} bytes for {paper.id}")
                    
                    with open(pdf_path, 'wb') as f:
                        f.write(content)
                    
                    paper.pdf_path = str(pdf_path)
                    logger.info(f"Successfully saved PDF for {paper.id}")
                    return True
                else:
                    logger.warning(f"PDF download failed with status {response.status} for {paper.id}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error downloading PDF for {paper.id}: {e}")
            logger.error(f"PDF URL was: {paper.pdf_url}")
        
        return False