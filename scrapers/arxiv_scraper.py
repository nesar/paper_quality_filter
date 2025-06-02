"""
Scraper for arXiv papers
"""
import logging
import urllib.parse
import xml.etree.ElementTree as ET
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import aiohttp

from models.paper import Paper
from utils.categories import ARXIV_CATEGORIES

logger = logging.getLogger(__name__)

class ArXivScraper:
    """Handles scraping and downloading papers from arXiv with category support"""
    
    BASE_URL = "http://export.arxiv.org/api/query"
    PDF_BASE_URL = "https://arxiv.org/pdf"
    
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
    
    def construct_query_url(self, category: Optional[str] = None, max_papers: int = 100) -> str:
        """Construct arXiv API query URL"""
        if category and category in ARXIV_CATEGORIES:
            search_query = f"cat:{category}"
        elif category:
            # Try direct category input
            search_query = f"cat:{category}"
        else:
            # Default to physics categories
            physics_cats = ["astro-ph", "gr-qc", "hep-th", "hep-ph", "quant-ph"]
            search_query = " OR ".join([f"cat:{cat}" for cat in physics_cats])
        
        # Sort by submission date (most recent first)
        params = {
            'search_query': search_query,
            'start': 0,
            'max_results': max_papers,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        url = f"{self.BASE_URL}?" + urllib.parse.urlencode(params)
        return url
    
    async def get_recent_papers(self, 
                              category: Optional[str] = None,
                              days_back: int = 30,
                              max_papers: int = 100) -> List[Paper]:
        """Scrape recent papers from arXiv using the API"""
        papers = []
        
        if category:
            logger.info(f"Searching in category: {ARXIV_CATEGORIES.get(category, category)}")
        
        query_url = self.construct_query_url(category, max_papers)
        logger.info(f"Querying arXiv API: {query_url}")
        
        try:
            async with self.session.get(query_url) as response:
                logger.info(f"Response status: {response.status}")
                
                if response.status != 200:
                    logger.error(f"Failed: HTTP {response.status}")
                    return papers
                
                content = await response.text()
                logger.info(f"Content length: {len(content)} characters")
                
                papers = await self._parse_arxiv_xml(content, max_papers, category)
                
        except Exception as e:
            logger.error(f"Error querying arXiv API: {e}")
        
        logger.info(f"Total papers collected: {len(papers)}")
        return papers
    
    async def _parse_arxiv_xml(self, xml_content: str, max_papers: int, category: Optional[str] = None) -> List[Paper]:
        """Parse arXiv API XML response"""
        papers = []
        
        try:
            # Parse the XML
            root = ET.fromstring(xml_content)
            
            # arXiv API uses Atom namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom',
                  'arxiv': 'http://arxiv.org/schemas/atom'}
            
            entries = root.findall('atom:entry', ns)
            logger.info(f"Found {len(entries)} entries in XML")
            
            for entry in entries[:max_papers]:
                try:
                    # Extract paper ID
                    id_elem = entry.find('atom:id', ns)
                    if id_elem is None:
                        continue
                    
                    full_id = id_elem.text
                    paper_id = full_id.split('/')[-1]  # Extract ID from URL
                    
                    # Extract title
                    title_elem = entry.find('atom:title', ns)
                    title = title_elem.text.strip() if title_elem is not None else f"arXiv Paper {paper_id}"
                    
                    # Extract authors
                    authors = []
                    author_elems = entry.findall('atom:author', ns)
                    for author_elem in author_elems:
                        name_elem = author_elem.find('atom:name', ns)
                        if name_elem is not None:
                            authors.append(name_elem.text.strip())
                    
                    if not authors:
                        authors = ["Unknown"]
                    
                    # Extract abstract
                    summary_elem = entry.find('atom:summary', ns)
                    abstract = summary_elem.text.strip() if summary_elem is not None else "Abstract not available"
                    
                    # Extract submission date
                    published_elem = entry.find('atom:published', ns)
                    submission_date = published_elem.text if published_elem is not None else datetime.now().isoformat()
                    
                    # Extract category/subject
                    category_elems = entry.findall('atom:category', ns)
                    subject = category or "unknown"
                    if category_elems:
                        primary_cat = category_elems[0].get('term', 'unknown')
                        subject = ARXIV_CATEGORIES.get(primary_cat, primary_cat)
                    
                    # Construct PDF URL
                    pdf_url = f"{self.PDF_BASE_URL}/{paper_id}.pdf"
                    
                    paper = Paper(
                        id=paper_id,
                        title=title,
                        authors=authors,
                        subject=subject,
                        abstract=abstract,
                        submission_date=submission_date,
                        pdf_url=pdf_url
                    )
                    
                    papers.append(paper)
                    logger.info(f"Extracted paper: {paper_id} - {title[:50]}...")
                    
                except Exception as e:
                    logger.error(f"Error parsing entry: {e}")
                    continue
                    
        except Exception as e:
            logger.error(f"Error parsing XML: {e}")
        
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