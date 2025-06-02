"""
Scrapers package for scientific paper repositories
"""
from scrapers.arxiv_scraper import ArXivScraper
from scrapers.vixra_scraper import ViXraScraper

__all__ = ['ArXivScraper', 'ViXraScraper']