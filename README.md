# Paper Quality Filter

A modular system for identifying papers with subtle physics issues from arXiv or viXra repositories.

## Features

- Scrape physics papers from arXiv or viXra
- Focus on papers that show sophistication but may contain errors
- Three-stage analysis using LLM:
  1. Filter out obviously flawed papers
  2. Analyze subtle physics issues
  3. Evaluate whether errors are interesting or educational
- Support for both OpenAI and Google Gemini models

## Project Structure

```
paper_quality_filter/
├── paper_analyser.py           # Main entry point
├── scrapers/                   # Paper scraping modules
│   ├── __init__.py
│   ├── arxiv_scraper.py        # arXiv scraper
│   └── vixra_scraper.py        # viXra scraper
├── analysis/                   # Analysis modules
│   ├── __init__.py
│   ├── pdf_processor.py        # PDF text extraction
│   └── classifier.py           # LLM-based classification
├── prompts/                    # LLM prompt templates
│   ├── __init__.py
│   └── classifier_prompts.py   # Classification prompts
├── models/                     # Data models
│   ├── __init__.py
│   └── paper.py                # Paper and QualityAssessment classes
└── utils/                      # Utility functions
    ├── __init__.py
    └── categories.py           # Category mappings
```

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/paper_quality_filter.git
cd paper_quality_filter

# Install required packages
pip install -r requirements.txt
```

## Usage

```bash
# Analyze arXiv papers
python paper_analyser.py --source arxiv --provider gemini --gemini-key $GEMINI_API_KEY --category astro-ph --max-papers 10

# Analyze viXra papers
python paper_analyser.py --source vixra --provider gemini --gemini-key $GEMINI_API_KEY --category astro --max-papers 10

# List available categories for a source
python paper_analyser.py --source arxiv --list-categories
python paper_analyser.py --source vixra --list-categories
```

### Command Line Arguments

- `--source`: Repository to scrape (arxiv or vixra)
- `--provider`: LLM provider (openai or gemini)
- `--openai-key`: OpenAI API key (required for OpenAI provider)
- `--gemini-key`: Google Gemini API key (required for Gemini provider)
- `--category`: Category to process (use --list-categories to see options)
- `--days-back`: Days back to scrape (default: 30)
- `--max-papers`: Maximum papers to process (default: 5)
- `--download-dir`: Directory to download papers to (default: ./papers_[source])
- `--depth`: Analysis depth (basic=stage 1 only, technical=stages 1-2, full=all stages, force=analyze all papers)
- `--list-categories`: List available categories for the selected source and exit

## Example Output

The script will output detailed information about each paper analyzed, including:

- Paper metadata (title, authors, abstract)
- Sophistication score (0.0-1.0)
- Technical scores in various categories
- Subtle issues identified in the paper
- Overall recommendation (EDUCATIONAL_FAILURE, SOPHISTICATED_ERROR, CREATIVE_APPROACH, etc.)
- Detailed reasoning for the assessment

At the end, it provides a summary of the most interesting papers found.

## How It Works

1. **Scraping**: The system scrapes papers from arXiv or viXra based on the specified category.
2. **PDF Processing**: It downloads and extracts text from the PDFs.
3. **LLM Analysis**: The papers are analyzed using a three-stage process:
   - **Stage 1**: Filter out obvious crackpot papers while keeping sophisticated attempts.
   - **Stage 2**: Analyze for subtle physics issues, such as mathematical errors, physics assumption issues, logical inconsistencies, and literature integration problems.
   - **Stage 3**: Evaluate whether the errors are interesting, educational, or just boring mistakes.
4. **Scoring**: Papers are scored based on their sophistication, technical issues, and educational value.

## Dependencies

- Python 3.8+
- aiohttp: For asynchronous HTTP requests
- Beautiful Soup 4: For HTML parsing
- PyPDF2 and pdfplumber: For PDF text extraction
- OpenAI API or Google Generative AI: For LLM analysis

See requirements.txt for the complete list of dependencies.

## License

MIT