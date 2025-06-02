# Enhanced Paper Quality Filter

A comprehensive system for identifying papers with subtle physics issues from arXiv or viXra repositories, with advanced capabilities for building reasoning benchmarks and training datasets for LLM development.

## Features

### Core Analysis
- Scrape physics papers from arXiv or viXra
- Focus on papers that show sophistication but may contain errors
- Three-stage analysis using LLM:
  1. Filter out obviously flawed papers
  2. Analyze subtle physics issues
  3. Evaluate whether errors are interesting or educational
- Support for both OpenAI and Google Gemini models

### Enhanced Capabilities
- **Analysis Storage**: Individual JSON files for each paper's complete analysis
- **Reasoning Benchmark Creation**: Automated generation of physics reasoning tests
- **Training Data Extraction**: Step-by-step derivations for reinforcement learning
- **Comprehensive Documentation**: Detailed metadata and source attribution

## Project Structure

```
paper_quality_filter/
├── enhanced_paper_analyser.py  # Enhanced main entry point
├── paper_analyser.py           # Original entry point
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

### Enhanced Analysis (Recommended)

The enhanced version provides all original functionality plus benchmark creation and training data extraction:

```bash
# Full enhanced analysis with all features
python enhanced_paper_analyser.py \
    --source vixra \
    --provider gemini \
    --gemini-key $GEMINI_API_KEY \
    --category astro-ph \
    --max-papers 10 \
    --output-dir ./physics_analysis

# Create reasoning benchmark from arXiv papers
python enhanced_paper_analyser.py \
    --source arxiv \
    --provider openai \
    --openai-key $OPENAI_API_KEY \
    --category quant-ph \
    --max-papers 20 \
    --create-benchmark

# Extract training data for reinforcement learning
python enhanced_paper_analyser.py \
    --source vixra \
    --provider gemini \
    --gemini-key $GEMINI_API_KEY \
    --category rel \
    --max-papers 25 \
    --create-training
```

### Original Analysis

For basic analysis without enhanced features:

```bash
# Analyze arXiv papers (original functionality)
python paper_analyser.py \
    --source arxiv \
    --provider gemini \
    --gemini-key $GEMINI_API_KEY \
    --category astro-ph \
    --max-papers 10

# Analyze viXra papers (original functionality)  
python paper_analyser.py \
    --source vixra \
    --provider gemini \
    --gemini-key $GEMINI_API_KEY \
    --category astro \
    --max-papers 10
```

### Command Line Arguments

#### Enhanced Analyser
- `--source`: Repository to scrape (arxiv or vixra)
- `--provider`: LLM provider (openai or gemini)
- `--openai-key`: OpenAI API key (required for OpenAI provider)
- `--gemini-key`: Google Gemini API key (required for Gemini provider)
- `--category`: Category to process (use --list-categories to see options)
- `--days-back`: Days back to scrape (default: 30)
- `--max-papers`: Maximum papers to process (default: 5)
- `--output-dir`: Base directory for all outputs (default: ./enhanced_analysis)
- `--download-dir`: Directory to download papers to (default: ./papers_[source])
- `--depth`: Analysis depth (basic, technical, full, force)
- `--create-benchmark`: Create reasoning benchmark (default: True)
- `--create-training`: Extract training data (default: True)
- `--list-categories`: List available categories and exit
- `--debug`: Enable debug logging

#### List Categories
```bash
python enhanced_paper_analyser.py --source arxiv --list-categories
python enhanced_paper_analyser.py --source vixra --list-categories
```

## Output Structure

The enhanced analyser creates organized output directories:

```
enhanced_analysis_YYYYMMDD_HHMMSS/
├── analysis_results/
│   ├── analysis_paper1.json      # Individual paper analysis
│   ├── analysis_paper2.json      # Complete assessment data
│   └── ...
├── benchmark/
│   └── physics_reasoning_benchmark_YYYYMMDD_HHMMSS.json
├── training_data/
│   └── physics_training_data_YYYYMMDD_HHMMSS.json
└── analysis_summary.json         # Overall summary and statistics
```

### Output Data Formats

#### Individual Analysis Files
Each paper gets a detailed JSON file containing:
- Complete paper metadata (title, authors, abstract, etc.)
- Full assessment results (scores, issues, recommendations)
- Processing timestamps and text samples
- Technical analysis details

#### Reasoning Benchmark
Structured benchmark questions including:
- **General Analysis**: Overall physics reasoning evaluation
- **Mathematical Analysis**: Equation consistency and derivation errors  
- **Assumption Analysis**: Physics assumptions and approximations
- **Reasoning Chain**: Step-by-step logical progression
- Ground truth answers from expert LLM analysis

#### Training Data
Step-by-step physics derivations formatted for RL training:
- Problem statements and solution steps
- Difficulty levels (introductory/intermediate/advanced)
- Topic classification (mechanics, electromagnetism, quantum, etc.)
- Prerequisite knowledge requirements
- Physics concept annotations

## How It Works

### Original Three-Stage Analysis
1. **Scraping**: The system scrapes papers from arXiv or viXra based on the specified category
2. **PDF Processing**: Downloads and extracts text from PDFs using multiple extraction methods
3. **LLM Analysis**: Papers analyzed through sophisticated three-stage process:
   - **Stage 1**: Filter out obvious crackpot papers while keeping sophisticated attempts
   - **Stage 2**: Analyze for subtle physics issues (mathematical errors, physics assumptions, logical inconsistencies, literature integration problems)
   - **Stage 3**: Evaluate whether the errors are interesting, educational, or just boring mistakes
4. **Scoring**: Papers scored based on sophistication, technical issues, and educational value

### Enhanced Capabilities

#### Benchmark Creation
- Extracts mathematical equations and derivations from papers
- Identifies physics assumptions and approximations  
- Creates targeted questions testing different reasoning aspects
- Provides ground truth based on expert LLM analysis
- Enables systematic evaluation of LLM physics reasoning

#### Training Data Extraction  
- Identifies worked examples, derivations, and proofs in papers
- Segments solutions into step-by-step reasoning chains
- Normalizes mathematical notation and units
- Annotates with difficulty levels and topic classifications
- Preserves source attribution for all examples

## Example Output

### Analysis Summary
```
Total papers analyzed: 10
Papers with interesting issues: 6
Individual analysis files created: 10
Benchmark questions created: 24
Training examples extracted: 45
All outputs saved to: enhanced_analysis_20240115_103000
```

### Most Interesting Papers
```
1. Novel Approach to Dark Matter Detection via Quantum Entanglement
   Score: 0.85
   Type: SOPHISTICATED_ERROR
   Key Issues: Dimensional inconsistency, Invalid approximation, Misunderstood symmetries

2. Unified Field Theory Based on Modified Einstein Equations  
   Score: 0.72
   Type: EDUCATIONAL_FAILURE
   Key Issues: Circular reasoning, Inappropriate gauge choice
```

## Use Cases

### For Physics Researchers
- Identify papers with sophisticated but flawed reasoning
- Study common patterns in physics misconceptions
- Find educational examples of subtle errors

### For ML/AI Developers  
- Train reasoning models on physics problem-solving
- Benchmark LLM understanding of physics concepts
- Develop domain-specific evaluation datasets

### For Educators
- Create teaching materials highlighting common errors
- Develop problem sets with realistic difficulty progression
- Find examples of sophisticated reasoning gone wrong

## Dependencies

- Python 3.8+
- aiohttp: For asynchronous HTTP requests
- Beautiful Soup 4: For HTML parsing  
- PyPDF2 and pdfplumber: For PDF text extraction
- OpenAI API or Google Generative AI: For LLM analysis

See requirements.txt for the complete list of dependencies.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

MIT License - see LICENSE file for details

## Citation

If you use this tool in your research, please cite:

```bibtex
@software{enhanced_paper_quality_filter,
  title={Enhanced Paper Quality Filter: Automated Analysis and Benchmark Creation for Physics Papers},
  author={Nesar Ramachandra},
  year={2025},
  url={https://github.com/nesar/paper_quality_filter}
}
```