"""
Category mappings for arXiv and viXra
"""

# arXiv category mappings
ARXIV_CATEGORIES = {
    "astro-ph": "Astrophysics",
    "astro-ph.CO": "Cosmology and Nongalactic Astrophysics",
    "astro-ph.EP": "Earth and Planetary Astrophysics", 
    "astro-ph.GA": "Astrophysics of Galaxies",
    "astro-ph.HE": "High Energy Astrophysical Phenomena",
    "astro-ph.IM": "Instrumentation and Methods for Astrophysics",
    "astro-ph.SR": "Solar and Stellar Astrophysics",
    "cond-mat": "Condensed Matter Physics",
    "gr-qc": "General Relativity and Quantum Cosmology",
    "hep-ex": "High Energy Physics - Experiment",
    "hep-lat": "High Energy Physics - Lattice",
    "hep-ph": "High Energy Physics - Phenomenology", 
    "hep-th": "High Energy Physics - Theory",
    "math-ph": "Mathematical Physics",
    "nucl-ex": "Nuclear Experiment",
    "nucl-th": "Nuclear Theory",
    "physics": "Physics (Other)",
    "quant-ph": "Quantum Physics",
    "physics.atom-ph": "Atomic Physics",
    "physics.class-ph": "Classical Physics",
    "physics.comp-ph": "Computational Physics",
    "physics.gen-ph": "General Physics"
}

# viXra category mappings
VIXRA_CATEGORIES = {
    "astro": "Astrophysics",
    "atom": "Atomic and Molecular Physics", 
    "bio": "Biological Physics",
    "cond": "Condensed Matter Physics",
    "gen": "General Physics",
    "hep": "High Energy Particle Physics",
    "grav": "Quantum Gravity and String Theory",
    "rel": "Relativity and Cosmology", 
    "quant": "Quantum Physics",
    "nucl": "Nuclear Physics",
    "plasma": "Plasma Physics",
    "math": "Mathematical Physics",
    "stat": "Statistical Mechanics",
    "thermo": "Thermodynamics",
    "class": "Classical Physics"
}

# Mapping from arXiv category codes to viXra category codes
ARXIV_TO_VIXRA_CATEGORIES = {
    "astro-ph": "astro",
    "astro-ph.CO": "astro",
    "astro-ph.EP": "astro",
    "astro-ph.GA": "astro",
    "astro-ph.HE": "astro",
    "astro-ph.IM": "astro",
    "astro-ph.SR": "astro",
    "cond-mat": "cond",
    "gr-qc": "grav",
    "hep-ex": "hep",
    "hep-lat": "hep",
    "hep-ph": "hep", 
    "hep-th": "hep",
    "math-ph": "math",
    "nucl-ex": "nucl",
    "nucl-th": "nucl",
    "physics": "gen",
    "quant-ph": "quant",
    "physics.atom-ph": "atom",
    "physics.class-ph": "class",
    "physics.plasma-ph": "plasma"
}