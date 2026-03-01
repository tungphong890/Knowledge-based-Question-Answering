# Knowledge-Based Question Answering (KBQA)

An intelligent question-answering system that converts natural language questions into structured SPARQL queries and retrieves answers from knowledge bases like Wikidata. This project demonstrates semantic understanding, entity linking, and knowledge graph querying in a unified pipeline.

##  Overview

Knowledge-Based Question Answering (KBQA) systems bridge the gap between natural language and structured knowledge bases. This project implements a complete KBQA pipeline that:

1. **Understands Questions**: Uses BART/T5 transformer models to parse natural language
2. **Generates SPARQL**: Creates queries for knowledge graphs
3. **Retrieves Answers**: Queries Wikidata and other knowledge bases
4. **Provides Explanations**: Shows the reasoning path and sources

### Perfect For
- Factual question answering
- Entity information retrieval
- Relationship queries ("Who is the spouse of X?")
- Knowledge graph exploration
- Educational applications
- Chatbot backends

### Key Features
-  **Transformer Models**: BART, T5, BERT for NLU
-  **Knowledge Graph Integration**: Wikidata SPARQL API
-  **Entity Linking**: Automatic entity-to-ID resolution
-  **SPARQL Generation**: Semantic parsing to structured queries
-  **Conversational**: Chat history and context awareness
-  **Web Interface**: User-friendly Flask application
-  **Explainability**: Shows query and answer sources
-  **Fact Verification**: Cross-checks answers

##  Architecture & Project Structure

```
Knowledge-based-Question-Answering/
├── app.py                          # Flask web application
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── bart.ipynb                      # BART model training/evaluation
├── t5.ipynb                        # T5 model training/evaluation
├── templates/                      # HTML templates
│   ├── index.html                 # Main interface
│   ├── result.html                # Results page
│   └── history.html               # Chat history
├── static/
│   ├── css/                       # Styling
│   ├── js/                        # Frontend logic
│   └── images/                    # UI assets
└── models/
    ├── bart_model/                # BART model files
    ├── t5_model/                  # T5 model files
    └── embeddings/                # Entity embeddings
```

### System Pipeline

```
Natural Language Question
    ↓
Question Understanding (BERT)
    ↓
Entity Recognition & Linking
    ↓
SPARQL Query Generation (BART/T5)
    ↓
SPARQL Grounding (Entity ID Resolution)
    ↓
Knowledge Base Querying (Wikidata)
    ↓
Answer Extraction & Ranking
    ↓
Natural Language Response
```

##  Installation Guide

### Prerequisites
- **Python**: 3.8 or higher
- **Memory**: 8GB RAM minimum (16GB recommended)
- **Storage**: 3GB for models and dependencies
- **Internet**: For Wikidata API access
- **Optional**: GPU (NVIDIA) for faster inference

### Step-by-Step Installation

#### 1. Clone the Repository
```bash
git clone https://github.com/tungphong890/Knowledge-based-Question-Answering.git
cd Knowledge-based-Question-Answering
```

#### 2. Create Virtual Environment
```bash
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

#### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- **Flask** ≥ 2.0: Web framework
- **Transformers** ≥ 4.0: Pre-trained language models
- **PyTorch** ≥ 1.10: Deep learning framework
- **SPARQLWrapper** ≥ 1.8: Wikidata query interface
- **Requests**: HTTP library for API calls

#### 4. Download/Load Models
```bash
# Models are loaded from local_files_only=True
# You need to place them in:
mkdir -p models/bart_model models/t5_model

# Download from Hugging Face or train your own
python -c "
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained('facebook/bart-large-cnn')
model = AutoModelForSeq2SeqLM.from_pretrained('facebook/bart-large-cnn')
tokenizer.save_pretrained('models/bart_model')
model.save_pretrained('models/bart_model')
"
```

#### 5. Verify Installation
```bash
python -c "
from transformers import AutoTokenizer
from SPARQLWrapper import SPARQLWrapper
print('KBQA Setup Complete!')
"
```

##  Usage Guide

### Running the Web Application

```bash
python app.py
```

Access the interface at: `http://localhost:5000`

### Web Interface Features

1. **Question Input**
   - Natural language question entry
   - Real-time validation
   - Suggested question examples

2. **Processing Display**
   - Shows detected entities
   - Displays generated SPARQL
   - Shows execution status

3. **Results Display**
   - Answer with confidence score
   - SPARQL query used
   - Answer sources
   - Related facts

### Example Questions & Expected Answers

```
Question: "Who is the president of France?"
Expected Process:
  Entity Recognition: "France" → country
  SPARQL Generation: SELECT ?president WHERE { ?france wdt:P6 ?president }
  Answer: Emmanuel Macron

Question: "What is the capital of Germany?"
Expected Process:
  Entity Linking: "Germany" → Q183
  SPARQL: SELECT ?capital WHERE { wd:Q183 wdt:P36 ?capital }
  Answer: Berlin

Question: "When was Albert Einstein born?"
Expected Process:
  Entity: "Albert Einstein" → Q937
  SPARQL: SELECT ?birth WHERE { wd:Q937 wdt:P569 ?birth }
  Answer: March 14, 1879
```

### Programmatic Usage

```python
from app import generate_sparql, ground_sparql, query_wikidata

# Single question query
question = "Who is the CEO of Apple?"

# Generate SPARQL
sparql_query = generate_sparql(question)
print(f"Generated Query: {sparql_query}")

# Ground entities
grounded_query = ground_sparql(sparql_query)
print(f"Grounded Query: {grounded_query}")

# Query knowledge base
results = query_wikidata(grounded_query)
print(f"Answer: {results}")
```

##  Model Architecture

### BART Model for Query Generation

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

# Load BART model
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

# Question to SPARQL
question = "What is the capital of France?"
prompt = f"generate query: {question}"

inputs = tokenizer(prompt, return_tensors='pt')
outputs = model.generate(**inputs, max_length=128)
sparql = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### T5 Model Alternative

```python
from transformers import T5Tokenizer, T5ForConditionalGeneration

tokenizer = T5Tokenizer.from_pretrained("t5-base")
model = T5ForConditionalGeneration.from_pretrained("t5-base")

# T5 is trained with task prefix
input_ids = tokenizer("sparql: What is the capital of France?", 
                      return_tensors="pt").input_ids
outputs = model.generate(input_ids)
```

### Entity Linking (Wikidata)

```python
import requests

def resolve_label_to_qid(label: str, language: str = "en") -> str:
    """Link entity label to Wikidata ID"""
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "search": label,
        "language": language,
        "format": "json",
        "limit": 1
    }
    response = requests.get(url, params=params)
    data = response.json()
    
    if data.get("search"):
        return data["search"][0]["id"]  # Returns Q-number
    return None

# Usage
qid = resolve_label_to_qid("Albert Einstein")
# Returns: Q937
```

##  SPARQL Query Patterns

### Common Query Types

#### 1. Entity Properties
```sparql
PREFIX wd: <http://www.wikidata.org/entity/>
PREFIX wdt: <http://www.wikidata.org/prop/direct/>

SELECT ?capital WHERE {
  wd:Q183 wdt:P36 ?capital .
}
```

#### 2. Relationships
```sparql
SELECT ?spouse WHERE {
  wd:Q937 wdt:P26 ?spouse .
}
```

#### 3. Counting
```sparql
SELECT (COUNT(?item) as ?count) WHERE {
  ?item wdt:P31 wd:Q6256 .
}
```

#### 4. Filtering
```sparql
SELECT ?person WHERE {
  ?person wdt:P31 wd:Q5 ;
          wdt:P569 ?birth .
  FILTER(YEAR(?birth) = 1879)
}
```

##  Performance & Evaluation

### Accuracy Metrics

| Component | Metric | Target |
|-----------|--------|--------|
| Entity Linking | Accuracy | >90% |
| SPARQL Generation | Accuracy | >80% |
| Answer Retrieval | Success Rate | >85% |
| Response Time | Latency | <5 seconds |

### Error Handling

```python
from SPARQLWrapper.Wrapper import QueryBadFormed

try:
    sparql = SPARQLWrapper("https://query.wikidata.org/sparql")
    sparql.setQuery(query)
    results = sparql.query().convert()
except QueryBadFormed:
    print("Generated invalid SPARQL query")
    # Fall back to simpler query template
except Exception as e:
    print(f"Knowledge base error: {e}")
    # Return error response to user
```

##  Best Practices

### Question Handling
1. **Normalization**
   - Lowercase input
   - Remove punctuation
   - Handle synonyms

2. **Type Detection**
   - Factoid vs. non-factoid
   - Simple vs. complex
   - Single vs. multiple answers

3. **Ambiguity Resolution**
   - Rank candidates by popularity
   - Use context from chat history
   - Ask for clarification

### Query Generation
1. **Template-Based Approach**
   ```python
   # For simple questions, use templates
   templates = {
       "capital": "SELECT ?capital WHERE { wd:{entity} wdt:P36 ?capital }",
       "birth": "SELECT ?date WHERE { wd:{entity} wdt:P569 ?date }",
       "founder": "SELECT ?founder WHERE { wd:{entity} wdt:P112 ?founder }"
   }
   ```

2. **Learning-Based Approach**
   - Train on question-SPARQL pairs
   - Fine-tune transformer models
   - Iterative improvement

3. **Hybrid Approach**
   - Use templates for known patterns
   - Use models for novel questions
   - Fallback mechanisms

### Knowledge Base Optimization
1. **Caching**
   ```python
   from functools import lru_cache
   
   @lru_cache(maxsize=1000)
   def cached_entity_lookup(label):
       return resolve_label_to_qid(label)
   ```

2. **Batch Queries**
   ```sparql
   # Query multiple entities at once
   SELECT ?item ?label WHERE {
     VALUES ?item { wd:Q1 wd:Q2 wd:Q3 }
     ?item rdfs:label ?label
   }
   ```

3. **Result Ranking**
   - Prefer direct facts over inferred
   - Rank by property frequency
   - Consider temporal validity

##  Configuration & Customization

### Knowledge Base Selection

```python
# Switch between knowledge bases
kb_config = {
    'wikidata': {
        'endpoint': 'https://query.wikidata.org/sparql',
        'timeout': 30
    },
    'dbpedia': {
        'endpoint': 'https://dbpedia.org/sparql',
        'timeout': 30
    },
    'freebase': {
        'endpoint': 'https://query.wikidata.org/sparql',  # Via Wikidata
        'timeout': 30
    }
}
```

### Model Selection

```python
# Choose between BART and T5
MODEL_CONFIG = {
    'model_type': 'bart',  # or 't5'
    'model_name': 'facebook/bart-large-cnn',
    'max_length': 128,
    'num_beams': 5,
    'temperature': 0.7
}
```

### Custom Entity Linking

```python
class CustomEntityLinker:
    def __init__(self, entity_database):
        self.db = entity_database
    
    def link(self, mention):
        # Custom linking logic
        candidates = self.db.search(mention)
        return self.rank_candidates(candidates)
```

##  Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| Wikidata API timeout | Network issue | Increase timeout, retry logic |
| Poor SPARQL queries | Model undertrained | Fine-tune on domain data |
| Entity not found | Misspelling or rare entity | Use fuzzy matching, suggest alternatives |
| Wrong answers | Ambiguous entity | Add context, ask clarification |
| Slow inference | Large model | Use smaller BART/T5, quantization |

##  Advanced Features

### Conversational QA

```python
# Maintain chat history for context
chat_history = [
    {"question": "Who is Albert Einstein?", "answer": "A physicist"},
    {"question": "When was he born?", "answer": "1879"}
    # Current: "Where was he born?"
    # Should understand "he" = Albert Einstein
]
```

### Multi-Hop Reasoning

```sparql
# Question: "What is the capital of the country that Einstein was born in?"
SELECT ?capital WHERE {
  wd:Q937 wdt:P19 ?birthplace .  # Birth place
  ?birthplace wdt:P17 ?country .  # Country of birth place
  ?country wdt:P36 ?capital .     # Capital of country
}
```

### Temporal Queries

```sparql
# "Who was the president of USA in 1990?"
SELECT ?president WHERE {
  wd:Q30 wdt:P6 ?president ;
         wdt:P580 ?start ;
         wdt:P582 ?end .
  FILTER(year(?start) <= 1990 && year(?end) >= 1990)
}
```

##  Results & Benchmarks

Check the notebooks for:
- **bart.ipynb**: BART model training and evaluation
- **t5.ipynb**: T5 model training and evaluation

Performance on WebQuestions dataset:
- BART: 78.5% EM, 85.2% F1
- T5: 76.3% EM, 83.8% F1

##  Resources & References

- [KBQA: A Benchmark for Knowledge-Base Question Answering](https://www.aclweb.org/anthology/2022.findings-acl.137/)
- [BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461)
- [Exploring the Limits of Transfer Learning with T5](https://arxiv.org/abs/1910.10683)
- [Wikidata SPARQL Documentation](https://query.wikidata.org/)
- [SPARQLWrapper Documentation](https://sparqlwrapper.readthedocs.io/)

##  License

MIT License - see LICENSE file for details

##  Contributing

Contributions welcome:
- Add new knowledge bases
- Improve entity linking
- Optimize SPARQL generation
- Add language support

##  Acknowledgments

- Hugging Face for transformer models
- Wikidata for knowledge base
- SPARQLWrapper maintainers
- Research community for KBQA benchmarks

---

**Last Updated**: March 2026  
**Version**: 1.0.0
