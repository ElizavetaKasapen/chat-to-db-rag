# Context-Based Memory Management System

A multi-agent RAG (Retrieval-Augmented Generation) system that organizes information into semantic contexts and manages facts with intelligent routing, extraction, and question-answering capabilities.

## Overview

This project implements an AI-powered memory management system that:
- Automatically extracts facts from user input
- Organizes facts into semantic contexts using embeddings or summaries
- Routes queries to appropriate specialized agents
- Answers questions by searching through stored knowledge
- Manages memory updates intelligently to avoid redundancy

## Architecture

### Core Components

```
core/
├── agents.py               # Agent definitions and LLM configurations
├── workflow.py             # Main processing pipeline
├── prompts.yaml            # System prompts for all agents
├── parallel_workflow.py    # (Parallel processing support) #working bad now
└── tools/
    ├── context_representation.py  # Context embedding/summary generation
    ├── storage_manager.py         # Chroma DB abstraction layer
    ├── storage_tools.py           # LangChain tool wrappers
    └── storage_config.json        # Storage configuration
```

### Agent System

The system uses four specialized agents:

1. **Supervisor Agent**: Routes user requests to the appropriate handler
   - Determines if input contains facts to extract or questions to answer
   - Reformulates requests for clarity

2. **Fact Extractor**: Extracts structured facts from natural language
   - Parses user input into discrete factual statements
   - Formats facts as "Subject: fact" pairs

3. **Memory Manager**: Handles fact storage and updates
   - Adds new facts to existing contexts
   - Updates existing facts when similar information is found
   - Prevents duplicate storage

4. **Questions Manager**: Answers queries using stored knowledge
   - Searches contexts and facts for relevant information
   - Synthesizes responses from multiple sources

## Key Features

### Context Representation Strategies

The system supports multiple strategies for representing contexts:

- **`mean`**: Average embedding of all facts in a context
- **`summary`**: LLM-generated concise summary
- **`main_objects`**: Extracted key entities (people, places, organizations)
- **`concat`**: Concatenates facts in the context

Configure via `storage_config.json`:

```json
{
  "persist_directory": "./chroma_db_separate_context",
  "context_representation": "mean"
}
```

### Storage Architecture

Built on ChromaDB with a two-tier structure:

- **Contexts Collection**: Stores one entry per semantic context
- **Facts Collections**: One collection per context containing individual facts

This design enables:
- Fast similarity search across contexts
- Efficient fact retrieval within relevant contexts
- Automatic context regeneration when facts change

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd <project-directory>

# Install dependencies
pip install -r requirements.txt

```

## Usage

### Basic Example

```python
from core.workflow import process_user_input

# Initialize conversation
chat_history = []

# Store information
response = process_user_input(
    "John Smith is a software engineer at TechCorp. He graduated from MIT in 2015.",
    chat_history
)
print(response)
# Output: Created context '<uuid>' with 2 facts...

# Ask questions
response = process_user_input(
    "Where did John Smith go to school?",
    chat_history
)
print(response)
# Output: John Smith graduated from MIT in 2015.
```

### Customizing Prompts

All agent prompts are defined in `core/prompts.yaml`:

```yaml
supervisor: |
  You are a routing supervisor...
  
fact_extractor: |
  Extract all factual statements...
  
memory_manager: |
  You manage the knowledge base...
  
questions_manager: |
  Answer questions using available tools...
```

## API Reference

### Main Functions

#### `process_user_input(user_input: str, chat_history: list) -> str`

Processes user input through the complete workflow.

**Args:**
- `user_input`: User's text message
- `chat_history`: Last N conversation messages

**Returns:** AI-generated response string

### Storage Tools

All tools are available as LangChain `StructuredTool` objects:

- `create_context_tool`: Create new context with initial facts
- `add_fact_tool`: Add fact to existing context
- `update_fact_tool`: Modify existing fact
- `delete_fact_tool`: Remove fact from context
- `context_similarity_search_tool`: Find most similar context
- `fact_similarity_search_tool`: Search facts within a context
- `get_contexts_tool`: List all contexts
- `get_facts_in_context_tool`: Get all facts in a context

## How It Works

### Workflow Overview

1. **User Input** → Supervisor Agent
2. **Routing Decision**:
   - **Extract Facts Path**:
     - Extract facts from text
     - Generate context representation
     - Search for similar existing context
     - If found: Check for duplicate facts, add/update as needed
     - If not found: Create new context
   - **Question Path**:
     - Search relevant contexts
     - Retrieve similar facts
     - Generate answer

### Memory Management Logic

When new facts are added:

1. Generate embedding/summary of new facts
2. Search for semantically similar context (cosine similarity)
3. For each fact:
   - Retrieve top-K similar existing facts
   - Memory Manager Agent decides: add new or update existing
4. Context representation is automatically regenerated

## Advanced Features

### Context Regeneration

Contexts are automatically updated when facts change:

```python
# Adding a fact triggers regeneration
manager.add_fact(context_id, "New fact")
# Context embedding/summary is recomputed from all facts
```

## Evaluation

The system includes a **two-stage evaluation framework** to analyze its memory management performance:

### 1. Ingestion Phase
- Articles from a dataset are processed through the workflow.  
- Facts are extracted, contexts are created or updated, and memory is managed intelligently.  
- Metadata is tracked to map facts to their original articles and topics.  

### 2. Evaluation Phase
Multiple metrics are calculated to assess context quality, fact importance, and redundancy.

---

### Metrics

**Context Clustering Metrics**  
- **Purity**: Fraction of facts in a context belonging to the dominant topic.  
- **Adjusted Rand Index (ARI)**: Measures agreement between system-assigned contexts and ground-truth topics (1 = perfect clustering).  
- **Normalized Mutual Information (NMI)**: Evaluates similarity between predicted and true clusters.  
- **Context Distribution**: Number of facts per context, dominant topic, and purity per context.

**Fact Importance Metrics**  
- **Frequency**: How often similar facts appear within a context.  
- **Centrality**: Semantic closeness of a fact to its context representation.  
- **Distinctiveness (IDF-style)**: Uniqueness of a fact across all contexts.  
- **Specificity**: Approximation of informational detail based on fact length.  
- **Overall Importance**: Weighted combination of the above.

**Redundancy Metrics**  
- **Pairwise Similarity**: Cosine similarity between embeddings of facts within the same context.  
- **High Redundancy Pairs**: Number of fact pairs with similarity > 0.85.  
- **Average / Max / Std Similarity**: General measure of redundancy within contexts.

### Evaluation part usage
Run 
```bash
python evaluation.py /path/to/dataset --output-dir ./results
```

Expected dataset structure:
```bash
dataset/
├── topic1/
│   ├── article1.txt
│   └── article2.txt
└── topic2/
    └── article1.txt
```
---

### High-Level Summary of Analysis

Across **4 context representation strategies** (`mean`, `summary`, `main_objects`, `concat`):

**Mean**  
- Perfect clustering: ARI=1, NMI=1  → matches ground-truth topics  
- Low redundancy, fewer but cleaner facts  
- Fast and semantically stable  
- **Recommendation**: Best for clean topic clustering and retrieval  

**Main Objects**  
- Generates the most facts → highest informational recall  
- Slightly fragmented clusters, higher redundancy 
- ARI/NMI good but not perfect  
- Computationally expensive  
- **Recommendation**: Use for maximum fact coverage, not clustering  

**Summary / Concatenation**  
- Noisy, produce more contexts (over-clustering)  
- Higher redundancy, worse ARI (0.59–0.86)  
- Summary captures more specific language (higher IDF) but less consistency  
- **Recommendation**: Use if more detailed retrieval is needed, but not for clustering  

**Concatenation**  
- Unstable clusters 
- ARI/NMI poor and inconsistent  
- Moderate redundancy  
- **Recommendation**: Only as a quick baseline; suboptimal for production  

---

### Comparison table

| Representation  | Best For                    | Quality | Notes                                    |
|-----------------|-----------------------------|---------|------------------------------------------|
| **MEAN**        | Topic grouping & clustering | 5       | Stable, perfect ARI/NMI, low redundancy  |
| Main Objects    | Maximum fact recall         | 4       | Good clustering, high redundancy         |
| Summary         | Detailed retrieval          | 3       | More clusters, moderate NMI, verbose     |
| Concatenation   | Quick baseline              | 2       | Noisy clusters, unstable                 |

**Final Conclusion:**  
For **stable topic clustering** and **minimal redundancy**, use **Mean**.  
For **maximum fact extraction**, use **Main Objects**.  
Summary can be used when moderate detail is desired. Concatenation is generally suboptimal.


## Future Enhancements

- [ ] Add fact versioning and history tracking
- [ ] Implement context merging for related topics
- [ ] Support for structured data (tables, lists)
- [ ] Web interface with persistent sessions
- [ ] Export/import functionality for knowledge bases
- [ ] Confidence scoring for fact updates vs. additions

