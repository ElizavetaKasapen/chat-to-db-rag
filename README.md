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

## Limitations

- Requires local Ollama installation for LLM inference
- GPU recommended for embedding generation
- Context regeneration can be slow with many facts
- No built-in authentication or multi-user support

## Future Enhancements

- [ ] Add fact versioning and history tracking
- [ ] Implement context merging for related topics
- [ ] Support for structured data (tables, lists)
- [ ] Web interface with persistent sessions
- [ ] Export/import functionality for knowledge bases
- [ ] Confidence scoring for fact updates vs. additions

## Contributing

Contributions welcome! Please ensure:
- Code follows existing style conventions
- New agents include prompts in `prompts.yaml`
- Storage operations maintain context consistency
- Tests cover core functionality