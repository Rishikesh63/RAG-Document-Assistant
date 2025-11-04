# RAG Document Assistant

**A fast, scalable document Q&A system with professional architecture patterns.**

Built this for a technical assessment - what started as a basic RAG implementation turned into a pretty solid system with design patterns, multiple response modes, and production-ready features. The focus was on creating something that actually works well and could scale in a real environment.

## What Makes This Different

**Architecture:**
- Strategy Pattern: Easy to swap between different LLM providers
- Builder Pattern: Clean, readable configuration setup
- Factory Pattern: Organized component creation
- Centralized configuration management

**Performance:**
- Fast retrieval (usually under 100ms)
- Three response modes: fast extractive, auto-switching, and high-quality LLM
- Smart caching to avoid recomputing embeddings
- Graceful fallbacks when things go wrong

## Tech Stack

### **Core Technologies**
- **Python 3.8+** - Main programming language
- **FAISS** - High-performance vector similarity search and clustering
- **Sentence Transformers** - State-of-the-art sentence and document embeddings
- **Streamlit** - Modern web UI framework for ML applications
- **PyPDF2** - PDF document processing and text extraction

### **AI & LLM Integration**
- **Groq API** - Ultra-fast LLM inference (Llama 3.1 models)
- **Hugging Face Transformers** - Pre-trained embedding models
- **all-MiniLM-L6-v2** - Lightweight, high-quality sentence embeddings

### **Architecture & Design**
- **Design Patterns** - Strategy, Builder, Factory patterns for scalability
- **Type Hints** - Full typing support for better code quality
- **Dataclasses** - Clean configuration management
- **Protocol Classes** - Interface definitions for extensibility

### **Development & Deployment**
- **dotenv** - Environment variable management
- **NumPy** - Numerical computing and array operations
- **JSON** - Data serialization and configuration storage
- **Pathlib** - Modern file path handling
- **Time & Performance** - Built-in timing and monitoring

### **Safety & Quality**
- **Regex** - Pattern matching for content filtering
- **difflib** - Text similarity for hallucination detection
- **Custom Guardrails** - Profanity filtering and content validation

### **Optional Integrations**
- **Redis** - Response caching (mentioned in scaling examples)
- **ChromaDB** - Alternative vector database (extensibility example)
- **Async/Await** - Concurrent processing (future improvement)

### **Why These Choices**

**FAISS**: Chosen for production-ready vector search with excellent performance
**Groq**: Selected for ultra-fast LLM inference (much faster than OpenAI)
**Streamlit**: Rapid UI development with built-in data visualization
**Sentence Transformers**: Industry standard for semantic embeddings
**Design Patterns**: Enterprise-grade architecture for maintainability

### **Performance Characteristics**
- **Retrieval**: ~50ms (FAISS + optimized embeddings)
- **LLM Generation**: ~1-2s (Groq API with Llama 3.1)
- **Memory Usage**: Efficient caching with global model storage
- **Scalability**: Designed for extension to cloud vector DBs

## Getting Started

### Clone the Repository
```bash
git clone https://github.com/Rishikesh63/RAG-Document-Assistant.git
cd RAG-Document-Assistant
```

### Setup
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Optional: Add your Groq API key
Create a `.env` file:
```
GROQ_API_KEY=your_key_here
```
*The system works fine without this - you'll just get extractive answers instead of LLM-generated ones.*

### Process your documents
```powershell
python build_embeddings.py --pdfs "agentstarter.pdf,Matering RAG.pdf" --out_dir data
```

### Try it out
```powershell
# Interactive demo with different response modes
python test_demo.py

# Web interface (recommended)
streamlit run rag_ui.py
```

## How to Use It

### Simple Commands
```python
from rag_pipeline import answer_query_enhanced

# Quick answers (usually under 200ms)
response = answer_query_enhanced("What is RAG?", mode="fast")

# Better quality with LLM
response = answer_query_enhanced("Explain the architecture", mode="quality")

# Let the system decide
response = answer_query_enhanced("How does it work?", mode="auto")
```

### Builder Pattern (More Control)
```python
from rag_pipeline import RAGPipelineBuilder

# Custom pipeline setup
pipeline = (RAGPipelineBuilder()
    .quality_mode()
    .with_index_dir("data")
    .build())

response = pipeline.query("Your question here")
```

### Web Interface
```powershell
streamlit run rag_ui.py
```
The web UI is actually pretty nice - has a chat interface, shows performance metrics, lets you switch between modes, and displays source citations.

## How It's Built

### Main Components

**RAGPipeline** - The main orchestrator that coordinates everything
**LLMProvider Protocol** - Makes it easy to swap between different AI providers  
**RAGConfig** - All the settings in one place
**RAGPipelineBuilder** - Clean way to set things up

### Response Modes

| Mode | When to Use | Speed | Quality |
|------|-------------|-------|---------|
| **Fast** | Demos, quick answers | <200ms | Extractive summaries |
| **Auto** | General use | Varies | Smart switching |
| **Quality** | Detailed answers | ~1s | Full LLM generation |

### Performance Notes

- Retrieval is usually under 100ms (pretty fast)
- Models and indexes are cached globally so they don't reload
- The strategy pattern makes it easy to add new LLM providers
- Everything has fallbacks so it shouldn't just crash

## Safety and Quality

**Content Filtering:**
- Basic profanity filtering (nothing fancy, just regex-based)
- Hallucination detection by checking if answers actually match the retrieved content
- Input validation to prevent weird edge cases

**Quality Checks:**
- Confidence scoring based on how well documents match the query
- Every answer includes source citations
- Performance monitoring to catch bottlenecks

## System Monitoring

The web interface shows you what's happening under the hood:

- Response time breakdown (retrieval vs generation)
- Confidence scores and content grounding analysis
- Performance suggestions when things are slow
- Configuration details for debugging

## Test Queries

I included 5 test queries that cover different scenarios:

1. **Basic Concepts**: "What is Retrieval-Augmented Generation (RAG)?"
2. **Technical Details**: "How do I set up the system for local inference?"  
3. **Feature Questions**: "What safety features are mentioned for the system?"
4. **Out of Scope**: "How do I contact customer support?"
5. **Complex Analysis**: "Does the document describe limitations or future improvements?"

Run `python test_demo.py` to try them all in different modes.

## Project Files

```
ZuduAI-RAG/
├── rag_pipeline.py          # Main implementation with design patterns
├── build_embeddings.py      # PDF processing and FAISS index creation
├── test_demo.py             # Test suite with the 5 required queries
├── rag_ui.py               # Streamlit web interface
├── guardrails.py           # Safety filters and quality checks
├── optimize_performance.py  # Performance benchmarking tools
├── requirements.txt
└── data/                   # Generated vector index and metadata
```

## Extending the System

### Adding a New LLM Provider
```python
class CustomProvider:
    def generate(self, prompt: str) -> tuple[str, Dict[str, float]]:
        # Your implementation here
        return response, timing
    
    def supports_streaming(self) -> bool:
        return True

# Register it
LLMFactory.register("custom", CustomProvider)
```

### Custom Configuration
```python
@dataclass  
class CustomRAGConfig(RAGConfig):
    custom_feature: bool = True
    
    @classmethod
    def for_enterprise(cls):
        return cls(top_k=10, use_llm=True, custom_feature=True)
```

### Adding Different Vector Databases
```python
class ChromaRetriever:
    def __init__(self, collection_name: str, persist_dir: str):
        # setup Chroma client
        pass
    
    def retrieve(self, query: str, top_k: int = 5):
        # Chroma search logic
        return retrieved_results

# Use it
pipeline = (RAGPipelineBuilder()
           .with_retriever("chroma", collection_name="docs")
           .build())
```

### Production Scaling Ideas

#### Caching
```python
from functools import lru_cache
import redis

class CacheManager:
    def __init__(self, redis_url=None):
        self.redis_client = redis.from_url(redis_url) if redis_url else None
    
    @lru_cache(maxsize=1000)
    def get_embeddings(self, text: str):
        # Cache embeddings to avoid recomputation
        pass
```

#### Async Support
```python
import asyncio

class AsyncRAGPipeline:
    async def query_async(self, question: str):
        # Async retrieval and LLM calls
        retrieved = await self.retrieve_async(question)
        response = await self.llm_provider.generate_async(prompt)
        return response
```

#### Batch Processing
```python
class BatchRAGPipeline:
    def query_batch(self, questions: List[str]) -> List[Dict]:
        # Process multiple questions efficiently
        embeddings = self.model.encode(questions, batch_size=32)
        responses = self.llm_provider.generate_batch(prompts)
        return responses
```

## What This Covers

**Core Requirements:**
- Document processing and chunking ✓
- Vector-based retrieval system ✓
- LLM integration with fallbacks ✓
- Source citations for all responses ✓
- Graceful handling of out-of-scope queries ✓
- 5 test scenarios ✓

**Design Pattern Implementation:**
- Strategy Pattern for swappable providers ✓
- Builder Pattern for clean configuration ✓
- Factory Pattern for component creation ✓
- Scalable architecture ✓

**Performance and Quality:**
- Fast mode responses under 200ms ✓
- Safety guardrails and content filtering ✓
- Production-ready error handling ✓
- Performance monitoring ✓

## Performance Numbers

| Operation | Target | What I Got |
|-----------|---------|------------|
| Index Loading | <1s | ~0.3s |
| Document Retrieval | <100ms | ~50ms |
| Fast Mode Response | <200ms | ~150ms |
| Quality Mode Response | <2s | ~1.2s |

## Design Approach

This implementation focuses on:

1. **Maintainability** - Clean separation using design patterns
2. **Extensibility** - Easy to add new providers and features  
3. **Performance** - Optimized for real usage
4. **Reliability** - Comprehensive error handling and fallbacks
5. **Usability** - Intuitive interfaces for both developers and users

The goal was to show not just that I can build a RAG system, but that I can architect it properly for real-world use.

## Limitations & Future Improvements

### Current Limitations

**Context Window Constraints:**
- Each query is processed independently - no conversation memory
- Limited by the model's context window for very long documents
- Chunking strategy is simple (fixed-size) rather than semantic

**Retrieval Scope:**
- Only searches within the loaded document set
- No real-time document updates without rebuilding the index
- Single embedding model limits multilingual support

**Response Quality:**
- Extractive mode responses can be choppy compared to full LLM generation
- No advanced reasoning chains for complex multi-step questions
- Limited fact-checking beyond basic hallucination detection

**Scalability:**
- FAISS index is in-memory, limiting document set size
- No distributed processing for very large document collections
- Single-threaded processing during index building

### Future Improvements

**Enhanced Conversation:**
```python
# Multi-turn dialogue support
class ConversationalRAG(RAGPipeline):
    def __init__(self):
        self.conversation_history = []
    
    def query_with_context(self, query: str, max_history: int = 5):
        # Include relevant conversation context
        context = self.get_relevant_history(query, max_history)
        return super().query(f"{context}\n{query}")
```

**Advanced Chunking:**
- Semantic chunking based on document structure
- Overlapping chunks with cross-references
- Dynamic chunk sizing based on content type
- Multi-level indexing (summary + detail chunks)

**Real-time Updates:**
```python
# Live document monitoring
class LiveRAGPipeline(RAGPipeline):
    def watch_directory(self, path: str):
        # Monitor for new/updated documents
        # Incremental index updates
        pass
```

**Enterprise Features:**
- User authentication and query logging
- A/B testing for different retrieval strategies  
- Analytics dashboard for system performance
- Multi-tenant support with isolated document spaces

**Advanced Retrieval:**
```python
# Hybrid search combining dense + sparse retrieval
class HybridRetriever:
    def __init__(self):
        self.dense_retriever = FAISSRetriever()
        self.sparse_retriever = BM25Retriever()
    
    def retrieve(self, query: str, alpha: float = 0.5):
        # Combine dense and sparse search results
        return self.ensemble_results(dense_results, sparse_results, alpha)
```

**Quality Improvements:**
- Chain-of-thought reasoning for complex queries
- Multi-step verification and fact-checking
- Confidence calibration and uncertainty quantification
- Source credibility scoring

**Performance Optimizations:**
- Async processing for concurrent queries
- GPU acceleration for embedding computation
- Distributed vector search with Qdrant/Weaviate
- Response caching with Redis

**Integration Capabilities:**
- Plugin system for custom document processors
- Webhook support for external system integration
- API gateway for production deployment
- Monitoring with Prometheus/Grafana

### Migration Path

The current architecture makes these improvements straightforward:

1. **Phase 1**: Replace FAISS with cloud vector DB (Qdrant/Weaviate)
2. **Phase 2**: Add conversation memory and context management
3. **Phase 3**: Implement hybrid retrieval and advanced chunking
4. **Phase 4**: Add enterprise features and monitoring

The design patterns ensure that core functionality remains stable while new features are added incrementally.

## Migration from Basic RAG

If you're upgrading from a simpler implementation:

### Quick Migration
```python
# Old way
from old_system import answer_query
response = answer_query("What is RAG?", out_dir="data", top_k=5)

# New way (same interface)
from rag_pipeline import answer_query_enhanced
response = answer_query_enhanced("What is RAG?", mode="auto")
```

### Gradual Migration
```python
# Phase 1: Basic builder usage
pipeline = RAGPipelineBuilder().fast_mode().build()
response = pipeline.query("What is RAG?")

# Phase 2: Custom configuration
config = RAGConfig(top_k=7, temperature=0.1)
pipeline = RAGPipelineBuilder().with_config(config).build()

# Phase 3: Advanced features
pipeline = (RAGPipelineBuilder()
           .quality_mode()
           .with_llm("groq", model="llama-3.1-70b")
           .with_caching()
           .build())
```

### Migration Steps

**Basic Setup (30 minutes):**
- Test `answer_query_enhanced()` with existing queries
- Verify backward compatibility
- Update import statements

**Configuration Migration (1 hour):**
- Replace parameter passing with `RAGConfig`
- Use mode presets (`fast`, `auto`, `quality`)
- Test with existing test suite

**Builder Pattern Adoption (1 hour):**
- Migrate to `RAGPipelineBuilder()` for new features
- Add custom configurations for different use cases
- Update UI to use new pipeline

**Production Features (as needed):**
- Add monitoring and metrics
- Implement caching for performance
- Consider async support for high-load scenarios
- Add custom LLM providers if needed

## Troubleshooting

### Common Questions

**Q: How do I keep my existing code working?**
A: Use `answer_query_enhanced()` - it's a drop-in replacement with the same interface.

**Q: How do I test different LLM providers?**
A: Use the builder pattern:
```python
groq_pipeline = RAGPipelineBuilder().with_llm("groq").build()
# Add other providers as needed
```

**Q: Performance is slower than expected**
A: Try these:
- Use `fast` mode for sub-200ms responses
- Enable caching for repeated queries
- Check timing metrics to see where the bottleneck is
- Consider batch processing for multiple queries

**Q: How do I add custom processing to responses?**
A: Extend the pipeline class:
```python
class CustomRAGPipeline(RAGPipeline):
    def post_process_response(self, response):
        # Add your custom logic
        return enhanced_response
```

**Q: Can I use this with my existing vector database?**
A: Yes, implement the retriever interface for your database and add it to the factory pattern.

### Performance Tips

1. **Caching**: Model/index caching is already implemented
2. **Mode Selection**: 
   - `fast` for demos (under 200ms)
   - `auto` for general use
   - `quality` for comprehensive answers
3. **Monitoring**: Use timing metrics to identify bottlenecks
4. **Batch Processing**: For multiple queries, batch the embedding computation
5. **Async**: For high-load scenarios, implement async patterns

### Testing Different Configurations
```python
configs = [
    RAGConfig.for_speed(),
    RAGConfig.for_quality(),
    RAGConfig(top_k=10, temperature=0.3)
]

results = {}
for i, config in enumerate(configs):
    pipeline = RAGPipelineBuilder().with_config(config).build()
    response = pipeline.query(test_question)
    results[f"config_{i}"] = evaluate_response(response)
```

### Production Health Checks
```python
def health_check():
    try:
        pipeline = RAGPipelineBuilder().fast_mode().build()
        response = pipeline.query("test query")
        return {
            "status": "healthy", 
            "response_time": response["timing"]["total_time"]
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}
```
