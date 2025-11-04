"""
Design Pattern-Based RAG Pipeline

A scalable, production-ready RAG implementation using design patterns.
This is the main RAG implementation that replaces all other versions.

Key Design Patterns:
- Strategy Pattern: Swappable LLM providers (Groq, extractive, future providers)
- Builder Pattern: Fluent configuration and pipeline creation
- Factory Pattern: Component creation and dependency injection
- Configuration Pattern: Centralized settings management
"""

from dataclasses import dataclass
from typing import Dict, List, Any, Protocol
import time
import os
import json
import numpy as np
from pathlib import Path

# Core dependencies
from sentence_transformers import SentenceTransformer
import faiss  # type: ignore

# Try to import optional dependencies
try:
    import groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    print("⚠️  Groq not installed. Install with: pip install groq")

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Global caches for efficiency
_MODEL_CACHE: Dict[str, SentenceTransformer] = {}
_INDEX_CACHE: Dict[str, tuple] = {}


def l2_normalize(x: np.ndarray) -> np.ndarray:
    """L2 normalize embeddings for cosine similarity."""
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return x / norms


def get_cached_model(model_name: str) -> SentenceTransformer:
    """Get cached embedding model or load it if not cached."""
    if model_name not in _MODEL_CACHE:
        start_time = time.time()
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
        load_time = time.time() - start_time
        print(f"Loaded model {model_name} in {load_time:.3f}s")
    return _MODEL_CACHE[model_name]


def get_cached_index(index_dir: str) -> tuple:
    """Get cached FAISS index or load it if not cached."""
    if index_dir not in _INDEX_CACHE:
        start_time = time.time()
        
        # Load FAISS index
        index_path = Path(index_dir) / "faiss.index"
        index = faiss.read_index(str(index_path))
        
        # Load metadata
        metadata_path = Path(index_dir) / "metadata.json"
        with open(metadata_path, 'r', encoding='utf-8') as f:
            metadata = json.load(f)
        
        # Load chunks lookup
        chunks_path = Path(index_dir) / "chunks.jsonl"
        chunks_lookup = {}
        with open(chunks_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                chunk_data = json.loads(line.strip())
                chunks_lookup[line_num] = chunk_data
        
        _INDEX_CACHE[index_dir] = (index, metadata, chunks_lookup)
        load_time = time.time() - start_time
        print(f"Loaded index from {index_dir} in {load_time:.3f}s")
    
    return _INDEX_CACHE[index_dir]


def retrieve_documents(index: Any, metadata: Dict, model: SentenceTransformer, 
                      query: str, top_k: int = 5) -> tuple:
    """Retrieve most relevant documents for a query."""
    start_time = time.time()
    
    # Encode query
    query_embedding = model.encode([query])
    query_embedding = l2_normalize(query_embedding.astype(np.float32))
    
    # Search FAISS index
    scores, indices = index.search(query_embedding, top_k)
    
    # Format results
    retrieved = []
    for i, (score, idx) in enumerate(zip(scores[0], indices[0])):
        if idx != -1:  # Valid result
            retrieved.append({
                "chunk_id": int(idx),
                "score": float(score),
                "rank": i + 1
            })
    
    retrieval_time = time.time() - start_time
    return retrieved, {"retrieval_time": retrieval_time}


def call_groq_api(prompt: str, stream: bool = False) -> tuple:
    """Call Groq API with error handling."""
    if not GROQ_AVAILABLE:
        return None, {"llm_time": 0, "error": "Groq not available"}
    
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None, {"llm_time": 0, "error": "GROQ_API_KEY not found"}
    
    start_time = time.time()
    
    try:
        client = groq.Groq(api_key=api_key)
        
        chat_completion = client.chat.completions.create(
            messages=[{"role": "user", "content": prompt}],
            model="llama-3.1-8b-instant",
            stream=stream,
            temperature=0.0,
            max_tokens=1024  # Increased from 256 to allow complete responses
        )
        
        if stream:
            # Handle streaming response
            response_text = ""
            for chunk in chat_completion:
                if chunk.choices[0].delta.content:
                    response_text += chunk.choices[0].delta.content
        else:
            response_text = chat_completion.choices[0].message.content
        
        # Ensure response ends properly (not cut off mid-sentence)
        if response_text and not response_text.rstrip().endswith(('.', '!', '?', ':', ')')):
            # Check if response was likely truncated
            if len(response_text.split()) > 50:  # Only add if it's a substantial response
                response_text = response_text.rstrip() + "."
        
        llm_time = time.time() - start_time
        return response_text, {"llm_time": llm_time}
        
    except Exception as e:
        llm_time = time.time() - start_time
        print(f"Groq API error: {e}")
        return None, {"llm_time": llm_time, "error": str(e)}


def build_prompt(question: str, retrieved: List[Dict], chunks_lookup: Dict) -> str:
    """Build prompt for LLM from question and retrieved chunks."""
    context_parts = []
    
    for item in retrieved:
        chunk_id = item["chunk_id"]
        chunk = chunks_lookup.get(chunk_id, {})
        
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        page = chunk.get("page", 0)
        
        context_parts.append(f"[Source: {source}, Page: {page}]\n{text}")
    
    context = "\n\n".join(context_parts)
    
    return f"""Based on the following context, answer the question completely and thoroughly. Be specific, cite sources when possible, and ensure your response ends with a complete sentence.

Context:
{context}

Question: {question}

Answer:"""


def extractive_answer(retrieved: List[Dict], chunks_lookup: Dict, 
                     score_threshold: float = 0.25) -> Dict[str, Any]:
    """Generate extractive answer from retrieved chunks."""
    if not retrieved:
        return {
            "answer": "No relevant information found.",
            "sources": [],
            "used_llm": False,
            "confidence": 0.0
        }
    
    # Filter by score threshold
    high_confidence = [r for r in retrieved if r["score"] > score_threshold]
    
    if not high_confidence:
        return {
            "answer": "I found some related information, but confidence is too low to provide a reliable answer.",
            "sources": format_sources(retrieved, chunks_lookup),
            "used_llm": False,
            "confidence": retrieved[0]["score"] if retrieved else 0.0
        }
    
    # Extract text from top chunks
    answer_parts = []
    for item in high_confidence[:3]:  # Top 3 chunks
        chunk = chunks_lookup.get(item["chunk_id"], {})
        text = chunk.get("text", "")
        source = chunk.get("source", "unknown")
        
        # Take first 200 chars as excerpt
        excerpt = text[:200] + "..." if len(text) > 200 else text
        answer_parts.append(f"From {source}: {excerpt}")
    
    answer = "\n\n".join(answer_parts)
    
    return {
        "answer": answer,
        "sources": format_sources(high_confidence, chunks_lookup),
        "used_llm": False,
        "confidence": high_confidence[0]["score"]
    }


def format_sources(retrieved: List[Dict], chunks_lookup: Dict) -> List[Dict]:
    """Format sources for consistent output."""
    sources = []
    for r in retrieved:
        chunk = chunks_lookup.get(r["chunk_id"], {})
        sources.append({
            "source": chunk.get("source", "unknown"),
            "page": chunk.get("page", 0),
            "chunk_id": r["chunk_id"],
            "score": r["score"],
            "snippet": chunk.get("text", "")[:200] + "..."
        })
    return sources


@dataclass
class RAGConfig:
    """Configuration for the RAG pipeline."""
    model_name: str = "all-MiniLM-L6-v2"
    top_k: int = 5
    score_threshold: float = 0.25
    use_llm: bool = True
    temperature: float = 0.0
    max_tokens: int = 256
    index_dir: str = "data"

    @classmethod
    def for_speed(cls) -> 'RAGConfig':
        """Fast responses for demos."""
        return cls(top_k=3, use_llm=False, max_tokens=128)
    
    @classmethod  
    def for_quality(cls) -> 'RAGConfig':
        """High-quality detailed responses."""
        return cls(top_k=7, temperature=0.1, max_tokens=512)


class LLMProvider(Protocol):
    """Interface for LLM providers."""
    
    def generate(self, prompt: str) -> tuple[str, Dict[str, float]]:
        """Generate response from prompt."""
        ...
    
    def supports_streaming(self) -> bool:
        """Check if provider supports streaming."""
        ...


class GroqProvider:
    """Groq provider implementation."""
    
    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model
        self.check_api_key()
    
    def check_api_key(self):
        """Check if Groq API key is available."""
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            print("⚠️  GROQ_API_KEY not found in environment variables!")
            print("   Set it in your .env file or environment")
            self.available = False
        else:
            print(f"✅ Found Groq API key: {api_key[:10]}...")
            self.available = True
    
    def generate(self, prompt: str) -> tuple[str, Dict[str, float]]:
        """Use Groq API to generate response."""
        if not self.available:
            return "", {"llm_time": 0, "error": "Groq API key not available"}
        
        response, timing = call_groq_api(prompt, stream=False)
        if response is None:
            return "", {"llm_time": 0, "error": "Groq call failed"}
        return response, timing
    
    def supports_streaming(self) -> bool:
        return True and self.available


class ExtractiveProvider:
    """Extractive provider implementation."""
    
    def generate(self, prompt: str) -> tuple[str, Dict[str, float]]:
        """Generate extractive response."""
        return "Response based on retrieved documents (extractive mode)", {"llm_time": 0}
    
    def supports_streaming(self) -> bool:
        return False


class LLMFactory:
    """Factory for creating LLM providers."""
    
    @staticmethod
    def create(provider: str, **kwargs) -> LLMProvider:
        """Create LLM provider by name."""
        if provider.lower() == "groq":
            return GroqProvider(**kwargs)
        elif provider.lower() == "extractive":
            return ExtractiveProvider()
        else:
            raise ValueError(f"Unknown provider: {provider}")


class RAGPipeline:
    """Main RAG pipeline with design patterns."""
    
    def __init__(self, config: RAGConfig, llm_provider: LLMProvider):
        self.config = config
        self.llm_provider = llm_provider
        
        print(f"🔧 Loading pipeline with config: {config}")
        
        # Load components
        try:
            self.model = get_cached_model(config.model_name)
            self.index, self.metadata, self.chunks_lookup = get_cached_index(config.index_dir)
            print(f"✅ Loaded FAISS index from {config.index_dir}")
            print(f"📚 Index contains {len(self.metadata)} chunks")
        except Exception as e:
            print(f"❌ Failed to load index: {e}")
            print("   Make sure you've run: python build_embeddings.py")
            raise
    
    def query(self, question: str, timing_enabled: bool = True) -> Dict[str, Any]:
        """Query the pipeline."""
        start_time = time.time() if timing_enabled else 0
        
        print(f"\n🔍 Processing query: {question}")
        
        # Step 1: Retrieve documents
        try:
            retrieved, timing = retrieve_documents(
                self.index, self.metadata, self.model, 
                question, self.config.top_k
            )
            print(f"📖 Retrieved {len(retrieved)} chunks")
            print(f"🎯 Top score: {retrieved[0]['score']:.3f}" if retrieved else "No results")
        except Exception as e:
            print(f"❌ Retrieval failed: {e}")
            return {"answer": "Retrieval failed", "error": str(e)}
        
        # Step 2: Check quality
        if not retrieved or retrieved[0]["score"] < self.config.score_threshold:
            print(f"⚠️  Low confidence (threshold: {self.config.score_threshold})")
            return self._create_fallback_response(retrieved, timing, question)
        
        # Step 3: Generate response
        if self.config.use_llm and self.llm_provider.supports_streaming():
            print("🤖 Generating LLM response...")
            return self._generate_llm_response(question, retrieved, timing, start_time)
        else:
            print("📝 Generating extractive response...")
            return self._generate_extractive_response(retrieved, timing, start_time)
    
    def _generate_llm_response(self, question: str, retrieved: List[Dict], 
                              retrieval_timing: Dict, start_time: float) -> Dict[str, Any]:
        """Generate LLM response."""
        
        # Build prompt
        prompt = build_prompt(question, retrieved, self.chunks_lookup)
        print(f"📝 Built prompt ({len(prompt)} chars)")
        
        # Generate using provider
        answer, llm_timing = self.llm_provider.generate(prompt)
        
        if not answer or "error" in llm_timing:
            print("⚠️  LLM failed, falling back to extractive")
            return self._generate_extractive_response(retrieved, retrieval_timing, start_time)
        
        # Combine timing
        total_timing = {**retrieval_timing, **llm_timing}
        if start_time:
            total_timing["total_time"] = time.time() - start_time
        
        print(f"✅ Generated {len(answer)} char response in {total_timing.get('total_time', 0):.2f}s")
        
        return {
            "answer": answer,
            "sources": format_sources(retrieved, self.chunks_lookup),
            "used_llm": True,
            "timing": total_timing,
            "config": self.config.__dict__
        }
    
    def _generate_extractive_response(self, retrieved: List[Dict], 
                                    retrieval_timing: Dict, start_time: float) -> Dict[str, Any]:
        """Generate extractive response."""
        
        # Use extractive function
        response = extractive_answer(retrieved, self.chunks_lookup, self.config.score_threshold)
        
        # Add timing
        if start_time:
            response["timing"] = {**retrieval_timing, "total_time": time.time() - start_time}
        
        response["config"] = self.config.__dict__
        print(f"✅ Generated extractive response: {len(response['answer'])} chars")
        return response
    
    def _create_fallback_response(self, retrieved: List[Dict], timing: Dict, query: str = "") -> Dict[str, Any]:
        """Create response for low-quality retrievals or greetings."""
        
        # Check if it's a simple greeting (very basic ones only)
        simple_greetings = ["hello", "hi", "hey"]
        query_lower = query.lower().strip()
        
        # Only use hardcoded response for very simple greetings without additional context
        if query_lower in simple_greetings and len(query_lower.split()) == 1:
            answer = """Hello! 👋 I'm your AI assistant, ready to help you with questions about the documents I have access to.

Here's what I can do for you:

🔍 **Document Q&A**: Ask me anything about the content in the loaded documents
📊 **Technical Details**: Explain concepts, implementations, and methodologies  
💡 **Guidance**: Help you understand complex topics step by step
🔧 **System Info**: Provide information about RAG systems, AI, and technical processes

**Try asking me:**
- "What is RAG and how does it work?"
- "Explain the main concepts from the documents"
- "How do I implement this system?"
- "What are the key benefits mentioned?"

I'll search through the documents and give you accurate, sourced answers. What would you like to know?"""
            
            return {
                "answer": answer,
                "sources": [],
                "used_llm": False,
                "timing": timing,
                "config": self.config.__dict__,
                "reason": "Simple greeting response"
            }
        
        # For more complex queries or greetings with context, try to use LLM if available
        if self.config.use_llm and self.llm_provider.supports_streaming():
            try:
                # Use LLM even for low-confidence retrievals to provide helpful responses
                prompt = build_prompt(query, retrieved[:3] if retrieved else [], self.chunks_lookup)
                response_text, llm_timing = self.llm_provider.generate(prompt)
                
                timing.update(llm_timing)
                
                return {
                    "answer": response_text,
                    "sources": format_sources(retrieved, self.chunks_lookup) if retrieved else [],
                    "used_llm": True,
                    "timing": timing,
                    "config": self.config.__dict__,
                    "reason": "LLM fallback response"
                }
            except Exception as e:
                print(f"⚠️  LLM fallback failed: {e}")
        
        # Default fallback for insufficient information
        return {
            "answer": "I couldn't find sufficient information to answer your question. Try asking about topics covered in the loaded documents, or rephrase your question to be more specific.",
            "sources": format_sources(retrieved, self.chunks_lookup) if retrieved else [],
            "used_llm": False,
            "timing": timing,
            "config": self.config.__dict__,
            "reason": "Low retrieval confidence"
        }


class RAGPipelineBuilder:
    """Builder for easy pipeline setup."""
    
    def __init__(self):
        self.config = RAGConfig()
        self.llm_provider = "groq"
        self.llm_kwargs = {}
    
    def with_config(self, config: RAGConfig) -> 'RAGPipelineBuilder':
        """Set custom configuration."""
        self.config = config
        return self
    
    def fast_mode(self) -> 'RAGPipelineBuilder':
        """Configure for fast responses."""
        self.config = RAGConfig.for_speed()
        self.llm_provider = "extractive"
        return self
    
    def quality_mode(self) -> 'RAGPipelineBuilder':
        """Configure for high-quality responses."""
        self.config = RAGConfig.for_quality()
        self.llm_provider = "groq"
        self.llm_kwargs = {"model": "llama-3.1-8b-instant"}
        return self
    
    def with_llm(self, provider: str, **kwargs) -> 'RAGPipelineBuilder':
        """Set LLM provider."""
        self.llm_provider = provider
        self.llm_kwargs = kwargs
        return self
    
    def with_index_dir(self, index_dir: str) -> 'RAGPipelineBuilder':
        """Set index directory."""
        self.config.index_dir = index_dir
        return self
    
    def build(self) -> RAGPipeline:
        """Build the pipeline."""
        llm_provider = LLMFactory.create(self.llm_provider, **self.llm_kwargs)
        return RAGPipeline(self.config, llm_provider)


# Backward compatibility functions
def answer_query_enhanced(query: str, out_dir: str = "data", mode: str = "auto") -> Dict[str, Any]:
    """
    Drop-in replacement for existing answer_query function.
    This is the main function to use for queries.
    """
    print(f"\n🚀 Enhanced RAG Query (mode: {mode})")
    
    if mode == "fast":
        pipeline = RAGPipelineBuilder().fast_mode().with_index_dir(out_dir).build()
    elif mode == "quality":
        pipeline = RAGPipelineBuilder().quality_mode().with_index_dir(out_dir).build()
    else:  # auto mode
        config = RAGConfig(index_dir=out_dir)
        pipeline = RAGPipelineBuilder().with_config(config).build()
    
    return pipeline.query(query)


def answer_query(query: str, out_dir: str = "data", top_k: int = 5) -> Dict[str, Any]:
    """
    Simple backward compatibility function.
    """
    config = RAGConfig(index_dir=out_dir, top_k=top_k)
    pipeline = RAGPipelineBuilder().with_config(config).build()
    return pipeline.query(query)


def demo_pipeline():
    """Demonstrate the pipeline."""
    print("🎯 Design Pattern RAG Pipeline Demo")
    print("=" * 50)
    
    # Check if index exists
    if not Path("data/faiss.index").exists():
        print("❌ No FAISS index found!")
        print("   Run: python build_embeddings.py --pdfs 'your_file.pdf' --out_dir data")
        return
    
    queries = [
        "What is RAG?",
        "How does retrieval work?",
        "What are the limitations?"
    ]
    
    # Demo different modes
    print("\n🔍 Testing different modes:")
    
    for mode in ["fast", "auto", "quality"]:
        print(f"\n--- {mode.upper()} MODE ---")
        
        try:
            if mode == "fast":
                pipeline = RAGPipelineBuilder().fast_mode().build()
            elif mode == "quality":
                pipeline = RAGPipelineBuilder().quality_mode().build()
            else:  # auto mode
                pipeline = RAGPipelineBuilder().build()
            
            response = pipeline.query(queries[0])
            
            print(f"Answer: {response['answer'][:100]}...")
            print(f"Used LLM: {response['used_llm']}")
            print(f"Sources: {len(response['sources'])}")
            if "timing" in response:
                print(f"Time: {response['timing'].get('total_time', 0):.2f}s")
        except Exception as e:
            print(f"❌ {mode} mode failed: {e}")


if __name__ == "__main__":
    demo_pipeline()