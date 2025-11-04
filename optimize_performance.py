"""Performance optimization and cache warming script.

This script pre-loads models and indices to optimize first-query performance.
Run before using the RAG system for best performance.
"""
import time
import os
from pathlib import Path
from rag_pipeline import get_cached_model, get_cached_index, answer_query

def warm_cache(data_dir: str = "data", model_name: str = "all-MiniLM-L6-v2"):
    """Pre-load models and indices for better performance."""
    print("🔥 Warming up RAG system cache...")
    
    start_time = time.time()
    
    # Pre-load embedding model
    print(f"Loading embedding model: {model_name}")
    model_start = time.time()
    model = get_cached_model(model_name)
    model_time = time.time() - model_start
    print(f"✅ Model loaded in {model_time:.3f}s")
    
    # Pre-load index and metadata
    print(f"Loading FAISS index from: {data_dir}")
    index_start = time.time()
    index, metadata, chunks_lookup = get_cached_index(data_dir)
    index_time = time.time() - index_start
    print(f"✅ Index loaded in {index_time:.3f}s")
    
    # Test query to ensure everything works
    print("Running test query...")
    test_start = time.time()
    test_response = answer_query(
        "What is RAG?", 
        out_dir=data_dir, 
        top_k=3
    )
    test_time = time.time() - test_start
    
    total_time = time.time() - start_time
    
    print("\n🎉 Cache warming complete!")
    print(f"Total setup time: {total_time:.3f}s")
    print(f"Test query time: {test_time*1000:.1f}ms")
    
    if test_response.get("timing"):
        timing = test_response["timing"]
        print("\nDetailed test query timing:")
        print(f"  - Total: {timing.get('total_time', 0)*1000:.1f}ms")
        print(f"  - Load: {timing.get('load_time', 0)*1000:.1f}ms (cached)")
        print(f"  - Embedding: {timing.get('embedding_time', 0)*1000:.1f}ms")
        print(f"  - Search: {timing.get('search_time', 0)*1000:.1f}ms")
        print(f"  - LLM: {timing.get('llm_time', 0)*1000:.1f}ms")
    
    print("\n📊 Performance Status:")
    total_ms = test_time * 1000
    if total_ms < 200:
        print(f"🟢 EXCELLENT: {total_ms:.0f}ms < 200ms target")
    elif total_ms < 500:
        print(f"🟡 GOOD: {total_ms:.0f}ms < 500ms")
    else:
        print(f"🔴 SLOW: {total_ms:.0f}ms > 500ms")
    
    return {
        "total_setup_time": total_time,
        "test_query_time": test_time,
        "model_load_time": model_time,
        "index_load_time": index_time,
        "performance_rating": "excellent" if total_ms < 200 else "good" if total_ms < 500 else "slow"
    }


def benchmark_queries(data_dir: str = "data", model_name: str = "all-MiniLM-L6-v2", num_queries: int = 5):
    """Benchmark multiple queries focusing on retrieval latency and time to first response."""
    print(f"\n🏁 Running benchmark with {num_queries} queries...")
    print("📊 Key Metrics:")
    print("  - Retrieval Latency: Time to find relevant documents (target <100ms)")
    print("  - Time to First Token: Time to start responding (target <200ms)")
    print("  - Total Response Time: Complete answer generation (quality vs speed)")
    
    test_queries = [
        "What is RAG?",
        "How does retrieval work?",
        "What are the benefits?",
        "How to implement this?",
        "What are the limitations?"
    ]
    
    retrieval_times = []
    first_token_times = []
    total_times = []
    
    for i, query in enumerate(test_queries[:num_queries], 1):
        print(f"\nQuery {i}/{num_queries}: {query[:30]}...")
        
        start_time = time.time()
        response = answer_query(
            query,
            out_dir=data_dir,
            top_k=3
        )
        total_time = time.time() - start_time
        total_times.append(total_time)
        
        timing = response.get("timing", {})
        
        # Key metrics
        retrieval_time = (timing.get("embedding_time", 0) + timing.get("search_time", 0)) * 1000
        first_response_time = (timing.get("load_time", 0) + timing.get("embedding_time", 0) + 
                              timing.get("search_time", 0) + timing.get("prompt_time", 0)) * 1000
        total_ms = timing.get("total_time", total_time) * 1000
        
        retrieval_times.append(retrieval_time)
        first_token_times.append(first_response_time)
        
        # Status indicators
        retrieval_status = "🟢" if retrieval_time < 100 else "🟡" if retrieval_time < 200 else "🔴"
        first_token_status = "🟢" if first_response_time < 200 else "🟡" if first_response_time < 400 else "🔴"
        
        print(f"  Retrieval: {retrieval_status} {retrieval_time:.1f}ms")
        print(f"  First Response: {first_token_status} {first_response_time:.1f}ms") 
        print(f"  Total: {total_ms:.1f}ms")
    
    # Statistics
    avg_retrieval = sum(retrieval_times) / len(retrieval_times)
    avg_first_token = sum(first_token_times) / len(first_token_times)
    avg_total = sum(total_times) / len(total_times) * 1000
    
    print("\n📈 Benchmark Results:")
    print(f"  Average Retrieval Latency: {avg_retrieval:.1f}ms (target <100ms)")
    print(f"  Average Time to First Response: {avg_first_token:.1f}ms (target <200ms)")
    print(f"  Average Total Time: {avg_total:.1f}ms")
    
    # Performance assessment
    retrieval_ok = avg_retrieval < 100
    first_token_ok = avg_first_token < 200
    
    print("\n🎯 Performance Assessment:")
    if retrieval_ok and first_token_ok:
        print("🟢 EXCELLENT: Both retrieval and first response within targets")
    elif first_token_ok:
        print("🟡 GOOD: First response fast, but retrieval could be optimized")
    elif retrieval_ok:
        print("🟡 MIXED: Fast retrieval, but slow first response (check prompt processing)")
    else:
        print("🔴 NEEDS OPTIMIZATION: Both metrics exceed targets")
    
    return {
        "average_retrieval_ms": avg_retrieval,
        "average_first_token_ms": avg_first_token,
        "average_total_ms": avg_total,
        "retrieval_within_target": retrieval_ok,
        "first_token_within_target": first_token_ok,
        "all_retrieval_times": retrieval_times,
        "all_first_token_times": first_token_times
    }


def optimize_performance():
    """Run performance optimizations and suggestions."""
    print("\n🚀 Performance Optimization Suggestions:")
    
    # Check if GROQ API key is set
    if os.environ.get("GROQ_API_KEY"):
        print("✅ GROQ API key found - LLM calls will be fast")
    else:
        print("⚠️  GROQ API key not set - will use slower extractive answers")
        print("   Set GROQ_API_KEY environment variable for better performance")
    
    # Check data directory
    data_path = Path("data")
    if data_path.exists():
        index_size = (data_path / "faiss.index").stat().st_size if (data_path / "faiss.index").exists() else 0
        print(f"📊 Index size: {index_size / 1024 / 1024:.1f} MB")
        
        if index_size > 100 * 1024 * 1024:  # 100MB
            print("⚠️  Large index detected - consider reducing chunk count for faster search")
    
    print("\n💡 Performance Tips:")
    print("1. Keep the Streamlit app running to maintain cache")
    print("2. Use smaller top_k values (3-5) for faster search")
    print("3. Set GROQ_API_KEY for fastest LLM responses")
    print("4. Consider using a smaller embedding model for faster encoding")


def main():
    """Main performance optimization and benchmarking."""
    print("⚡ RAG Performance Optimizer")
    print("=" * 50)
    
    # Check if index exists
    if not Path("data/faiss.index").exists():
        print("❌ FAISS index not found. Please run: python build_embeddings.py")
        return
    
    # Warm cache
    warm_results = warm_cache()
    
    # Run benchmark
    benchmark_results = benchmark_queries()
    
    # Optimization suggestions
    optimize_performance()
    
    # Final summary
    print("\n" + "=" * 60)
    print("📋 PERFORMANCE SUMMARY")
    print("=" * 60)
    
    retrieval_ok = benchmark_results["retrieval_within_target"]
    first_token_ok = benchmark_results["first_token_within_target"]
    avg_retrieval = benchmark_results["average_retrieval_ms"]
    avg_first_token = benchmark_results["average_first_token_ms"]
    
    print(f"🔍 Retrieval Latency: {avg_retrieval:.1f}ms (target <100ms)")
    if retrieval_ok:
        print("  ✅ EXCELLENT: Fast document retrieval")
    else:
        print("  ⚠️  Needs optimization: Consider smaller embeddings or index optimization")
    
    print(f"⚡ Time to First Response: {avg_first_token:.1f}ms (target <200ms)")
    if first_token_ok:
        print("  ✅ EXCELLENT: Quick response initiation")
    else:
        print("  ⚠️  Needs optimization: Consider model caching or prompt optimization")
    
    print(f"⏱️  Total Response Time: {benchmark_results['average_total_ms']:.1f}ms")
    print("  💡 Note: Total time includes answer generation and can be longer for quality")
    
    print(f"\nSetup time: {warm_results['total_setup_time']:.2f}s (one-time cost)")
    
    if retrieval_ok and first_token_ok:
        print("\n🎉 SUCCESS: System meets latency targets for real-time user experience!")
        print("   Users will experience snappy, responsive interactions.")
    elif first_token_ok:
        print("\n🟡 GOOD: Fast user response, but retrieval could be optimized further.")
    else:
        print("\n🔧 NEEDS WORK: Latency optimization required for smooth user experience.")
    
    print("\n💡 Key Insight: Latency = Time to start responding, not total generation time")
    print(f"   Your system's responsiveness is measured by first response time ({avg_first_token:.0f}ms)")


if __name__ == "__main__":
    main()