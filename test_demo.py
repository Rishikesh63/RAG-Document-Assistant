"""Test Demo - The 5 Required Queries

Design pattern-based implementation test with 5 example queries.
Tests different response modes and architecture patterns.

Run this after building embeddings to see the system in action.
"""
from pathlib import Path
from rag_pipeline import answer_query_enhanced, demo_pipeline
import json

EXAMPLE_QUERIES = [
    "What is Retrieval-Augmented Generation (RAG)?",
    "How do I set up the system for local inference?", 
    "What safety features are mentioned for the system?",
    "How do I contact customer support?",
    "Does the document describe limitations or future improvements?",
]


def run_demo():
    """Run demo with design pattern implementation."""
    print("🏗️ Design Pattern RAG Demo")
    print("=" * 50)
    
    out_dir = "data"
    results = []
    
    # Test different modes
    modes = ["fast", "auto", "quality"]
    
    for mode in modes:
        print(f"\n🔍 Testing {mode.upper()} mode:")
        print("-" * 30)
        
        mode_results = []
        
        for i, query in enumerate(EXAMPLE_QUERIES, 1):
            print(f"\n{i}. Query: {query}")
            
            try:
                response = answer_query_enhanced(query, out_dir=out_dir, mode=mode)
                
                print(f"Answer: {response.get('answer', '')[:100]}...")
                print(f"Used LLM: {response.get('used_llm', False)}")
                print(f"Sources: {len(response.get('sources', []))}")
                
                if 'timing' in response:
                    timing = response['timing']
                    total_time = timing.get('total_time', 0) * 1000
                    print(f"Response time: {total_time:.0f}ms")
                
                mode_results.append({
                    "query": query,
                    "response": response,
                    "mode": mode
                })
                
            except Exception as e:
                print(f"❌ Error: {e}")
                mode_results.append({
                    "query": query,
                    "error": str(e),
                    "mode": mode
                })
        
        results.extend(mode_results)
    
    # Save results
    output_path = Path(out_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    with open(output_path / "demo_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Saved demo results to {output_path / 'demo_results.json'}")
    
    # Performance summary
    print("\n📊 Performance Summary:")
    for mode in modes:
        mode_results = [r for r in results if r.get('mode') == mode and 'response' in r]
        if mode_results:
            avg_time = sum(
                r['response'].get('timing', {}).get('total_time', 0) * 1000
                for r in mode_results
            ) / len(mode_results)
            print(f"{mode.capitalize()}: {avg_time:.0f}ms average")


def run_single_query_demo():
    """Run a single query across all modes for comparison."""
    test_query = "What is Retrieval-Augmented Generation (RAG)?"
    
    print("🔍 Single Query Mode Comparison")
    print("=" * 50)
    print(f"Query: {test_query}\n")
    
    modes = ["fast", "auto", "quality"]
    
    for mode in modes:
        print(f"--- {mode.upper()} MODE ---")
        
        try:
            response = answer_query_enhanced(test_query, mode=mode)
            
            print(f"Answer: {response.get('answer', '')[:200]}...")
            print(f"Used LLM: {response.get('used_llm', False)}")
            print(f"Sources: {len(response.get('sources', []))}")
            
            if 'timing' in response:
                timing = response['timing']
                total_time = timing.get('total_time', 0) * 1000
                retrieval_time = timing.get('retrieval_time', 0) * 1000
                llm_time = timing.get('llm_time', 0) * 1000
                
                print(f"Total time: {total_time:.0f}ms")
                print(f"Retrieval: {retrieval_time:.0f}ms")
                if llm_time > 0:
                    print(f"LLM generation: {llm_time:.0f}ms")
            
        except Exception as e:
            print(f"❌ Error: {e}")
        
        print()


if __name__ == "__main__":
    print("Choose demo type:")
    print("1. Full demo (all queries, all modes)")
    print("2. Single query comparison")
    print("3. Architecture demo")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        run_demo()
    elif choice == "2":
        run_single_query_demo()
    elif choice == "3":
        demo_pipeline()
    else:
        print("Running full demo...")
        run_demo()
