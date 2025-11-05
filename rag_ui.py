"""Streamlit Web Interface for Design Pattern RAG System

A modern UI for the design pattern-based RAG implementation.
Supports performance metrics, chat interface, and design pattern configuration.

Features:
- Design pattern-based architecture
- Performance monitoring and optimization tips
- Chat history with source citations
- Safety features and quality metrics
- Multiple response modes (fast, auto, quality)

Run: streamlit run rag_ui.py
"""
import streamlit as st
import json
from pathlib import Path
from rag_pipeline import answer_query_enhanced
from guardrails import apply_guardrails


def display_timing_metrics(timing_info: dict):
    """Display timing breakdown focusing on user-perceived latency."""
    if not timing_info:
        return
    
    st.subheader("⚡ Performance Metrics")
    
    # Calculate key latency metrics
    retrieval_time = timing_info.get("retrieval_time", 0) * 1000
    llm_time = timing_info.get("llm_time", 0) * 1000
    total_time = timing_info.get("total_time", 0) * 1000
    
    # Main metrics focused on user experience
    col1, col2, col3 = st.columns(3)
    
    with col1:
        retrieval_color = "🟢" if retrieval_time < 100 else "🟡" if retrieval_time < 200 else "🔴"
        retrieval_status = "Fast" if retrieval_time < 100 else "Moderate" if retrieval_time < 200 else "Slow"
        st.metric("🔍 Retrieval Time", f"{retrieval_time:.0f} ms", 
                 delta=f"{retrieval_color} {retrieval_status}")
    
    with col2:
        if llm_time > 0:
            llm_color = "🟢" if llm_time < 500 else "🟡" if llm_time < 1000 else "🔴"
            llm_status = "Fast" if llm_time < 500 else "Moderate" if llm_time < 1000 else "Slow"
            st.metric("🤖 LLM Generation", f"{llm_time:.0f} ms", 
                     delta=f"{llm_color} {llm_status}")
        else:
            st.metric("📄 Extractive Mode", "0 ms", delta="🟢 Instant")
    
    with col3:
        total_color = "🟢" if total_time < 500 else "🟡" if total_time < 1000 else "🔴"
        total_status = "Excellent" if total_time < 500 else "Good" if total_time < 1000 else "Slow"
        st.metric("⏱️ Total Time", f"{total_time:.0f} ms", 
                 delta=f"{total_color} {total_status}")
    
    # User experience indicator
    if total_time < 500:
        st.success("🎉 Excellent user experience: System feels instant and responsive!")
    elif total_time < 1000:
        st.info("👍 Good user experience: Users will notice quick responses")
    else:
        st.warning("⚠️ Users may notice lag: Consider optimization for better experience")


# Page config
st.set_page_config(
    page_title="AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main > div {
        padding-top: 2rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .source-card {
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: white;
    }
</style>
""", unsafe_allow_html=True)

# Title and description
st.title("🤖 AI Assistant")
st.markdown("**RAG system built with design patterns - fast and scalable!**")

# Initialize session state
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Sidebar for settings
st.sidebar.header("⚙️ Configuration")
data_dir = st.sidebar.text_input("Data Directory", value="data")

# Design Pattern Settings
st.sidebar.subheader("🏗️ Design Pattern Configuration")
response_mode = st.sidebar.selectbox(
    "Response Mode",
    ["auto", "fast", "quality"],
    help="""
    Fast: Extractive answers, <200ms response
    Auto: Intelligent switching between modes
    Quality: LLM-powered detailed answers
    """
)

# Advanced settings
st.sidebar.subheader("🛡️ Safety Features")
enable_guardrails = st.sidebar.checkbox("Enable Guardrails", value=True, 
                                       help="Filter profanity and check for hallucinations")

# Performance settings
st.sidebar.subheader("⚡ Performance")
show_timing = st.sidebar.checkbox("Show Timing Metrics", value=True, 
                                help="Display detailed performance breakdown")
show_quality_metrics = st.sidebar.checkbox("Show Quality Metrics", value=True)
show_confidence_scores = st.sidebar.checkbox("Show Confidence Scores", value=True)

# Check if index exists
index_path = Path(data_dir) / "faiss.index"
if not index_path.exists():
    st.error("⚠️ No FAISS index found! Please run `python build_embeddings.py` first.")
    st.code("python build_embeddings.py --pdfs 'your_pdfs.pdf' --out_dir data")
    st.stop()

# Load index info if available
info_path = Path(data_dir) / "index_info.json"
if info_path.exists():
    with open(info_path, 'r') as f:
        index_info = json.load(f)
    st.sidebar.success(f"✅ Index loaded: {index_info.get('num_chunks', 'Unknown')} chunks")
    st.sidebar.info(f"Model: {index_info.get('model_name', 'Unknown')}")

# Display design pattern info
st.sidebar.markdown("---")
st.sidebar.subheader("🏗️ Architecture Info")
st.sidebar.info("""
**Design Patterns Used:**
- Strategy: Swappable LLM providers
- Builder: Fluent configuration
- Factory: Component creation
- Caching: Global model/index caching
""")

# Main query interface
st.header("💬 Ask a Question")

# Chat-like interface
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.write(message["content"])
        if "sources" in message:
            with st.expander("📚 View Sources"):
                for i, source in enumerate(message["sources"][:3], 1):
                    st.write(f"**{i}.** {source.get('source')} (Page {source.get('page')}) - Score: {source.get('score', 0):.3f}")

# Query input
query = st.chat_input("Enter your question here...")

# Example queries section
with st.expander("💡 Try these example queries"):
    example_queries = [
        "What is Retrieval-Augmented Generation (RAG)?",
        "How does the system architecture work?",
        "What are the performance characteristics?",
        "How do I implement custom providers?",
        "What safety features are included?"
    ]
    
    cols = st.columns(2)
    for i, example in enumerate(example_queries):
        col = cols[i % 2]
        if col.button(f"📝 {example[:40]}...", key=f"example_{i}"):
            query = example

# Process query
if query:
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": query})
    
    with st.chat_message("user"):
        st.write(query)
    
    with st.chat_message("assistant"):
        try:
            # Use design pattern implementation
            with st.spinner(f"🔍 Processing query in {response_mode} mode..."):
                response = answer_query_enhanced(
                    query=query,
                    out_dir=data_dir,
                    mode=response_mode
                )
                
                full_response = response.get("answer", "No answer generated.")
                sources = response.get("sources", [])
                used_llm = response.get("used_llm", False)
                timing_info = response.get("timing", {})
                config_info = response.get("config", {})
                
                # Display response
                st.markdown(full_response)
            
            # Display timing metrics
            if show_timing and timing_info:
                display_timing_metrics(timing_info)
                
                # Show configuration used
                with st.expander("🔧 Configuration Used"):
                    col1, col2 = st.columns(2)
                    with col1:
                        st.json({
                            "mode": response_mode,
                            "top_k": config_info.get("top_k"),
                            "use_llm": config_info.get("use_llm"),
                            "model_name": config_info.get("model_name")
                        })
                    with col2:
                        st.json({
                            "score_threshold": config_info.get("score_threshold"),
                            "temperature": config_info.get("temperature"),
                            "max_tokens": config_info.get("max_tokens"),
                            "index_dir": config_info.get("index_dir")
                        })
            
            # Apply guardrails if enabled
            if enable_guardrails:
                try:
                    chunks_file = Path(data_dir) / "chunks.jsonl"
                    chunks_lookup = {}
                    if chunks_file.exists():
                        with open(chunks_file, 'r', encoding='utf-8') as f:
                            for line_num, line in enumerate(f):
                                try:
                                    chunk_data = json.loads(line.strip())
                                    chunks_lookup[line_num] = chunk_data
                                except Exception:
                                    continue
                    
                    guardrail_result = apply_guardrails(query, full_response, sources, chunks_lookup)
                    
                    # Show warnings if any
                    if guardrail_result.get("warnings"):
                        for warning in guardrail_result["warnings"]:
                            st.warning(f"⚠️ {warning}")
                    
                    # Use filtered response if blocked
                    if guardrail_result.get("blocked"):
                        full_response = guardrail_result["filtered_response"]
                        st.error("🚫 Response was filtered due to safety concerns.")
                    
                    # Show hallucination check
                    if show_quality_metrics and "hallucination_check" in guardrail_result:
                        halluc_check = guardrail_result["hallucination_check"]
                        confidence = halluc_check.get("confidence", 0)
                        
                        if confidence >= 0.7:
                            st.success(f"✅ High content grounding: {confidence:.1%}")
                        elif confidence >= 0.3:
                            st.warning(f"⚠️ Moderate content grounding: {confidence:.1%}")
                        else:
                            st.error(f"❌ Low content grounding: {confidence:.1%}")
                
                except Exception as e:
                    st.warning(f"⚠️ Guardrails check failed: {str(e)}")
            
            # Show response type and quality
            col1, col2, col3 = st.columns(3)
            with col1:
                if used_llm:
                    st.success("🤖 LLM Generated")
                else:
                    st.info("📄 Extractive Answer")
            
            with col2:
                if show_confidence_scores and sources:
                    avg_score = sum(s.get('score', 0) for s in sources) / len(sources)
                    st.metric("Avg. Confidence", f"{avg_score:.3f}")
            
            with col3:
                st.metric("Sources Found", len(sources))
            
            # Display sources with enhanced cards
            if sources:
                st.subheader("📚 Sources")
                
                for i, source in enumerate(sources, 1):
                    with st.container():
                        st.markdown(f"""
                        <div class="source-card">
                            <h5>📄 Source {i}: {source.get('source')} (Page {source.get('page')})</h5>
                            <p><strong>Relevance Score:</strong> {source.get('score', 0):.4f}</p>
                            <p><strong>Chunk ID:</strong> {source.get('chunk_id')}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Show chunk content if available
                        if st.button(f"View Chunk {i} Content", key=f"chunk_view_{i}"):
                            chunks_file = Path(data_dir) / "chunks.jsonl"
                            if chunks_file.exists():
                                with open(chunks_file, 'r', encoding='utf-8') as f:
                                    for line_num, line in enumerate(f):
                                        try:
                                            chunk_data = json.loads(line.strip())
                                            if line_num == source.get('chunk_id'):
                                                st.text_area(
                                                    f"Content from {source.get('source')}:",
                                                    chunk_data.get('text', ''),
                                                    height=150,
                                                    key=f"chunk_content_{i}"
                                                )
                                                break
                                        except Exception:
                                            continue
            else:
                st.warning("❌ No relevant sources found for this query.")
            
            # Add to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": full_response,
                "sources": sources
            })
                
        except Exception as e:
            st.error(f"❌ Error processing query: {str(e)}")
            st.write("Make sure you have:")
            st.write("1. Built the FAISS index with `python build_embeddings.py`")
            st.write("2. Installed all required dependencies")
            st.write("3. Set GROQ_API_KEY for enhanced answers (optional)")

# Sidebar controls
st.sidebar.markdown("---")
st.sidebar.subheader("💬 Chat Controls")
if st.sidebar.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

if st.sidebar.button("💾 Export Chat"):
    if st.session_state.messages:
        chat_export = {
            "messages": st.session_state.messages,
            "settings": {
                "data_dir": data_dir,
                "response_mode": response_mode,
                "guardrails_enabled": enable_guardrails
            }
        }
        st.sidebar.download_button(
            label="📥 Download Chat JSON",
            data=json.dumps(chat_export, indent=2),
            file_name="rag_chat_export.json",
            mime="application/json"
        )

# Demo the design patterns
st.sidebar.markdown("---")
if st.sidebar.button("🏗️ Demo Design Patterns"):
    st.session_state.show_demo = True

if st.session_state.get('show_demo', False):
    with st.expander("🏗️ Design Pattern Demo", expanded=True):
        st.markdown("**See how different patterns affect the same query:**")
        
        demo_query = "What is RAG?"
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🚀 Fast Mode")
            try:
                fast_response = answer_query_enhanced(demo_query, data_dir, "fast")
                st.write(f"**Time:** {fast_response.get('timing', {}).get('total_time', 0)*1000:.0f}ms")
                st.write(f"**Type:** {'LLM' if fast_response.get('used_llm') else 'Extractive'}")
                st.write(f"**Answer:** {fast_response.get('answer', '')[:100]}...")
            except Exception as e:
                st.error(f"Error: {e}")
        
        with col2:
            st.subheader("🔄 Auto Mode")
            try:
                auto_response = answer_query_enhanced(demo_query, data_dir, "auto")
                st.write(f"**Time:** {auto_response.get('timing', {}).get('total_time', 0)*1000:.0f}ms")
                st.write(f"**Type:** {'LLM' if auto_response.get('used_llm') else 'Extractive'}")
                st.write(f"**Answer:** {auto_response.get('answer', '')[:100]}...")
            except Exception as e:
                st.error(f"Error: {e}")
        
        with col3:
            st.subheader("⭐ Quality Mode")
            try:
                quality_response = answer_query_enhanced(demo_query, data_dir, "quality")
                st.write(f"**Time:** {quality_response.get('timing', {}).get('total_time', 0)*1000:.0f}ms")
                st.write(f"**Type:** {'LLM' if quality_response.get('used_llm') else 'Extractive'}")
                st.write(f"**Answer:** {quality_response.get('answer', '')[:100]}...")
            except Exception as e:
                st.error(f"Error: {e}")

# Footer
st.markdown("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.markdown("🏗️ **Architecture:** Design pattern-based for scalability")
with col2:
    st.markdown("🛡️ **Safety:** Guardrails built in")
with col3:
    st.markdown("⚡ **Performance:** Multiple response modes available")

# System status in sidebar
st.sidebar.markdown("---")
st.sidebar.subheader("📊 System Status")
st.sidebar.success("✅ Design Pattern RAG Ready")
if Path(data_dir, "faiss.index").exists():
    st.sidebar.success("✅ Vector Index Loaded")
else:
    st.sidebar.error("❌ Vector Index Missing")

# Info about the system
with st.sidebar.expander("ℹ️ About This System"):
    st.write("""
    **Design Pattern RAG Features:**
    
    🏗️ **Architecture Patterns**: 
    - Strategy Pattern for provider switching
    - Builder Pattern for configuration
    - Factory Pattern for component creation
    
    🛡️ **Safety Features**: 
    - Guardrails and content filtering
    - Hallucination detection
    - Quality metrics
    
    ⚡ **Performance Modes**: 
    - Fast: <200ms extractive answers
    - Auto: Intelligent mode switching
    - Quality: LLM-powered responses
    
    🔧 **Extensible**: Easy to add new providers and features
    """)