"""Guardrails module for RAG system safety and quality.

This module provides basic guardrails including:
- Profanity filtering for queries and responses
- Hallucination detection by checking response grounding in context
"""
import re
from typing import List, Dict, Any
from difflib import SequenceMatcher

# Basic profanity word list (expandable)
PROFANITY_WORDS = {
    'damn', 'hell', 'shit', 'fuck', 'bitch', 'ass', 'bastard', 'crap',
    'piss', 'slut', 'whore', 'fag', 'nigger', 'retard', 'gay', 'stupid',
    'idiot', 'moron', 'dumb', 'hate', 'kill', 'die', 'murder', 'rape'
}

def contains_profanity(text: str) -> bool:
    """Check if text contains profanity."""
    if not text:
        return False
    
    text_lower = text.lower()
    # Remove punctuation and split into words
    words = re.findall(r'\b\w+\b', text_lower)
    
    for word in words:
        if word in PROFANITY_WORDS:
            return True
    
    return False


def filter_profanity(text: str) -> str:
    """Replace profanity in text with asterisks."""
    if not text:
        return text
    
    filtered_text = text
    for word in PROFANITY_WORDS:
        # Case-insensitive replacement
        pattern = re.compile(re.escape(word), re.IGNORECASE)
        replacement = '*' * len(word)
        filtered_text = pattern.sub(replacement, filtered_text)
    
    return filtered_text


def similarity(a: str, b: str) -> float:
    """Calculate similarity between two strings."""
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def check_hallucination(response: str, retrieved_chunks: List[Dict[str, Any]], 
                       chunks_lookup: Dict[int, Dict[str, Any]], 
                       min_similarity: float = 0.1) -> Dict[str, Any]:
    """
    Basic hallucination detection by checking if response content is grounded in retrieved context.
    
    Args:
        response: Generated response text
        retrieved_chunks: List of retrieved chunks with metadata
        chunks_lookup: Mapping of chunk_id to full chunk data
        min_similarity: Minimum similarity threshold to consider content grounded
    
    Returns:
        Dict with 'is_grounded', 'confidence', and 'warning' keys
    """
    if not response or not retrieved_chunks:
        return {"is_grounded": False, "confidence": 0.0, "warning": "No content to check"}
    
    # Get all retrieved text content
    all_context = ""
    for chunk in retrieved_chunks:
        chunk_data = chunks_lookup.get(chunk.get("chunk_id"))
        if chunk_data:
            all_context += " " + chunk_data.get("text", "")
    
    if not all_context.strip():
        return {"is_grounded": False, "confidence": 0.0, "warning": "No context available"}
    
    # Split response into sentences for checking
    response_sentences = re.split(r'[.!?]+', response)
    grounded_sentences = 0
    total_sentences = 0
    
    for sentence in response_sentences:
        sentence = sentence.strip()
        if len(sentence) < 10:  # Skip very short sentences
            continue
        
        total_sentences += 1
        max_sim = 0.0
        
        # Check similarity with context
        for context_chunk in re.split(r'[.!?]+', all_context):
            context_chunk = context_chunk.strip()
            if len(context_chunk) < 10:
                continue
            
            sim = similarity(sentence, context_chunk)
            max_sim = max(max_sim, sim)
        
        if max_sim >= min_similarity:
            grounded_sentences += 1
    
    if total_sentences == 0:
        confidence = 0.0
    else:
        confidence = grounded_sentences / total_sentences
    
    is_grounded = confidence >= 0.3  # At least 30% of content should be grounded
    
    warning = None
    if confidence < 0.3:
        warning = f"Low grounding confidence ({confidence:.1%}). Response may contain hallucinations."
    
    return {
        "is_grounded": is_grounded,
        "confidence": confidence,
        "warning": warning,
        "grounded_sentences": grounded_sentences,
        "total_sentences": total_sentences
    }


def apply_guardrails(query: str, response: str, retrieved_chunks: List[Dict[str, Any]], 
                    chunks_lookup: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Apply all guardrails to query and response.
    
    Returns:
        Dict with filtered content and warnings
    """
    result = {
        "filtered_query": query,
        "filtered_response": response,
        "warnings": [],
        "blocked": False
    }
    
    # Check query for profanity
    if contains_profanity(query):
        result["warnings"].append("Query contains inappropriate language")
        result["filtered_query"] = filter_profanity(query)
    
    # Check response for profanity
    if contains_profanity(response):
        result["warnings"].append("Response contains inappropriate language")
        result["filtered_response"] = filter_profanity(response)
    
    # Check for hallucinations
    hallucination_check = check_hallucination(response, retrieved_chunks, chunks_lookup)
    if hallucination_check.get("warning"):
        result["warnings"].append(hallucination_check["warning"])
    
    result["hallucination_check"] = hallucination_check
    
    # Block response if too many issues
    if len(result["warnings"]) >= 3:
        result["blocked"] = True
        result["filtered_response"] = "I cannot provide a response due to content safety concerns. Please rephrase your question."
    
    return result


# Quality checks
def check_response_quality(response: str, query: str) -> Dict[str, Any]:
    """Check basic response quality metrics."""
    quality = {
        "length_appropriate": 50 <= len(response) <= 2000,
        "contains_answer": "answer" in response.lower() or "according" in response.lower(),
        "not_empty": bool(response.strip()),
        "query_relevant": similarity(query, response) > 0.1
    }
    
    quality["score"] = sum(quality.values()) / len(quality)
    return quality