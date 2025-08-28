#!/usr/bin/env python3
"""
Simplified Semantic Similarity Analyzer
A lightweight version that demonstrates the functionality without heavy dependencies.
"""

import math
import re
from typing import Dict, Any, List
import argparse

class SimpleSemanticSimilarity:
    """A simplified semantic similarity analyzer using basic NLP techniques."""
    
    def __init__(self):
        """Initialize the simple analyzer."""
        print("Loaded Simple Semantic Similarity Analyzer")
        print("Note: This is a simplified version using basic NLP techniques.")
        print("For more accurate results, use the full version with sentence transformers.\n")
    
    def preprocess_text(self, text: str) -> List[str]:
        """Preprocess text by tokenizing and normalizing."""
        # Convert to lowercase and remove punctuation
        text = re.sub(r'[^\w\s]', '', text.lower())
        # Split into words and remove common stop words
        stop_words = {'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those', 'i', 'you', 'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'}
        words = [word for word in text.split() if word not in stop_words and len(word) > 2]
        return words
    
    def compute_jaccard_similarity(self, words1: List[str], words2: List[str]) -> float:
        """Compute Jaccard similarity between two sets of words."""
        set1 = set(words1)
        set2 = set(words2)
        
        if not set1 and not set2:
            return 1.0  # Both empty sets are considered identical
        
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        
        return intersection / union if union > 0 else 0.0
    
    def compute_cosine_similarity(self, words1: List[str], words2: List[str]) -> float:
        """Compute cosine similarity using word frequency vectors."""
        # Create word frequency dictionaries
        freq1 = {}
        freq2 = {}
        
        for word in words1:
            freq1[word] = freq1.get(word, 0) + 1
        
        for word in words2:
            freq2[word] = freq2.get(word, 0) + 1
        
        # Get all unique words
        all_words = set(freq1.keys()).union(set(freq2.keys()))
        
        if not all_words:
            return 1.0  # Both empty are considered identical
        
        # Compute dot product and magnitudes
        dot_product = 0
        mag1 = 0
        mag2 = 0
        
        for word in all_words:
            count1 = freq1.get(word, 0)
            count2 = freq2.get(word, 0)
            
            dot_product += count1 * count2
            mag1 += count1 * count1
            mag2 += count2 * count2
        
        mag1 = math.sqrt(mag1)
        mag2 = math.sqrt(mag2)
        
        if mag1 == 0 or mag2 == 0:
            return 0.0
        
        return dot_product / (mag1 * mag2)
    
    def compute_similarity(self, sentence1: str, sentence2: str) -> Dict[str, Any]:
        """
        Compute semantic similarity between two sentences.
        
        Args:
            sentence1: First sentence to compare
            sentence2: Second sentence to compare
            
        Returns:
            Dictionary containing similarity scores and analysis
        """
        # Preprocess sentences
        words1 = self.preprocess_text(sentence1)
        words2 = self.preprocess_text(sentence2)
        
        # Compute different similarity metrics
        jaccard_similarity = self.compute_jaccard_similarity(words1, words2)
        cosine_similarity = self.compute_cosine_similarity(words1, words2)
        
        # Use cosine similarity as the primary metric
        primary_similarity = cosine_similarity
        
        # Determine similarity level
        similarity_level = self._get_similarity_level(primary_similarity)
        
        return {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'cosine_similarity': cosine_similarity,
            'jaccard_similarity': jaccard_similarity,
            'similarity_level': similarity_level,
            'explanation': self._get_explanation(primary_similarity, similarity_level),
            'processed_words1': words1,
            'processed_words2': words2
        }
    
    def _get_similarity_level(self, similarity: float) -> str:
        """Determine the similarity level based on similarity score."""
        if similarity >= 0.8:
            return "Very High Similarity"
        elif similarity >= 0.6:
            return "High Similarity"
        elif similarity >= 0.4:
            return "Moderate Similarity"
        elif similarity >= 0.2:
            return "Low Similarity"
        else:
            return "Very Low Similarity"
    
    def _get_explanation(self, similarity: float, similarity_level: str) -> str:
        """Provide a human-readable explanation of the similarity."""
        explanations = {
            "Very High Similarity": "These sentences are very similar in meaning. They likely express the same concept or idea using different words.",
            "High Similarity": "These sentences are quite similar. They share related concepts and themes, though they may differ in some details.",
            "Moderate Similarity": "These sentences have some similarity. They may share some concepts but differ in their main focus or meaning.",
            "Low Similarity": "These sentences have limited similarity. They may share some words or concepts but express different ideas.",
            "Very Low Similarity": "These sentences are very different. They likely express unrelated concepts or ideas."
        }
        return explanations.get(similarity_level, "Unable to determine similarity level.")
    
    def compare_multiple_sentences(self, sentences: List[str]) -> Dict[str, Any]:
        """
        Compare multiple sentences and find the most similar pairs.
        
        Args:
            sentences: List of sentences to compare
            
        Returns:
            Dictionary containing pairwise similarities and rankings
        """
        if len(sentences) < 2:
            return {"error": "Need at least 2 sentences to compare"}
        
        # Get all pairwise combinations
        pairs = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                result = self.compute_similarity(sentences[i], sentences[j])
                pairs.append({
                    'sentence1': sentences[i],
                    'sentence2': sentences[j],
                    'similarity': result['cosine_similarity'],
                    'similarity_level': result['similarity_level']
                })
        
        # Sort by similarity (highest first)
        pairs.sort(key=lambda x: x['similarity'], reverse=True)
        
        return {
            'total_sentences': len(sentences),
            'total_pairs': len(pairs),
            'pairwise_comparisons': pairs,
            'most_similar': pairs[0] if pairs else None,
            'least_similar': pairs[-1] if pairs else None
        }

def print_similarity_results(results: Dict[str, Any]):
    """Print the similarity results in a formatted way."""
    print("\n" + "="*60)
    print("SEMANTIC SIMILARITY ANALYSIS (Simple Version)")
    print("="*60)
    print(f"Sentence 1: '{results['sentence1']}'")
    print(f"Sentence 2: '{results['sentence2']}'")
    print("-"*60)
    print(f"Cosine Similarity: {results['cosine_similarity']:.4f}")
    print(f"Jaccard Similarity: {results['jaccard_similarity']:.4f}")
    print("-"*60)
    print(f"Similarity Level: {results['similarity_level']}")
    print(f"Explanation: {results['explanation']}")
    print("-"*60)
    print(f"Processed Words 1: {results['processed_words1']}")
    print(f"Processed Words 2: {results['processed_words2']}")
    print("="*60)

def interactive_mode():
    """Run the program in interactive mode."""
    print("Welcome to the Simple Semantic Similarity Analyzer!")
    print("This program compares two sentences and tells you how semantically similar they are.")
    print("Type 'quit' to exit.\n")
    
    analyzer = SimpleSemanticSimilarity()
    
    while True:
        print("\nEnter two sentences to compare:")
        sentence1 = input("Sentence 1: ").strip()
        
        if sentence1.lower() == 'quit':
            break
            
        sentence2 = input("Sentence 2: ").strip()
        
        if sentence2.lower() == 'quit':
            break
        
        if not sentence1 or not sentence2:
            print("Please enter both sentences.")
            continue
        
        try:
            results = analyzer.compute_similarity(sentence1, sentence2)
            print_similarity_results(results)
        except Exception as e:
            print(f"Error computing similarity: {e}")

def main():
    """Main function to handle command line arguments and run the program."""
    parser = argparse.ArgumentParser(description='Compare semantic similarity between sentences (Simple Version)')
    parser.add_argument('--sentence1', '-s1', help='First sentence to compare')
    parser.add_argument('--sentence2', '-s2', help='Second sentence to compare')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    
    args = parser.parse_args()
    
    if args.interactive or (not args.sentence1 and not args.sentence2):
        interactive_mode()
    elif args.sentence1 and args.sentence2:
        analyzer = SimpleSemanticSimilarity()
        results = analyzer.compute_similarity(args.sentence1, args.sentence2)
        print_similarity_results(results)
    else:
        print("Please provide both sentences or use --interactive mode.")
        print("Example: python semantic_simple.py --sentence1 'Hello world' --sentence2 'Hi there'")

if __name__ == "__main__":
    main() 