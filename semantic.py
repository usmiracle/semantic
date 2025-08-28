import numpy as np
from sentence_transformers import SentenceTransformer, util
import torch
from typing import Tuple, Dict, Any
import argparse

class SemanticSimilarity:
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize the semantic similarity analyzer with a pre-trained model.
        
        Args:
            model_name: Name of the sentence transformer model to use
        """
        self.model = SentenceTransformer(model_name)
        print(f"Loaded model: {model_name}")
    
    def compute_similarity(self, sentence1: str, sentence2: str) -> Dict[str, Any]:
        """
        Compute semantic similarity between two sentences.
        
        Args:
            sentence1: First sentence to compare
            sentence2: Second sentence to compare
            
        Returns:
            Dictionary containing similarity scores and analysis
        """
        # Encode sentences to embeddings
        embeddings1 = self.model.encode(sentence1, convert_to_tensor=True)
        embeddings2 = self.model.encode(sentence2, convert_to_tensor=True)
        
        # Compute cosine similarity
        cosine_similarity = util.pytorch_cos_sim(embeddings1, embeddings2).item()
        
        # Compute dot product similarity
        dot_similarity = torch.dot(embeddings1, embeddings2).item()
        
        # Compute Euclidean distance
        euclidean_distance = torch.norm(embeddings1 - embeddings2).item()
        
        # Normalize Euclidean distance to 0-1 range (inverse relationship)
        max_distance = torch.norm(embeddings1) + torch.norm(embeddings2)
        normalized_euclidean_similarity = 1 - (euclidean_distance / max_distance)
        
        # Aggregate into a single final similarity score (0..1)
        # Weighted average favoring cosine similarity
        weight_cosine = 0.7
        weight_norm_euclid = 0.3
        final_similarity = (
            weight_cosine * float(cosine_similarity)
            + weight_norm_euclid * float(normalized_euclidean_similarity)
        )
        # Clamp to [0, 1]
        final_similarity = max(0.0, min(1.0, final_similarity))
        
        # Determine similarity level (kept based on cosine for backward-compatibility/tests)
        similarity_level = self._get_similarity_level(cosine_similarity)
        
        return {
            'sentence1': sentence1,
            'sentence2': sentence2,
            'cosine_similarity': cosine_similarity,
            'dot_similarity': dot_similarity,
            'euclidean_distance': euclidean_distance,
            'normalized_euclidean_similarity': normalized_euclidean_similarity,
            'final_similarity': final_similarity,
            'similarity_level': similarity_level,
            'explanation': self._get_explanation(cosine_similarity, similarity_level)
        }
    
    def _get_similarity_level(self, cosine_similarity: float) -> str:
        """Determine the similarity level based on cosine similarity score."""
        if cosine_similarity >= 0.8:
            return "Very High Similarity"
        elif cosine_similarity >= 0.6:
            return "High Similarity"
        elif cosine_similarity >= 0.4:
            return "Moderate Similarity"
        elif cosine_similarity >= 0.2:
            return "Low Similarity"
        else:
            return "Very Low Similarity"
    
    def _get_explanation(self, cosine_similarity: float, similarity_level: str) -> str:
        """Provide a human-readable explanation of the similarity."""
        explanations = {
            "Very High Similarity": "These sentences are very similar in meaning. They likely express the same concept or idea using different words.",
            "High Similarity": "These sentences are quite similar. They share related concepts and themes, though they may differ in some details.",
            "Moderate Similarity": "These sentences have some similarity. They may share some concepts but differ in their main focus or meaning.",
            "Low Similarity": "These sentences have limited similarity. They may share some words or concepts but express different ideas.",
            "Very Low Similarity": "These sentences are very different. They likely express unrelated concepts or ideas."
        }
        return explanations.get(similarity_level, "Unable to determine similarity level.")
    
    def compare_multiple_sentences(self, sentences: list) -> Dict[str, Any]:
        """
        Compare multiple sentences and find the most similar pairs.
        
        Args:
            sentences: List of sentences to compare
            
        Returns:
            Dictionary containing pairwise similarities and rankings
        """
        if len(sentences) < 2:
            return {"error": "Need at least 2 sentences to compare"}
        
        # Encode all sentences
        embeddings = self.model.encode(sentences, convert_to_tensor=True)
        
        # Compute pairwise similarities
        similarity_matrix = util.pytorch_cos_sim(embeddings, embeddings)
        
        # Get all pairwise combinations (excluding self-similarity)
        pairs = []
        for i in range(len(sentences)):
            for j in range(i + 1, len(sentences)):
                similarity = similarity_matrix[i][j].item()
                pairs.append({
                    'sentence1': sentences[i],
                    'sentence2': sentences[j],
                    'similarity': similarity,
                    'similarity_level': self._get_similarity_level(similarity)
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
    print("SEMANTIC SIMILARITY ANALYSIS")
    print("="*60)
    print(f"Sentence 1: '{results['sentence1']}'")
    print(f"Sentence 2: '{results['sentence2']}'")
    print("-"*60)
    # Aggregated score first
    if 'final_similarity' in results:
        print(f"Final Similarity (aggregated): {results['final_similarity']:.4f}")
    print(f"Cosine Similarity: {results['cosine_similarity']:.4f}")
    print(f"Dot Product Similarity: {results['dot_similarity']:.4f}")
    print(f"Euclidean Distance: {results['euclidean_distance']:.4f}")
    print(f"Normalized Euclidean Similarity: {results['normalized_euclidean_similarity']:.4f}")
    print("-"*60)
    print(f"Similarity Level: {results['similarity_level']}")
    print(f"Explanation: {results['explanation']}")
    print("="*60)

def interactive_mode():
    """Run the program in interactive mode."""
    print("Welcome to the Semantic Similarity Analyzer!")
    print("This program compares two sentences and tells you how semantically similar they are.")
    print("Type 'quit' to exit.\n")
    
    analyzer = SemanticSimilarity()
    
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
    parser = argparse.ArgumentParser(description='Compare semantic similarity between sentences')
    parser.add_argument('--sentence1', '-s1', help='First sentence to compare')
    parser.add_argument('--sentence2', '-s2', help='Second sentence to compare')
    parser.add_argument('--interactive', '-i', action='store_true', help='Run in interactive mode')
    parser.add_argument('--model', '-m', default='all-MiniLM-L6-v2', 
                       help='Sentence transformer model to use')
    
    args = parser.parse_args()
    
    if args.interactive or (not args.sentence1 and not args.sentence2):
        interactive_mode()
    elif args.sentence1 and args.sentence2:
        analyzer = SemanticSimilarity(args.model)
        results = analyzer.compute_similarity(args.sentence1, args.sentence2)
        print_similarity_results(results)
    else:
        print("Please provide both sentences or use --interactive mode.")
        print("Example: python semantic.py --sentence1 'Hello world' --sentence2 'Hi there'")

if __name__ == "__main__":
    main()
