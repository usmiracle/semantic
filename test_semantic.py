import unittest
import sys
import os
from unittest.mock import patch, MagicMock
import numpy as np
import torch

# Add the current directory to the path so we can import semantic
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from semantic import SemanticSimilarity, print_similarity_results, interactive_mode, main

class TestSemanticSimilarity(unittest.TestCase):
    """Test cases for the SemanticSimilarity class."""
    
    def setUp(self):
        """Set up test fixtures before each test method."""
        # Mock the SentenceTransformer to avoid downloading models during testing
        self.mock_model_patcher = patch('semantic.SentenceTransformer')
        self.mock_model_class = self.mock_model_patcher.start()
        self.mock_model = MagicMock()
        self.mock_model_class.return_value = self.mock_model
        
        # Create sample embeddings for testing
        self.sample_embedding1 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5])
        self.sample_embedding2 = torch.tensor([0.2, 0.3, 0.4, 0.5, 0.6])
        
        # Mock the encode method
        self.mock_model.encode.side_effect = [self.sample_embedding1, self.sample_embedding2]
        
        # Create the analyzer instance
        self.analyzer = SemanticSimilarity()
    
    def tearDown(self):
        """Clean up after each test method."""
        self.mock_model_patcher.stop()
    
    def test_init(self):
        """Test the initialization of SemanticSimilarity."""
        # Test default model
        analyzer = SemanticSimilarity()
        self.mock_model_class.assert_called_with('all-MiniLM-L6-v2')
        
        # Test custom model
        custom_analyzer = SemanticSimilarity('custom-model')
        self.mock_model_class.assert_called_with('custom-model')
    
    def test_compute_similarity_basic(self):
        """Test basic similarity computation."""
        sentence1 = "Hello world"
        sentence2 = "Hi there"
        
        # Mock cosine similarity
        with patch('semantic.util.pytorch_cos_sim') as mock_cos_sim:
            mock_cos_sim.return_value = torch.tensor([[0.75]])
            
            result = self.analyzer.compute_similarity(sentence1, sentence2)
            
            # Verify the result structure
            self.assertIn('sentence1', result)
            self.assertIn('sentence2', result)
            self.assertIn('cosine_similarity', result)
            self.assertIn('dot_similarity', result)
            self.assertIn('euclidean_distance', result)
            self.assertIn('normalized_euclidean_similarity', result)
            self.assertIn('similarity_level', result)
            self.assertIn('explanation', result)
            
            # Verify values
            self.assertEqual(result['sentence1'], sentence1)
            self.assertEqual(result['sentence2'], sentence2)
            self.assertEqual(result['cosine_similarity'], 0.75)
            self.assertEqual(result['similarity_level'], "High Similarity")
    
    def test_get_similarity_level(self):
        """Test similarity level categorization."""
        # Test different similarity levels
        test_cases = [
            (0.9, "Very High Similarity"),
            (0.7, "High Similarity"),
            (0.5, "Moderate Similarity"),
            (0.3, "Low Similarity"),
            (0.1, "Very Low Similarity"),
        ]
        
        for similarity, expected_level in test_cases:
            with self.subTest(similarity=similarity):
                level = self.analyzer._get_similarity_level(similarity)
                self.assertEqual(level, expected_level)
    
    def test_get_explanation(self):
        """Test explanation generation."""
        # Test that explanations are generated for each level
        levels = [
            "Very High Similarity",
            "High Similarity", 
            "Moderate Similarity",
            "Low Similarity",
            "Very Low Similarity"
        ]
        
        for level in levels:
            with self.subTest(level=level):
                explanation = self.analyzer._get_explanation(0.5, level)
                self.assertIsInstance(explanation, str)
                self.assertGreater(len(explanation), 0)
    
    def test_compare_multiple_sentences(self):
        """Test multiple sentence comparison."""
        sentences = ["Hello", "Hi", "Goodbye"]
        
        # Mock embeddings for multiple sentences
        embeddings = torch.tensor([[0.1, 0.2], [0.2, 0.3], [0.8, 0.9]])
        self.mock_model.encode.return_value = embeddings
        
        # Mock similarity matrix
        with patch('semantic.util.pytorch_cos_sim') as mock_cos_sim:
            # Create a mock similarity matrix
            similarity_matrix = torch.tensor([
                [1.0, 0.8, 0.2],
                [0.8, 1.0, 0.3],
                [0.2, 0.3, 1.0]
            ])
            mock_cos_sim.return_value = similarity_matrix
            
            result = self.analyzer.compare_multiple_sentences(sentences)
            
            # Verify result structure
            self.assertIn('total_sentences', result)
            self.assertIn('total_pairs', result)
            self.assertIn('pairwise_comparisons', result)
            self.assertIn('most_similar', result)
            self.assertIn('least_similar', result)
            
            # Verify values
            self.assertEqual(result['total_sentences'], 3)
            self.assertEqual(result['total_pairs'], 3)  # 3C2 = 3 pairs
            
            # Verify pairs are sorted by similarity (highest first)
            pairs = result['pairwise_comparisons']
            self.assertEqual(len(pairs), 3)
            self.assertEqual(pairs[0]['similarity'], 0.8)  # Highest similarity
            self.assertEqual(pairs[-1]['similarity'], 0.2)  # Lowest similarity
    
    def test_compare_multiple_sentences_insufficient(self):
        """Test multiple sentence comparison with insufficient sentences."""
        result = self.analyzer.compare_multiple_sentences(["Hello"])
        self.assertIn('error', result)
        self.assertEqual(result['error'], "Need at least 2 sentences to compare")
    
    def test_compute_similarity_edge_cases(self):
        """Test similarity computation with edge cases."""
        # Test with empty strings
        with patch('semantic.util.pytorch_cos_sim') as mock_cos_sim:
            mock_cos_sim.return_value = torch.tensor([[0.0]])
            
            result = self.analyzer.compute_similarity("", "")
            self.assertEqual(result['cosine_similarity'], 0.0)
            self.assertEqual(result['similarity_level'], "Very Low Similarity")
        
        # Test with very long sentences
        long_sentence = "This is a very long sentence with many words " * 10
        with patch('semantic.util.pytorch_cos_sim') as mock_cos_sim:
            mock_cos_sim.return_value = torch.tensor([[0.6]])
            
            result = self.analyzer.compute_similarity(long_sentence, long_sentence)
            self.assertEqual(result['cosine_similarity'], 0.6)
            self.assertEqual(result['similarity_level'], "High Similarity")

class TestPrintFunctions(unittest.TestCase):
    """Test cases for print and utility functions."""
    
    def test_print_similarity_results(self):
        """Test the print_similarity_results function."""
        # Mock the print function to capture output
        with patch('builtins.print') as mock_print:
            sample_results = {
                'sentence1': 'Hello world',
                'sentence2': 'Hi there',
                'cosine_similarity': 0.75,
                'dot_similarity': 10.5,
                'euclidean_distance': 2.0,
                'normalized_euclidean_similarity': 0.7,
                'similarity_level': 'High Similarity',
                'explanation': 'These sentences are quite similar.'
            }
            
            print_similarity_results(sample_results)
            
            # Verify that print was called multiple times
            self.assertGreater(mock_print.call_count, 5)
            
            # Verify specific content was printed
            printed_output = [call.args[0] for call in mock_print.call_args_list]
            output_text = '\n'.join(printed_output)
            
            self.assertIn('SEMANTIC SIMILARITY ANALYSIS', output_text)
            self.assertIn('Hello world', output_text)
            self.assertIn('Hi there', output_text)
            self.assertIn('0.7500', output_text)  # Formatted cosine similarity
            self.assertIn('High Similarity', output_text)

class TestMainFunction(unittest.TestCase):
    """Test cases for the main function and command line interface."""
    
    @patch('semantic.interactive_mode')
    def test_main_interactive_mode(self, mock_interactive):
        """Test main function in interactive mode."""
        with patch('sys.argv', ['semantic.py', '--interactive']):
            main()
            mock_interactive.assert_called_once()
    
    @patch('semantic.SemanticSimilarity')
    @patch('semantic.print_similarity_results')
    def test_main_with_sentences(self, mock_print_results, mock_analyzer_class):
        """Test main function with sentence arguments."""
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.compute_similarity.return_value = {'test': 'result'}
        
        with patch('sys.argv', ['semantic.py', '--sentence1', 'Hello', '--sentence2', 'Hi']):
            main()
            
            mock_analyzer_class.assert_called_once_with('all-MiniLM-L6-v2')
            mock_analyzer.compute_similarity.assert_called_once_with('Hello', 'Hi')
            mock_print_results.assert_called_once_with({'test': 'result'})
    
    @patch('builtins.print')
    def test_main_no_arguments(self, mock_print):
        """Test main function with no arguments."""
        with patch('sys.argv', ['semantic.py']):
            with patch('semantic.interactive_mode') as mock_interactive:
                main()
                mock_interactive.assert_called_once()
    
    @patch('builtins.print')
    def test_main_incomplete_arguments(self, mock_print):
        """Test main function with incomplete arguments."""
        with patch('sys.argv', ['semantic.py', '--sentence1', 'Hello']):
            main()
            
            # Verify help message was printed
            printed_output = [call.args[0] for call in mock_print.call_args_list]
            output_text = '\n'.join(printed_output)
            self.assertIn('Please provide both sentences', output_text)

class TestInteractiveMode(unittest.TestCase):
    """Test cases for interactive mode."""
    
    @patch('builtins.input')
    @patch('builtins.print')
    @patch('semantic.SemanticSimilarity')
    @patch('semantic.print_similarity_results')
    def test_interactive_mode_quit(self, mock_print_results, mock_analyzer_class, mock_print, mock_input):
        """Test interactive mode with quit command."""
        mock_input.return_value = 'quit'
        
        interactive_mode()
        
        # Verify quit was handled
        mock_input.assert_called_once()
    
    @patch('builtins.input')
    @patch('builtins.print')
    @patch('semantic.SemanticSimilarity')
    @patch('semantic.print_similarity_results')
    def test_interactive_mode_normal_usage(self, mock_print_results, mock_analyzer_class, mock_print, mock_input):
        """Test interactive mode with normal sentence comparison."""
        mock_analyzer = MagicMock()
        mock_analyzer_class.return_value = mock_analyzer
        mock_analyzer.compute_similarity.return_value = {'test': 'result'}
        
        # Simulate user entering sentences then quitting
        mock_input.side_effect = ['Hello world', 'Hi there', 'quit']
        
        interactive_mode()
        
        # Verify sentences were processed
        mock_analyzer.compute_similarity.assert_called_once_with('Hello world', 'Hi there')
        mock_print_results.assert_called_once_with({'test': 'result'})

class TestIntegration(unittest.TestCase):
    """Integration tests that test the system as a whole."""
    
    def test_full_workflow(self):
        """Test the complete workflow from initialization to result."""
        # This test would require actual model loading, so we'll mock it
        with patch('semantic.SentenceTransformer') as mock_transformer_class:
            mock_model = MagicMock()
            mock_transformer_class.return_value = mock_model
            
            # Create realistic embeddings
            embedding1 = torch.randn(384)  # Typical embedding size
            embedding2 = torch.randn(384)
            mock_model.encode.side_effect = [embedding1, embedding2]
            
            # Create analyzer and test
            analyzer = SemanticSimilarity()
            result = analyzer.compute_similarity("Test sentence 1", "Test sentence 2")
            
            # Verify the complete result structure
            required_keys = [
                'sentence1', 'sentence2', 'cosine_similarity', 'dot_similarity',
                'euclidean_distance', 'normalized_euclidean_similarity',
                'similarity_level', 'explanation'
            ]
            
            for key in required_keys:
                self.assertIn(key, result)
                self.assertIsNotNone(result[key])

def run_performance_test():
    """Run a simple performance test to ensure the system works with real data."""
    print("\n" + "="*50)
    print("PERFORMANCE TEST")
    print("="*50)
    
    try:
        # This will actually load the model and run a real comparison
        analyzer = SemanticSimilarity()
        
        # Test with some sample sentences
        test_cases = [
            ("The cat is on the mat", "A feline sits on the carpet"),
            ("I love programming", "Coding is my passion"),
            ("The weather is sunny", "It's raining outside"),
            ("Hello world", "Goodbye world"),
        ]
        
        for sentence1, sentence2 in test_cases:
            print(f"\nTesting: '{sentence1}' vs '{sentence2}'")
            result = analyzer.compute_similarity(sentence1, sentence2)
            print(f"Similarity: {result['cosine_similarity']:.4f} - {result['similarity_level']}")
        
        print("\nPerformance test completed successfully!")
        
    except Exception as e:
        print(f"Performance test failed: {e}")
        print("This is expected if the model hasn't been downloaded yet.")

if __name__ == '__main__':
    # Run the unit tests
    print("Running unit tests...")
    unittest.main(verbosity=2, exit=False)
    
    # Run performance test (optional)
    print("\n" + "="*50)
    response = input("Run performance test with real model? (y/n): ").lower().strip()
    if response == 'y':
        run_performance_test() 