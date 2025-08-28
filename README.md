# Semantic Similarity Analyzer

A Python program that compares two sentences and determines their semantic similarity using state-of-the-art sentence transformers.

## Features

- **Multiple Similarity Metrics**: Cosine similarity, dot product, and Euclidean distance
- **Human-readable Explanations**: Provides clear explanations of similarity levels
- **Interactive Mode**: Easy-to-use interactive interface
- **Command Line Interface**: Batch processing capabilities
- **Multiple Sentence Comparison**: Compare multiple sentences at once
- **Customizable Models**: Support for different pre-trained models

## Installation

1. **Install Python dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **First run will download the pre-trained model** (this may take a few minutes depending on your internet connection).

## Usage

### Interactive Mode (Recommended for beginners)

```bash
python semantic.py
```

This will start an interactive session where you can enter sentences one by one.

### Command Line Mode

```bash
# Compare two specific sentences
python semantic.py --sentence1 "The cat is on the mat" --sentence2 "A feline sits on the carpet"

# Or use short flags
python semantic.py -s1 "Hello world" -s2 "Hi there"

# Use a different model
python semantic.py -s1 "I love pizza" -s2 "I enjoy Italian food" -m "all-mpnet-base-v2"
```

### Programmatic Usage

```python
from semantic import SemanticSimilarity

# Initialize the analyzer
analyzer = SemanticSimilarity()

# Compare two sentences
results = analyzer.compute_similarity(
    "The weather is nice today", 
    "It's a beautiful day outside"
)

# Print results
print(f"Similarity: {results['cosine_similarity']:.4f}")
print(f"Level: {results['similarity_level']}")
print(f"Explanation: {results['explanation']}")

# Compare multiple sentences
sentences = [
    "I love programming",
    "Coding is my passion", 
    "The weather is sunny",
    "I enjoy writing code"
]

multi_results = analyzer.compare_multiple_sentences(sentences)
print(f"Most similar pair: {multi_results['most_similar']}")
```

## Similarity Levels

The program categorizes similarity into five levels:

- **Very High Similarity** (≥0.8): Sentences express the same concept using different words
- **High Similarity** (≥0.6): Sentences share related concepts and themes
- **Moderate Similarity** (≥0.4): Sentences have some shared concepts but differ in focus
- **Low Similarity** (≥0.2): Sentences share limited concepts but express different ideas
- **Very Low Similarity** (<0.2): Sentences express unrelated concepts

## Available Models

- `all-MiniLM-L6-v2` (default): Fast and efficient, good for most use cases
- `all-mpnet-base-v2`: Higher quality but slower
- `all-distilroberta-v1`: Good balance of speed and quality
- `paraphrase-multilingual-MiniLM-L12-v2`: Multilingual support

## Example Output

```
============================================================
SEMANTIC SIMILARITY ANALYSIS
============================================================
Sentence 1: 'The cat is on the mat'
Sentence 2: 'A feline sits on the carpet'
------------------------------------------------------------
Cosine Similarity: 0.8234
Dot Product Similarity: 15.6789
Euclidean Distance: 2.3456
Normalized Euclidean Similarity: 0.7890
------------------------------------------------------------
Similarity Level: Very High Similarity
Explanation: These sentences are very similar in meaning. They likely express the same concept or idea using different words.
============================================================
```

## Technical Details

The program uses:
- **Sentence Transformers**: Pre-trained models that convert sentences into high-dimensional vectors
- **Cosine Similarity**: Measures the cosine of the angle between two vectors (most commonly used)
- **Dot Product**: Measures the magnitude of projection of one vector onto another
- **Euclidean Distance**: Measures the straight-line distance between two points in vector space

## Requirements

- Python 3.7+
- Internet connection (for first run to download models)
- 2GB+ RAM recommended for larger models

## Testing

The project includes comprehensive tests to ensure reliability and correctness.

### Running Tests

```bash
# Run all unit tests
python test_semantic.py

# Or use the test runner for more options
python run_tests.py

# Run specific types of tests
python run_tests.py --unit          # Unit tests only
python run_tests.py --performance   # Performance tests with real model
python run_tests.py --coverage      # Tests with coverage reporting
python run_tests.py --all           # All tests

# Run a specific test
python run_tests.py --test TestSemanticSimilarity.test_compute_similarity_basic

# Verbose output
python run_tests.py --verbose
```

### Test Coverage

The test suite covers:
- ✅ Core similarity computation functionality
- ✅ Multiple similarity metrics (cosine, dot product, Euclidean)
- ✅ Similarity level categorization
- ✅ Multiple sentence comparison
- ✅ Edge cases (empty strings, long sentences)
- ✅ Command line interface
- ✅ Interactive mode
- ✅ Error handling
- ✅ Integration tests

### Test Structure

- **Unit Tests**: Test individual components in isolation
- **Integration Tests**: Test the complete workflow
- **Performance Tests**: Test with real model data
- **Mock Tests**: Fast tests that don't require model download

## Troubleshooting

1. **Model download issues**: Ensure you have a stable internet connection for the first run
2. **Memory issues**: Try using a smaller model like `all-MiniLM-L6-v2`
3. **CUDA errors**: The program will automatically use CPU if CUDA is not available
4. **Test failures**: Run `python run_tests.py --verbose` for detailed error information

## License

This project is open source and available under the MIT License. 