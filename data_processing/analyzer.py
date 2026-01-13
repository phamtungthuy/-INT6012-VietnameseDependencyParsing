"""
Data Analyzer for Vietnamese Dependency Parsing
Provides analysis tools to optimize hyperparameters based on dataset statistics.
"""

from collections import Counter
from typing import List, Dict, Tuple
from dataclasses import dataclass

from data_processing.loader import CoNLLUDataset
from utils.constants import TRAIN_FILE_PATH, VALIDATION_FILE_PATH, TEST_FILE_PATH
from utils.logs import logger


@dataclass
class MinFreqAnalysis:
    """Results of min_freq analysis"""
    min_freq: int
    vocab_size: int
    coverage: float  # % of tokens covered
    oov_rate: float  # % of tokens that are OOV
    unique_oov: int  # number of unique OOV words


class Analyzer:
    """Analyzer for dataset statistics and hyperparameter optimization."""
    
    def __init__(self):
        self.train_data: CoNLLUDataset | None = None
        self.dev_data: CoNLLUDataset | None = None
        self.test_data: CoNLLUDataset | None = None
        self.word_counter: Counter = Counter()
        self.pos_counter: Counter = Counter()
        self.rel_counter: Counter = Counter()
        
    def load_data(self):
        """Load all datasets"""
        logger.info("Loading datasets...")
        self.train_data = CoNLLUDataset(TRAIN_FILE_PATH)
        self.dev_data = CoNLLUDataset(VALIDATION_FILE_PATH)
        self.test_data = CoNLLUDataset(TEST_FILE_PATH)
        
        # Count word frequencies from training data
        self._count_frequencies()
        
    def _count_frequencies(self):
        assert self.train_data is not None, "Train data not loaded"
        
        self.word_counter.clear()
        self.pos_counter.clear()
        self.rel_counter.clear()
        
        for sentence in self.train_data.sentences:
            for token in sentence:
                self.word_counter[token['form'].lower()] += 1
                self.pos_counter[token['upos']] += 1
                self.rel_counter[token['deprel']] += 1
                
        logger.info(f"Counted {len(self.word_counter)} unique words in training data")
    
    def analyze_min_freq(self, freq_range: List[int] | None = None) -> List[MinFreqAnalysis]:
        """
        Analyze different min_freq values to find optimal setting.
        
        Args:
            freq_range: List of min_freq values to test. Default [1, 2, 3, 5, 10]
            
        Returns:
            List of MinFreqAnalysis for each min_freq value
        """
        if self.train_data is None:
            self.load_data()
            
        if freq_range is None:
            freq_range = [1, 2, 3, 5, 10, 15, 20]
        
        results = []
        total_tokens = sum(self.word_counter.values())
        
        for min_freq in freq_range:
            # Build vocab with this min_freq
            vocab_words = {word for word, count in self.word_counter.items() 
                          if count >= min_freq}
            vocab_size = len(vocab_words) + 3  # +3 for PAD, UNK, ROOT
            
            # Calculate coverage on training data
            covered_tokens = sum(count for word, count in self.word_counter.items() 
                                if count >= min_freq)
            coverage = covered_tokens / total_tokens * 100
            
            # Calculate OOV on dev/test
            oov_tokens = 0
            total_eval_tokens = 0
            oov_words = set()
            
            assert self.dev_data is not None and self.test_data is not None
            for dataset in [self.dev_data, self.test_data]:
                for sentence in dataset.sentences:
                    for token in sentence:
                        word = token['form'].lower()
                        total_eval_tokens += 1
                        # OOV if word not in vocab OR word freq < min_freq in training
                        if word not in vocab_words:
                            oov_tokens += 1
                            oov_words.add(word)
            
            oov_rate = oov_tokens / total_eval_tokens * 100 if total_eval_tokens > 0 else 0
            
            results.append(MinFreqAnalysis(
                min_freq=min_freq,
                vocab_size=vocab_size,
                coverage=coverage,
                oov_rate=oov_rate,
                unique_oov=len(oov_words)
            ))
        
        return results
    
    def recommend_min_freq(self) -> Tuple[int, str]:
        """
        Recommend optimal min_freq based on analysis.
        
        Returns:
            Tuple of (recommended_min_freq, explanation)
        """
        results = self.analyze_min_freq()
        
        # Heuristic: Find min_freq where:
        # - Coverage >= 95%
        # - OOV rate is acceptable (< 30%)
        # - Vocab size is reasonable
        
        best = None
        for r in results:
            # Prefer higher min_freq if coverage still good
            if r.coverage >= 95 and r.oov_rate < 35:
                if best is None or r.min_freq > best.min_freq:
                    best = r
        
        # Fallback to min_freq=2 if no good option
        if best is None:
            best = next((r for r in results if r.min_freq == 2), results[0])
        
        explanation = (
            f"Recommended min_freq={best.min_freq}:\n"
            f"  • Vocab size: {best.vocab_size:,} words\n"
            f"  • Training coverage: {best.coverage:.1f}%\n"
            f"  • Dev/Test OOV rate: {best.oov_rate:.1f}%\n"
            f"  • Unique OOV words: {best.unique_oov:,}"
        )
        
        return best.min_freq, explanation
    
    def print_analysis_report(self):
        """Print detailed analysis report"""
        if self.train_data is None:
            self.load_data()
            
        results = self.analyze_min_freq()
        
        print("""
╔══════════════════════════════════════════════════════════════════════════════╗
║                         MIN_FREQ ANALYSIS REPORT                             ║
╚══════════════════════════════════════════════════════════════════════════════╝
""")
        
        # Dataset stats
        assert self.train_data is not None and self.dev_data is not None and self.test_data is not None
        print("📊 Dataset Statistics:")
        print(f"   Train sentences: {len(self.train_data.sentences):,}")
        print(f"   Dev sentences:   {len(self.dev_data.sentences):,}")
        print(f"   Test sentences:  {len(self.test_data.sentences):,}")
        print(f"   Unique words:    {len(self.word_counter):,}")
        print(f"   Total tokens:    {sum(self.word_counter.values()):,}")
        print()
        
        # Frequency distribution
        print("📈 Word Frequency Distribution:")
        freq_buckets = {1: 0, 2: 0, 3: 0, 5: 0, 10: 0, 20: 0, 50: 0, 100: 0}
        for count in self.word_counter.values():
            for threshold in sorted(freq_buckets.keys()):
                if count >= threshold:
                    freq_buckets[threshold] += 1
        
        for threshold, count in freq_buckets.items():
            pct = count / len(self.word_counter) * 100
            bar = "█" * int(pct / 2)
            print(f"   freq >= {threshold:3d}: {count:5,} words ({pct:5.1f}%) {bar}")
        print()
        
        # Min_freq analysis table
        print("🔍 Min_freq Analysis:")
        print("   ┌─────────┬────────────┬───────────┬──────────┬────────────┐")
        print("   │ min_freq│ Vocab Size │ Coverage  │ OOV Rate │ Unique OOV │")
        print("   ├─────────┼────────────┼───────────┼──────────┼────────────┤")
        
        for r in results:
            print(f"   │ {r.min_freq:7d} │ {r.vocab_size:10,} │ {r.coverage:8.1f}% │ {r.oov_rate:7.1f}% │ {r.unique_oov:10,} │")
        
        print("   └─────────┴────────────┴───────────┴──────────┴────────────┘")
        print()
        
        # Recommendation
        recommended, explanation = self.recommend_min_freq()
        print("✅ Recommendation:")
        print(f"   {explanation}")
        print()
        
        # Tips
        print("💡 Tips:")
        print("   • Lower min_freq = larger vocab, better coverage, but more noise")
        print("   • Higher min_freq = smaller vocab, faster training, but more OOV")
        print("   • Typical sweet spot: min_freq=2 or 3")
        print()
        
        return recommended


def analyze_min_freq():
    """CLI entry point for min_freq analysis"""
    analyzer = Analyzer()
    return analyzer.print_analysis_report()

