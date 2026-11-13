"""
KurdRAG: Morphologically Calibrated Retrieval-Augmented Generation for Kurdish Sorani
Core Architectures, Algorithms, and Baselines
"""

import re
import math
import numpy as np
from typing import List, Dict, Tuple, Optional

class MorphoBPETokenizer:
    """
    Morphologically-aware tokenizer for Kurdish Sorani.
    Detaches inflectional clitics, demonstrative suffixes, and enclitics
    prior to subword embedding, ensuring morphological invariance.
    """
    CLITICS = [
        'ەکەمان', 'ەکەتان', 'ەکەیان', 'ەکانمان', 'ەکانتان', 'ەکانیان',
        'ەکان', 'ەکە', 'مان', 'تان', 'یان', 'دا', 'وە', 'وەی', 'پێ', 'لێ'
    ]
    
    def __init__(self, vocab_size: int = 32000):
        self.vocab_size = vocab_size
        # Sort clitics by length descending for greedy prefix/suffix detachment
        self.clitic_patterns = sorted(self.CLITICS, key=len, reverse=True)
    
    def normalize_orthography(self, text: str) -> str:
        """Standardize Kurdish Sorani Unicode characters."""
        text = text.replace('\u064a', '\u06cc')  # Arabic Yeh to Farsi/Kurdish Yeh
        text = text.replace('\u0643', '\u06a9')  # Arabic Kaf to Keheh
        text = text.replace('\u0647\u200c', '\u06d5')  # Heh + ZWNJ to Kurdish Ae
        text = re.sub(r'[\u200c\u200b]+', ' ', text)  # normalize zero-width spaces
        return re.sub(r'\s+', ' ', text).strip()
    
    def segment_word(self, word: str) -> List[str]:
        """Detaches bound morphemes and enclitics from lexical stem."""
        for clitic in self.clitic_patterns:
            if word.endswith(clitic) and len(word) > len(clitic) + 2:
                stem = word[:-len(clitic)]
                return [stem, f"##{clitic}"]
        return [word]
    
    def tokenize(self, text: str) -> List[str]:
        norm_text = self.normalize_orthography(text)
        words = norm_text.split()
        tokens = []
        for w in words:
            tokens.extend(self.segment_word(w))
        return tokens


