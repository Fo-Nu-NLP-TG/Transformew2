import re
import pandas as pd
from collections import Counter

class StoplistGenerator:
    def __init__(self):
        self.ewe_stopwords = set()
        self.english_stopwords = set()
    
    def regex_based_stoplist(self, corpus, language='ewe'):
        """Generate stoplist based on regex patterns"""
        stopwords = set()
        
        if language == 'ewe':
            # Common Ewe articles, pronouns, conjunctions, etc.
            patterns = [
                r'\b(la|na|be|ke|le|ɖe|kple|gake|eye)\b',  # Common function words
                r'\b(nye|wo|mi|mia|ye|ame)\b',  # Pronouns
                r'\b(ɖo|yi|va|le|nu)\b',  # Common verbs
                r'\b(alo|gɔ̃|hã)\b'  # Other common words
            ]
        else:  # English
            patterns = [
                r'\b(the|a|an|of|in|on|at|by|for|with|about)\b',  # Articles & prepositions
                r'\b(and|or|but|if|because|as|when|while)\b',  # Conjunctions
                r'\b(i|you|he|she|it|we|they|my|your|his|her|its|our|their)\b',  # Pronouns
                r'\b(is|am|are|was|were|be|been|being|have|has|had|do|does|did)\b'  # Aux verbs
            ]
        
        # Apply patterns to find stopwords
        for pattern in patterns:
            matches = re.findall(pattern, corpus, re.IGNORECASE)
            stopwords.update([match.lower() for match in matches])
        
        return stopwords
    
    def frequency_based_stoplist(self, corpus, language='ewe', threshold=0.01):
        """Generate stoplist based on word frequency"""
        words = re.findall(r'\b\w+\b', corpus.lower())
        word_count = Counter(words)
        total_words = len(words)
        
        # Words appearing with frequency > threshold are considered stopwords
        stopwords = {word for word, count in word_count.items() 
                    if count/total_words > threshold}
        return stopwords
    
    def generate_stoplists(self, ewe_corpus, english_corpus):
        """Generate stoplists for both languages using multiple methods"""
        # Regex-based stoplists
        self.ewe_stopwords.update(self.regex_based_stoplist(ewe_corpus, 'ewe'))
        self.english_stopwords.update(self.regex_based_stoplist(english_corpus, 'english'))
        
        # Frequency-based stoplists
        self.ewe_stopwords.update(self.frequency_based_stoplist(ewe_corpus, 'ewe'))
        self.english_stopwords.update(self.frequency_based_stoplist(english_corpus, 'english'))
        
        return self.ewe_stopwords, self.english_stopwords
    
    def save_stoplists(self, output_dir):
        """Save generated stoplists to files"""
        with open(f"{output_dir}/ewe_stopwords.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(self.ewe_stopwords)))
        
        with open(f"{output_dir}/english_stopwords.txt", 'w', encoding='utf-8') as f:
            f.write('\n'.join(sorted(self.english_stopwords)))