import pandas as pd
import numpy as np
import random
from tqdm import tqdm

# Hyper Large Ewe Verb and Function Word Lists
EWE_VERBS = {
    'ɖo': 'go', 'ɖi': 'go (past)', 'va': 'come', 'vɛ': 'come to', 'dzo': 'leave', 'dzɔ': 'depart', 
    'ɖu': 'run', 'tsɔ': 'jump', 'zɔ': 'walk', 'zɔzɔ': 'stroll', 'ɖe': 'reach', 'ɖa': 'move', 
    'kɔ': 'climb', 'sɔ': 'descend', 'fli': 'fly', 'no': 'stay', 'gbɔ': 'return', 'gbɔgbɔ': 'come back', 
    'tsu': 'chase', 'kpa': 'crawl', 'ɖɔ': 'arrive', 'nya': 'move toward', 'ƒo': 'hit', 'ƒɔ': 'strike', 
    'tso': 'cut', 'tsɔ': 'chop', 'ɖu': 'eat', 'nɔ': 'drink', 'ɖe': 'put', 'tsɔ': 'take', 'ƒle': 'buy', 
    'ɖo': 'sell', 'kpɔ': 'see', 'kpɔkpɔ': 'look', 'gblɔ': 'say', 'ŋlɔ': 'write', 'ŋɔ': 'draw', 
    'dɔ': 'work', 'dɔdɔ': 'labor', 'kplɔ': 'sweep', 'tɔ': 'grind', 'tu': 'build', 'tutu': 'construct', 
    'fa': 'sew', 'ɖa': 'cook', 'ɖaɖa': 'prepare food', 'kpe': 'dig', 'ƒu': 'pull', 'te': 'push', 
    'gɔ': 'break', 'sɛ': 'tear', 'vɔ': 'finish', 'kaka': 'scatter', 'ɖi': 'throw', 'tsa': 'tie', 
    'vo': 'untie', 'kplo': 'carry on head', 'ɖe': 'carry on back', 'di': 'search', 'didi': 'seek', 
    'nya': 'know', 'ŋlɔ': 'tell', 'ŋu': 'call', 'gbe': 'greet', 'gble': 'curse', 'sɔ': 'pray', 
    'tsɔ': 'offer', 'kpe': 'help', 'kpeɖe': 'assist', 'ɖo': 'send', 'ɖɔ': 'order', 'ŋlɔŋlɔ': 'explain', 
    'tsɔtsɔ': 'share', 'ƒoƒo': 'forgive', 'dzedze': 'judge', 'sɛ': 'decide', 'ŋuŋu': 'invite', 
    'gblẽ': 'refuse', 'ɖa': 'promise', 'vava': 'agree', 'kpɔ': 'be visible', 'nɔ': 'be at', 
    'sɔ': 'exist', 'nyo': 'be good', 'nyui': 'be beautiful', 'gble': 'be bad', 'xɔ': 'receive', 
    'se': 'hear', 'seŋu': 'listen', 'nu': 'feel', 'nui': 'be sweet', 'gbɔ': 'be near', 
    'vɛ': 'be far', 'ɖi': 'be strong', 'ɖu': 'be weak', 'tsu': 'be tall', 'kɔ': 'be high', 
    'dɔ': 'be sick', 'dɔdɔ': 'be healthy', 'tsɛ': 'be young', 'gbã': 'be old', 'sĩ': 'be small', 
    'gã': 'be big', 'fɛ': 'be cold', 'dzi': 'be hot', 'nyɔ': 'be wet', 'kpa': 'be dry', 
    'susɔ': 'think', 'susu': 'reflect', 'xɔ': 'believe', 'di': 'want', 'didi': 'desire', 
    'ŋlɔ': 'learn', 'ŋɔ': 'forget', 'nya': 'understand', 'nyanya': 'realize', 'ɖo': 'hope', 
    'gbɔ': 'fear', 'gbɔgbɔ': 'be afraid', 'lɔ': 'love', 'lɔlɔ': 'cherish', 'sɛ': 'hate', 
    'tsɔ': 'trust', 'tsɔtsɔ': 'rely', 'ɖi': 'decide', 'kpe': 'doubt', 'dzi': 'be happy', 
    'dzidzɔ': 'rejoice', 'vɔ': 'be sad', 'vɔvɔ': 'mourn', 'ŋu': 'be angry', 'ŋuŋu': 'rage', 
    'va': 'come (serial)', 'ɖo': 'go (serial)', 'tsɔ': 'take (serial)', 'ɖe': 'put (serial)', 
    'kɔ': 'carry (serial)', 'gbɔ': 'return (serial)', 'di': 'search (serial)', 'tu': 'build (serial)', 
    'dɔ': 'work (serial)', 'vɔ': 'finish (serial)', 'tsi': 'use (serial)', 'sɔ': 'join (serial)', 
    'ɖom': 'going', 'ɖim': 'going (progressive)', 'ɖiɖi': 'went', 'vam': 'coming', 'vav': 'came', 
    'kpɔm': 'seeing', 'kpɔɖi': 'saw', 'dɔm': 'working', 'dɔɖi': 'worked', 'lɔm': 'loving', 
    'lɔɖi': 'loved', 'nyam': 'knowing', 'nyaɖi': 'knew', 'tum': 'building', 'tuɖi': 'built'
}

EWE_FUNCTION_WORDS = {
    'ɖe': 'on, at', 'na': 'to, for', 'le': 'in, at', 'me': 'in, inside', 'gbɔ': 'near, by', 
    'dzi': 'on, above', 'te': 'under', 'kpɔ': 'beside', 'ŋu': 'behind', 'ŋgɔ': 'before, front', 
    'tso': 'from', 'va': 'until', 'kple': 'with', 'sɔ': 'with (instrument)', 'la': 'at (place)', 
    'ƒe': 'of (possessive)', 'nu': 'at (general)', 'gome': 'underneath', 'ta': 'top', 'mɔ': 'way, path', 
    'ke': 'and', 'kaka': 'and then', 'ale': 'or', 'ne': 'if', 'etsɔ': 'because', 'gake': 'but', 
    'vɔ': 'so', 'sɔgbo': 'although', 'be': 'that (complementizer)', 'nenye': 'if only', 
    'tsɔ': 'since', 'kpɔ': 'as', 'ɖe': 'whether', 'vaɖe': 'until', 'se': 'when', 
    'sese': 'whenever', 'tsi': 'while', 'o': 'negative marker', 'a': 'question marker', 
    'm': 'progressive marker', 'ɖi': 'past marker', 'ɖa': 'future marker', 'e': 'focus marker', 
    'yè': 'logophoric pronoun', 'la': 'definite article', 'sia': 'this', 'esiae': 'this is', 
    'nenema': 'thus', 'ɖe': 'some', 'wò': 'it', 'ma': 'that', 'si': 'which', 'ɖo': 'already', 
    'tsɔ': 'still', 'me': 'I', 'mí': 'we', 'nè': 'you (sg)', 'míá': 'you (pl)', 'é': 'he/she/it', 
    'wó': 'they', 'yè': 'he/she (logophoric)', 'ama': 'someone', 'nuka': 'what', 'ɖe': 'one', 
    'wòa': 'self', 'míatɔ': 'ourselves', 'nètɔ': 'yourself', 'ŋutɔ': 'very', 'kata': 'all', 
    'ɖeka': 'one', 'eve': 'two', 'etɔ': 'three', 'tsi': 'again', 'kpɛ': 'only', 
    'sɔgbɔ': 'many', 'nyuie': 'well', 'vɔvɔ': 'slowly', 'kɔkɔ': 'quickly', 'ɖaŋ': 'quietly', 
    'ŋkeke': 'daily', 'ŋkekeɖe': 'sometimes', 'ɖoɖo': 'always', 'ɖokui': 'alone', 'fia': 'here', 
    'afima': 'there', 'nukae': 'why', 'aleke': 'how', 'ɖe': 'where', 'ɖetsi': 'when', 
    'ŋutɔe': 'truly', 'sɔsɔ': 'together', 'ɖiɖi': 'long', 'sia': 'this', 'ma': 'that', 
    'si': 'which', 'ɖe': 'some', 'ɖekae': 'any', 'kata': 'every', 'vovovo': 'different', 
    'nyɔ': 'other', 'tɔ': 'own', 'gbã': 'first', 'vɛ': 'last', 'sɔgbɔ': 'many', 
    'fɛ': 'few', 'ŋkeke': 'several', 'ɖekawo': 'ones', 'evewo': 'twos'
}

def augment_translation_data_ewe(df, src_col, tgt_col, techniques=None, dropout_prob=0.05, swap_prob=0.1):
    """Augment Ewe translation data with enhanced techniques
    
    Args:
        df: DataFrame with parallel text (Ewe as source)
        src_col: Source language column name (Ewe)
        tgt_col: Target language column name
        techniques: List of techniques to apply
        dropout_prob: Probability for word dropout
        swap_prob: Probability for word swap
    
    Returns:
        Augmented DataFrame
    """
    if techniques is None:
        techniques = ['word_dropout', 'word_swap', 'back_translation', 'function_insertion']
    
    augmented_data = [df]
    
    for technique in tqdm(techniques, desc="Applying augmentation techniques"):
        aug_df = df.copy()
        if technique == 'word_dropout':
            aug_df[src_col] = aug_df[src_col].apply(
                lambda x: smart_word_dropout(x, prob=dropout_prob)
            )
            augmented_data.append(aug_df)
            
        elif technique == 'word_swap':
            aug_df[src_col] = aug_df[src_col].apply(
                lambda x: smart_word_swap(x, prob=swap_prob)
            )
            augmented_data.append(aug_df)
            
        elif technique == 'back_translation':
            print("Warning: Real back translation unavailable for Ewe. Using simulation.")
            aug_df[src_col] = aug_df[src_col].apply(improved_simulate_back_translation)
            augmented_data.append(aug_df)
            
        elif technique == 'function_insertion':
            aug_df[src_col] = aug_df[src_col].apply(
                lambda x: function_insertion(x, prob=0.05)
            )
            augmented_data.append(aug_df)
    
    # Filter low-quality augmentations (length-based)
    result_df = pd.concat(augmented_data).reset_index(drop=True)
    result_df = result_df[
        result_df[src_col].apply(lambda x: len(x.split()) >= 3 and 
                                abs(len(x.split()) - len(df[src_col].iloc[0].split())) <= 3)
    ]
    
    return result_df.reset_index(drop=True)

def smart_word_dropout(text, prob=0.05, min_words=3):
    """Drop words, avoiding Ewe verbs"""
    if not text.strip():
        return text
    
    words = text.split()
    if len(words) <= min_words:
        return text
    
    new_words = [w for w in words if w in EWE_VERBS or random.random() > prob]
    if len(new_words) < min_words:
        return ' '.join(words[:min_words])
    return ' '.join(new_words)

def smart_word_swap(text, prob=0.1):
    """Swap words, avoiding verb disruption"""
    if not text.strip():
        return text
    
    words = text.split()
    if len(words) < 2:
        return text
    
    new_words = words.copy()
    indices = list(range(len(words) - 1))
    random.shuffle(indices)
    
    for i in indices:
        if random.random() < prob and words[i] not in EWE_VERBS and words[i + 1] not in EWE_VERBS:
            new_words[i], new_words[i + 1] = new_words[i + 1], new_words[i]
    return ' '.join(new_words)

def improved_simulate_back_translation(text, dropout_prob=0.05, reorder_prob=0.1):
    """Simulate back translation with reordering and function word insertion"""
    if not text.strip():
        return text
    
    words = text.split()
    # Dropout, protecting verbs
    new_words = [w for w in words if w in EWE_VERBS or random.random() > dropout_prob]
    
    # Reorder within a small window (mimic syntactic variation)
    if len(new_words) > 3 and random.random() < reorder_prob:
        window = min(3, len(new_words) - 1)
        i = random.randint(0, len(new_words) - window)
        j = i + random.randint(1, window)
        if new_words[i] not in EWE_VERBS and new_words[j] not in EWE_VERBS:
            new_words[i], new_words[j] = new_words[j], new_words[i]
    
    # Insert a function word (mimic translation artifacts)
    if random.random() < 0.05:
        insert_pos = random.randint(0, len(new_words))
        new_words.insert(insert_pos, random.choice(list(EWE_FUNCTION_WORDS)))
    
    return ' '.join(new_words)

def function_insertion(text, prob=0.05):
    """Insert Ewe function words at random positions"""
    if not text.strip():
        return text
    
    words = text.split()
    new_words = words.copy()
    num_insertions = max(1, int(len(words) * prob))
    
    for _ in range(num_insertions):
        word_to_insert = random.choice(list(EWE_FUNCTION_WORDS))
        insert_pos = random.randint(0, len(new_words))
        new_words.insert(insert_pos, word_to_insert)
    
    return ' '.join(new_words)