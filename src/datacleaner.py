import re
import os
import sys


class TextCleaner:
    """text cleaner using indic-nlp"""
    
    def __init__(self):
        self.setup_indicnlp()
        self.normalizers = {}
    
    def setup_indicnlp(self):
        """Setup IndicNLP library"""
        try:
            # add to path
            indic_path = "indic_nlp_library"  # Adjust path as needed
            if os.path.exists(indic_path) and indic_path not in sys.path:
                sys.path.append(indic_path)

            from indicnlp import common
            from indicnlp import loader
            from indicnlp.normalize.indic_normalize import BaseNormalizer
            
            # resources path
            common.set_resources_path("indic_nlp_resources")  # Adjust path as needed
            loader.load()
            
            self.BaseNormalizer = BaseNormalizer
            self.indicnlp_available = True
            
        except Exception as e:
            print(f"IndicNLP not available: {e}")
            self.indicnlp_available = False

    def get_normalizer(self, lang):
        """normalizer for language"""
        if not self.indicnlp_available or lang == 'en':
            return None
            
        if lang not in self.normalizers:
            try:
                self.normalizers[lang] = self.BaseNormalizer(lang, remove_nuktas=False)
            except:
                self.normalizers[lang] = None
        
        return self.normalizers[lang]
    
    def clean_text(self, text, lang='en'):
        """
        Clean text for given language
        
        Args:
            text (str): Input text
            lang (str): Language code ('en', 'hi', 'bn')
        
        Returns:
            str: Cleaned text
        """
        if not text or not isinstance(text, str):
            return ""
        
        text = text.strip()
        
        # indic-nlp normalizer for indic languages
        if lang != 'en':
            normalizer = self.get_normalizer(lang)
            if normalizer:
                try:
                    text = normalizer.normalize(text)
                except:
                    pass
        
        # remove extra whitespaces
        text = re.sub(r'\s+', ' ', text)
        
        # character filtering
        if lang == 'en':
            # keep only english alphanumeric and basic punctuation
            text = re.sub(r'[^\w\s\.\,\!\?\:\;\-\'\"]', ' ', text)
        else:
            # indic languages keep most characters, remove control characters and some symbols
            text = re.sub(r'[\x00-\x1f\x7f-\x9f]', ' ', text)
            text = re.sub(r'[#$%&*+=<>@\[\]\\^_`{|}~]', ' ', text)
        
        text = re.sub(r'\s+', ' ', text).strip()

        return text


# Usage
# cleaner = TextCleaner()

# english_text = "Hello!!!   This is a test... with extra spaces."
# hindi_text = "नमस्ते!!!   यह एक परीक्षण है... अतिरिक्त स्थान के साथ।"

# print("Original English:", english_text)
# print("Cleaned English:", cleaner.clean_text(english_text, 'en'))

# print("\nOriginal Hindi:", hindi_text)
# print("Cleaned Hindi:", cleaner.clean_text(hindi_text, 'hi'))
