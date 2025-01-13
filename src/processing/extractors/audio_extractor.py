from typing import List, Dict, Any
import speech_recognition as sr
from .base import BaseExtractor

class AudioExtractor(BaseExtractor):
    """Extract text from audio files using speech recognition."""
    
    def __init__(self):
        """Initialize speech recognizer."""
        self.recognizer = sr.Recognizer()
        
    def extract_text(self, file_path: str) -> str:
        """Extract text from audio using speech recognition."""
        try:
            with sr.AudioFile(file_path) as source:
                audio = self.recognizer.record(source)
                text = self.recognizer.recognize_google(audio)
                return text.strip()
        except Exception as e:
            raise ValueError(f"Error extracting text from audio: {str(e)}")
            
    def extract_media(self, file_path: str) -> List[Dict[str, Any]]:
        """Return audio file metadata."""
        try:
            with sr.AudioFile(file_path) as source:
                return [{
                    'type': 'audio',
                    'format': file_path.split('.')[-1].lower(),
                    'duration': source.DURATION
                }]
        except Exception as e:
            raise ValueError(f"Error extracting audio metadata: {str(e)}")
            
    def __del__(self):
        """Clean up resources."""
        pass
