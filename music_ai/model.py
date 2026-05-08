import numpy as np
import torch
import torch.nn as nn
from torch.nn import Transformer, TransformerEncoder, TransformerEncoderLayer
from typing import List, Dict, Tuple
import json
from datetime import datetime

class LyricTokenizer:
    """Custom tokenizer for lyrics with music-specific tokens"""
    
    def __init__(self):
        self.word_to_idx = {}
        self.idx_to_word = {}
        self.special_tokens = {
            '[PAD]': 0,
            '[START]': 1,
            '[END]': 2,
            '[VERSE]': 3,
            '[CHORUS]': 4,
            '[BRIDGE]': 5,
            '[BREAK]': 6,
            '[UNK]': 7,
        }
        self.idx = len(self.special_tokens)
        self.idx_to_word = {v: k for k, v in self.special_tokens.items()}
    
    def add_word(self, word: str):
        word_lower = word.lower()
        if word_lower not in self.word_to_idx:
            self.word_to_idx[word_lower] = self.idx
            self.idx_to_word[self.idx] = word_lower
            self.idx += 1
    
    def encode(self, text: str, add_special_tokens=True) -> List[int]:
        tokens = []
        if add_special_tokens:
            tokens.append(self.special_tokens['[START]'])
        
        words = text.lower().split()
        for word in words:
            # Remove punctuation for now, keep it simple
            clean_word = ''.join(c for c in word if c.isalnum() or c in "-'")
            if clean_word:
                self.add_word(clean_word)
                tokens.append(self.word_to_idx[clean_word])
        
        if add_special_tokens:
            tokens.append(self.special_tokens['[END]'])
        
        return tokens
    
    def decode(self, tokens: List[int]) -> str:
        words = [self.idx_to_word.get(idx, '[UNK]') for idx in tokens]
        # Filter out special tokens for display
        words = [w for w in words if not w.startswith('[')]
        return ' '.join(words)

class MusicEmbedding(nn.Module):
    """Enhanced embeddings with music-specific features"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, max_seq_len: int = 512):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.genre_embedding = nn.Embedding(10, embedding_dim)  # 10 genres
        self.mood_embedding = nn.Embedding(10, embedding_dim)   # 10 moods
        self.section_embedding = nn.Embedding(5, embedding_dim) # verse, chorus, bridge, etc
        
        self.embedding_dim = embedding_dim
    
    def forward(self, token_ids: torch.Tensor, 
                genre_id: int = 0, 
                mood_id: int = 0,
                section_ids: torch.Tensor = None) -> torch.Tensor:
        seq_len = token_ids.size(1)
        pos_ids = torch.arange(seq_len, device=token_ids.device).unsqueeze(0)
        
        # Token embeddings
        token_emb = self.token_embedding(token_ids)
        pos_emb = self.position_embedding(pos_ids)
        
        # Add genre and mood context
        genre_emb = self.genre_embedding(torch.tensor(genre_id, device=token_ids.device)).unsqueeze(0).unsqueeze(0)
        mood_emb = self.mood_embedding(torch.tensor(mood_id, device=token_ids.device)).unsqueeze(0).unsqueeze(0)
        
        # Combine embeddings
        embeddings = token_emb + pos_emb + genre_emb + mood_emb
        
        if section_ids is not None:
            section_emb = self.section_embedding(section_ids)
            embeddings = embeddings + section_emb
        
        return embeddings

class MusicTransformer(nn.Module):
    """Transformer-based music lyric generator"""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int, 
                 num_layers: int, ff_dim: int, max_seq_len: int = 512):
        super().__init__()
        
        self.embedding = MusicEmbedding(vocab_size, embedding_dim, max_seq_len)
        
        encoder_layer = TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=ff_dim,
            dropout=0.1,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers)
        self.output_projection = nn.Linear(embedding_dim, vocab_size)
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
    
    def forward(self, token_ids: torch.Tensor, 
                genre_id: int = 0, 
                mood_id: int = 0,
                section_ids: torch.Tensor = None) -> torch.Tensor:
        
        embeddings = self.embedding(token_ids, genre_id, mood_id, section_ids)
        
        # Create attention mask for padding
        mask = (token_ids == 0)  # PAD token
        
        encoded = self.transformer_encoder(embeddings, src_key_padding_mask=mask)
        logits = self.output_projection(encoded)
        
        return logits
    
    def generate(self, prompt_ids: List[int], 
                 genre_id: int, 
                 mood_id: int,
                 max_length: int = 200,
                 temperature: float = 0.8,
                 top_k: int = 50) -> List[int]:
        """Generate lyrics token by token"""
        
        self.eval()
        generated = prompt_ids.copy()
        
        with torch.no_grad():
            for _ in range(max_length - len(prompt_ids)):
                token_tensor = torch.tensor([generated], dtype=torch.long, device=next(self.parameters()).device)
                
                logits = self(token_tensor, genre_id, mood_id)
                logits = logits[:, -1, :] / temperature
                
                # Top-k sampling
                top_k_logits, top_k_indices = torch.topk(logits, min(top_k, self.vocab_size))
                probs = torch.softmax(top_k_logits, dim=-1)
                
                next_token_idx = torch.multinomial(probs, 1).item()
                next_token = top_k_indices[0, next_token_idx].item()
                
                generated.append(next_token)
                
                if next_token == 2:  # [END] token
                    break
        
        return generated

class LyricGenerator:
    """High-level API for lyric generation"""
    
    GENRES = {
        'pop': 0, 'rock': 1, 'hiphop': 2, 'country': 3, 
        'rnb': 4, 'electronic': 5, 'jazz': 6, 'classical': 7,
        'folk': 8, 'metal': 9
    }
    
    MOODS = {
        'happy': 0, 'sad': 1, 'angry': 2, 'melancholic': 3,
        'energetic': 4, 'calm': 5, 'romantic': 6, 'dark': 7,
        'uplifting': 8, 'mysterious': 9
    }
    
    RHYME_SCHEMES = ['AABB', 'ABAB', 'ABCB', 'AABBA']
    
    def __init__(self, vocab_size: int = 10000, embedding_dim: int = 256,
                 num_heads: int = 8, num_layers: int = 4, ff_dim: int = 1024):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.tokenizer = LyricTokenizer()
        self.model = MusicTransformer(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            ff_dim=ff_dim
        ).to(self.device)
        
        self.vocab_size = vocab_size
    
    def initialize_with_templates(self, template_lyrics: List[str]):
        """Initialize tokenizer with common words from template lyrics"""
        for lyrics in template_lyrics:
            words = lyrics.lower().split()
            for word in words:
                clean_word = ''.join(c for c in word if c.isalnum() or c in "-'")
                if clean_word:
                    self.tokenizer.add_word(clean_word)
    
    def generate(self, prompt: str, genre: str = 'pop', mood: str = 'happy',
                 length: int = 'verse', temperature: float = 0.8,
                 num_variations: int = 1) -> Dict:
        """Generate lyrics with given parameters"""
        
        genre_id = self.GENRES.get(genre.lower(), 0)
        mood_id = self.MOODS.get(mood.lower(), 0)
        
        # Determine max length based on section type
        length_map = {
            'verse': 100,
            'chorus': 80,
            'bridge': 60,
            'full': 300
        }
        max_len = length_map.get(length, 100)
        
        # Encode prompt
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=True)
        
        # Ensure prompt isn't too long
        if len(prompt_ids) > max_len // 2:
            prompt_ids = prompt_ids[:max_len // 2]
        
        results = []
        for _ in range(num_variations):
            generated_ids = self.model.generate(
                prompt_ids=prompt_ids,
                genre_id=genre_id,
                mood_id=mood_id,
                max_length=max_len,
                temperature=temperature,
                top_k=50
            )
            
            lyrics = self.tokenizer.decode(generated_ids)
            results.append(lyrics)
        
        return {
            'prompt': prompt,
            'genre': genre,
            'mood': mood,
            'lyrics': results,
            'generated_at': datetime.now().isoformat(),
            'num_variations': num_variations
        }
    
    def generate_full_song(self, theme: str, genre: str = 'pop', 
                          mood: str = 'happy') -> Dict:
        """Generate a complete song structure"""
        
        song = {
            'theme': theme,
            'genre': genre,
            'mood': mood,
            'structure': {}
        }
        
        # Generate each section
        song['structure']['intro'] = self.generate(
            f"{theme} intro", genre, mood, 'verse'
        )['lyrics'][0]
        
        song['structure']['verse1'] = self.generate(
            f"{theme}", genre, mood, 'verse'
        )['lyrics'][0]
        
        song['structure']['chorus'] = self.generate(
            f"{theme} chorus hook", genre, mood, 'chorus'
        )['lyrics'][0]
        
        song['structure']['verse2'] = self.generate(
            f"{theme} part two", genre, mood, 'verse'
        )['lyrics'][0]
        
        song['structure']['bridge'] = self.generate(
            f"{theme} bridge break", genre, mood, 'bridge'
        )['lyrics'][0]
        
        song['structure']['outro'] = self.generate(
            f"{theme} outro ending", genre, mood, 'verse'
        )['lyrics'][0]
        
        return song
    
    def save_model(self, path: str):
        """Save model weights"""
        torch.save(self.model.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model weights"""
        self.model.load_state_dict(torch.load(path, map_location=self.device))
