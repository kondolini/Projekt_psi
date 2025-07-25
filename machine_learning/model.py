import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import numpy as np


class GreyhoundRacingModel(nn.Module):
    """
    V1 Model Architecture:
    - Race-level features → Dense layers
    - Per-dog features: Static + RNN for history + Commentary embeddings
    - Output: Win probabilities p_i for each dog
    """
    
    def __init__(
        self,
        # Vocabulary sizes for embeddings
        num_tracks: int,
        num_classes: int,
        num_categories: int,
        num_trainers: int,
        num_going_conditions: int,
        commentary_vocab_size: int,
        
        # Model hyperparameters
        embedding_dim: int = 32,
        hidden_dim: int = 64,
        rnn_hidden_dim: int = 32,
        commentary_embed_dim: int = 16,
        max_history_length: int = 10,
        max_commentary_length: int = 5,
        dropout_rate: float = 0.2,
        max_dogs_per_race: int = 8
    ):
        super().__init__()
        
        # Store hyperparameters
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.rnn_hidden_dim = rnn_hidden_dim
        self.max_history_length = max_history_length
        self.max_commentary_length = max_commentary_length
        self.max_dogs_per_race = max_dogs_per_race
        
        # === Race-level Embeddings ===
        self.track_embedding = nn.Embedding(num_tracks, embedding_dim)
        self.class_embedding = nn.Embedding(num_classes, embedding_dim)
        self.category_embedding = nn.Embedding(num_categories, embedding_dim)
        
        # Race-level dense features: day_of_week, month, hour, minute, distance_norm, temp_norm, humidity_norm, rainfall_sum
        race_dense_dim = 8
        race_embed_dim = 3 * embedding_dim  # track + class + category
        
        self.race_processor = nn.Sequential(
            nn.Linear(race_dense_dim + race_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU()
        )
        
        # === Dog-level Features ===
        
        # Static dog embeddings
        self.trainer_embedding = nn.Embedding(num_trainers, embedding_dim)
        self.dog_id_embedding = nn.Embedding(100000, embedding_dim)  # Hash space for dog IDs
        
        # Dog static features: weight_norm, age_days_norm, trap_number_norm (3 features)
        dog_static_dim = 3
        dog_embed_dim = 2 * embedding_dim  # trainer + dog_id
        
        self.dog_static_processor = nn.Sequential(
            nn.Linear(dog_static_dim + dog_embed_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # === Race History RNN ===
        
        # History features per race: position_norm, distance_norm, time_norm (3 features)
        history_dense_dim = 3
        self.going_embedding = nn.Embedding(num_going_conditions, embedding_dim // 2)
        
        # Commentary embeddings
        self.commentary_embedding = nn.Embedding(commentary_vocab_size, commentary_embed_dim)
        self.commentary_processor = nn.Sequential(
            nn.Linear(max_commentary_length * commentary_embed_dim, embedding_dim // 2),
            nn.ReLU()
        )
        
        # RNN input: dense features + going embedding + commentary embedding
        rnn_input_dim = history_dense_dim + (embedding_dim // 2) + (embedding_dim // 2)
        
        self.history_rnn = nn.LSTM(
            input_size=rnn_input_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=2,
            batch_first=True,
            dropout=dropout_rate if rnn_hidden_dim > 1 else 0
        )
        
        # === Final Prediction Layers ===
        
        # Combine all features: race_features + dog_static + rnn_output
        combined_dim = hidden_dim + hidden_dim + rnn_hidden_dim
        
        self.final_processor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim // 2, 1)  # Single output: win probability logit
        )
        
    def forward(self, batch_data: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        Forward pass through the model
        
        Args:
            batch_data: Dictionary containing all input tensors
            
        Returns:
            win_probabilities: [batch_size, max_dogs_per_race] probabilities for each dog
        """
        batch_size = batch_data['race_features'].size(0)
        
        # === Process Race-level Features ===
        race_dense = batch_data['race_features']  # [batch_size, race_dense_dim]
        track_ids = batch_data['track_ids']  # [batch_size]
        class_ids = batch_data['class_ids']  # [batch_size]
        category_ids = batch_data['category_ids']  # [batch_size]
        
        # Embeddings
        track_emb = self.track_embedding(track_ids)  # [batch_size, embedding_dim]
        class_emb = self.class_embedding(class_ids)  # [batch_size, embedding_dim]
        category_emb = self.category_embedding(category_ids)  # [batch_size, embedding_dim]
        
        # Concatenate race features
        race_combined = torch.cat([race_dense, track_emb, class_emb, category_emb], dim=1)
        race_features = self.race_processor(race_combined)  # [batch_size, hidden_dim]
        
        # === Process Dog-level Features ===
        dog_dense = batch_data['dog_features']  # [batch_size, max_dogs, dog_static_dim]
        trainer_ids = batch_data['trainer_ids']  # [batch_size, max_dogs]
        dog_ids = batch_data['dog_ids']  # [batch_size, max_dogs]
        dog_mask = batch_data['dog_mask']  # [batch_size, max_dogs] - 1 for valid dogs, 0 for padding
        
        # Embeddings
        trainer_emb = self.trainer_embedding(trainer_ids)  # [batch_size, max_dogs, embedding_dim]
        dog_emb = self.dog_id_embedding(dog_ids)  # [batch_size, max_dogs, embedding_dim]
        
        # Concatenate dog static features
        dog_combined = torch.cat([dog_dense, trainer_emb, dog_emb], dim=2)
        dog_static_features = self.dog_static_processor(dog_combined)  # [batch_size, max_dogs, hidden_dim]
        
        # === Process Race History ===
        history_dense = batch_data['history_features']  # [batch_size, max_dogs, max_history, history_dense_dim]
        going_ids = batch_data['going_ids']  # [batch_size, max_dogs, max_history]
        commentary_ids = batch_data['commentary_ids']  # [batch_size, max_dogs, max_history, max_commentary]
        history_mask = batch_data['history_mask']  # [batch_size, max_dogs, max_history]
        
        # Reshape for batch processing
        bs, max_dogs, max_hist, _ = history_dense.shape
        history_dense_flat = history_dense.view(bs * max_dogs, max_hist, -1)
        going_ids_flat = going_ids.view(bs * max_dogs, max_hist)
        commentary_ids_flat = commentary_ids.view(bs * max_dogs, max_hist, -1)
        history_mask_flat = history_mask.view(bs * max_dogs, max_hist)
        
        # Going embeddings
        going_emb = self.going_embedding(going_ids_flat)  # [bs*max_dogs, max_hist, embedding_dim//2]
        
        # Commentary embeddings and processing
        commentary_emb = self.commentary_embedding(commentary_ids_flat)  # [bs*max_dogs, max_hist, max_commentary, commentary_embed_dim]
        commentary_emb_flat = commentary_emb.view(bs * max_dogs, max_hist, -1)  # Flatten last two dims
        commentary_features = self.commentary_processor(commentary_emb_flat)  # [bs*max_dogs, max_hist, embedding_dim//2]
        
        # Combine history features
        history_combined = torch.cat([history_dense_flat, going_emb, commentary_features], dim=2)
        
        # RNN processing
        history_output, _ = self.history_rnn(history_combined)  # [bs*max_dogs, max_hist, rnn_hidden_dim]
        
        # Get last valid output for each sequence
        # Use history_mask to find last valid timestep
        last_indices = history_mask_flat.sum(dim=1) - 1  # [bs*max_dogs]
        last_indices = torch.clamp(last_indices, min=0)  # Handle empty sequences
        
        # Gather last outputs
        batch_indices = torch.arange(bs * max_dogs, device=history_output.device)
        history_final = history_output[batch_indices, last_indices]  # [bs*max_dogs, rnn_hidden_dim]
        
        # Reshape back
        history_final = history_final.view(bs, max_dogs, -1)  # [batch_size, max_dogs, rnn_hidden_dim]
        
        # === Combine All Features ===
        # Expand race features to match dog dimension
        race_features_expanded = race_features.unsqueeze(1).expand(-1, max_dogs, -1)  # [batch_size, max_dogs, hidden_dim]
        
        # Concatenate all features
        all_features = torch.cat([race_features_expanded, dog_static_features, history_final], dim=2)
        
        # Final prediction
        logits = self.final_processor(all_features).squeeze(-1)  # [batch_size, max_dogs]
        
        # Apply mask to logits (set padded dogs to very negative values)
        masked_logits = logits.masked_fill(~dog_mask.bool(), -1e9)
        
        # Convert to probabilities
        win_probabilities = F.softmax(masked_logits, dim=1)  # [batch_size, max_dogs]
        
        return win_probabilities

    def get_model_info(self):
        """Return model architecture information"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'embedding_dim': self.embedding_dim,
            'hidden_dim': self.hidden_dim,
            'rnn_hidden_dim': self.rnn_hidden_dim,
            'max_history_length': self.max_history_length,
            'max_dogs_per_race': self.max_dogs_per_race
        }
