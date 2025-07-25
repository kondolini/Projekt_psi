#!/usr/bin/env python3
"""
Cache Manager for Greyhound Racing ML Pipeline

Handles caching of processed datasets, encoders, and vocabulary to avoid
reprocessing data on every training run.
"""

import os
import sys
import pickle
import hashlib
import json
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
import torch
from tqdm import tqdm

# Add parent directory for imports - make it more robust
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from models.race import Race
from models.dog import Dog

logger = logging.getLogger(__name__)


class CacheManager:
    """
    Manages caching of processed data for the ML pipeline
    
    Cache structure:
    cache/
    ├── datasets/
    │   ├── processed_data_<hash>.pkl
    │   └── metadata.json
    ├── encoders/
    │   ├── encoders_<hash>.pkl
    │   └── vocab_sizes_<hash>.json
    └── cache_info.json
    """
    
    def __init__(self, cache_dir: str = "cache"):
        """
        Initialize cache manager
        
        Args:
            cache_dir: Base directory for cache files
        """
        # Make cache_dir relative to project root if not absolute
        if not os.path.isabs(cache_dir):
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            cache_dir = os.path.join(project_root, cache_dir)
            
        self.cache_dir = cache_dir
        self.datasets_dir = os.path.join(cache_dir, "datasets")
        self.encoders_dir = os.path.join(cache_dir, "encoders")
        
        # Create cache directories
        os.makedirs(self.datasets_dir, exist_ok=True)
        os.makedirs(self.encoders_dir, exist_ok=True)
        
        self.cache_info_path = os.path.join(cache_dir, "cache_info.json")
        self.cache_info = self._load_cache_info()
        
        logger.info(f"Cache manager initialized: {cache_dir}")
    
    def _load_cache_info(self) -> Dict:
        """Load cache information"""
        if os.path.exists(self.cache_info_path):
            try:
                with open(self.cache_info_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load cache info: {e}")
        return {}
    
    def _save_cache_info(self):
        """Save cache information"""
        try:
            with open(self.cache_info_path, 'w') as f:
                json.dump(self.cache_info, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Could not save cache info: {e}")
    
    def _compute_data_hash(self, data_dirs: Dict[str, str], config: Dict) -> str:
        """
        Compute hash of data directories and configuration for cache key
        
        Args:
            data_dirs: Dictionary with paths to data directories
            config: Configuration parameters that affect processing
            
        Returns:
            Hash string for cache key
        """
        # Get modification times of key files
        hash_components = []
        
        for dir_name, dir_path in data_dirs.items():
            if os.path.exists(dir_path):
                # Get directory modification time
                mtime = os.path.getmtime(dir_path)
                hash_components.append(f"{dir_name}:{mtime}")
                
                # For critical directories, also check some file contents
                if dir_name == "unified" and os.path.exists(os.path.join(dir_path, "race_index.pkl")):
                    index_mtime = os.path.getmtime(os.path.join(dir_path, "race_index.pkl"))
                    hash_components.append(f"race_index:{index_mtime}")
        
        # Add configuration that affects processing
        config_str = json.dumps(sorted(config.items()), sort_keys=True)
        hash_components.append(f"config:{config_str}")
        
        # Create hash
        full_string = "|".join(hash_components)
        return hashlib.md5(full_string.encode()).hexdigest()[:12]
    
    def get_cached_datasets(self, data_dirs: Dict[str, str], config: Dict) -> Optional[Tuple[Any, Any, Dict]]:
        """
        Try to load cached processed datasets
        
        Args:
            data_dirs: Dictionary with data directory paths
            config: Processing configuration
            
        Returns:
            Tuple of (train_data, val_data, metadata) if cache hit, None if miss
        """
        cache_key = self._compute_data_hash(data_dirs, config)
        cache_file = os.path.join(self.datasets_dir, f"processed_data_{cache_key}.pkl")
        metadata_file = os.path.join(self.datasets_dir, f"metadata_{cache_key}.json")
        
        if os.path.exists(cache_file) and os.path.exists(metadata_file):
            try:
                logger.info(f"Loading cached datasets: {cache_key}")
                
                # Load metadata first
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                
                # Load processed data
                with open(cache_file, 'rb') as f:
                    data = pickle.load(f)
                
                logger.info(f"✅ Cache hit! Loaded cached data from {metadata['created_at']}")
                return data['train_data'], data['val_data'], metadata
                
            except Exception as e:
                logger.warning(f"Could not load cached datasets: {e}")
                # Clean up corrupted cache files
                for f in [cache_file, metadata_file]:
                    if os.path.exists(f):
                        os.remove(f)
        
        logger.info("❌ Cache miss - will process data from scratch")
        return None
    
    def save_datasets(self, train_data: Any, val_data: Any, data_dirs: Dict[str, str], 
                     config: Dict, metadata: Dict):
        """
        Save processed datasets to cache
        
        Args:
            train_data: Processed training data
            val_data: Processed validation data  
            data_dirs: Data directory paths
            config: Processing configuration
            metadata: Additional metadata to save
        """
        cache_key = self._compute_data_hash(data_dirs, config)
        cache_file = os.path.join(self.datasets_dir, f"processed_data_{cache_key}.pkl")
        metadata_file = os.path.join(self.datasets_dir, f"metadata_{cache_key}.json")
        
        try:
            logger.info(f"Caching processed datasets: {cache_key}")
            
            # Save processed data
            data_to_save = {
                'train_data': train_data,
                'val_data': val_data
            }
            
            with open(cache_file, 'wb') as f:
                pickle.dump(data_to_save, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save metadata
            full_metadata = {
                'cache_key': cache_key,
                'created_at': datetime.now().isoformat(),
                'data_dirs': data_dirs,
                'config': config,
                **metadata
            }
            
            with open(metadata_file, 'w') as f:
                json.dump(full_metadata, f, indent=2, default=str)
            
            # Update cache info
            self.cache_info[cache_key] = {
                'type': 'datasets',
                'created_at': full_metadata['created_at'],
                'size_mb': os.path.getsize(cache_file) / 1024 / 1024
            }
            self._save_cache_info()
            
            logger.info(f"✅ Datasets cached successfully")
            
        except Exception as e:
            logger.error(f"Could not save datasets to cache: {e}")
            # Clean up partially written files
            for f in [cache_file, metadata_file]:
                if os.path.exists(f):
                    os.remove(f)
    
    def get_cached_encoders(self, data_dirs: Dict[str, str], config: Dict) -> Optional[Tuple[Dict, Dict]]:
        """
        Try to load cached encoders and vocabulary sizes
        
        Args:
            data_dirs: Data directory paths
            config: Processing configuration
            
        Returns:
            Tuple of (encoders_dict, vocab_sizes_dict) if cache hit, None if miss
        """
        cache_key = self._compute_data_hash(data_dirs, config)
        encoders_file = os.path.join(self.encoders_dir, f"encoders_{cache_key}.pkl")
        vocab_file = os.path.join(self.encoders_dir, f"vocab_sizes_{cache_key}.json")
        
        if os.path.exists(encoders_file) and os.path.exists(vocab_file):
            try:
                logger.info(f"Loading cached encoders: {cache_key}")
                
                # Load encoders
                with open(encoders_file, 'rb') as f:
                    encoders = pickle.load(f)
                
                # Load vocabulary sizes
                with open(vocab_file, 'r') as f:
                    vocab_sizes = json.load(f)
                
                logger.info("✅ Encoders loaded from cache")
                return encoders, vocab_sizes
                
            except Exception as e:
                logger.warning(f"Could not load cached encoders: {e}")
                # Clean up corrupted cache files
                for f in [encoders_file, vocab_file]:
                    if os.path.exists(f):
                        os.remove(f)
        
        logger.info("❌ Encoder cache miss - will build encoders from scratch")
        return None
    
    def save_encoders(self, encoders: Dict, vocab_sizes: Dict, data_dirs: Dict[str, str], config: Dict):
        """
        Save encoders and vocabulary sizes to cache
        
        Args:
            encoders: Dictionary of trained encoders
            vocab_sizes: Dictionary of vocabulary sizes
            data_dirs: Data directory paths
            config: Processing configuration
        """
        cache_key = self._compute_data_hash(data_dirs, config)
        encoders_file = os.path.join(self.encoders_dir, f"encoders_{cache_key}.pkl")
        vocab_file = os.path.join(self.encoders_dir, f"vocab_sizes_{cache_key}.json")
        
        try:
            logger.info(f"Caching encoders: {cache_key}")
            
            # Save encoders
            with open(encoders_file, 'wb') as f:
                pickle.dump(encoders, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Save vocabulary sizes
            with open(vocab_file, 'w') as f:
                json.dump(vocab_sizes, f, indent=2)
            
            # Update cache info
            self.cache_info[cache_key + "_encoders"] = {
                'type': 'encoders',
                'created_at': datetime.now().isoformat(),
                'size_mb': os.path.getsize(encoders_file) / 1024 / 1024
            }
            self._save_cache_info()
            
            logger.info("✅ Encoders cached successfully")
            
        except Exception as e:
            logger.error(f"Could not save encoders to cache: {e}")
            # Clean up partially written files
            for f in [encoders_file, vocab_file]:
                if os.path.exists(f):
                    os.remove(f)
    
    def clear_cache(self, cache_type: Optional[str] = None):
        """
        Clear cache files
        
        Args:
            cache_type: Type of cache to clear ('datasets', 'encoders', or None for all)
        """
        dirs_to_clear = []
        
        if cache_type is None or cache_type == 'datasets':
            dirs_to_clear.append(self.datasets_dir)
        if cache_type is None or cache_type == 'encoders':
            dirs_to_clear.append(self.encoders_dir)
        
        total_cleared = 0
        for cache_dir in dirs_to_clear:
            if os.path.exists(cache_dir):
                for filename in os.listdir(cache_dir):
                    file_path = os.path.join(cache_dir, filename)
                    os.remove(file_path)
                    total_cleared += 1
        
        # Clear cache info
        if cache_type is None:
            self.cache_info = {}
        else:
            self.cache_info = {k: v for k, v in self.cache_info.items() 
                             if v.get('type') != cache_type}
        
        self._save_cache_info()
        logger.info(f"Cleared {total_cleared} cache files ({cache_type or 'all'})")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics"""
        stats = {
            'total_entries': len(self.cache_info),
            'total_size_mb': 0,
            'by_type': {}
        }
        
        for entry in self.cache_info.values():
            cache_type = entry.get('type', 'unknown')
            size_mb = entry.get('size_mb', 0)
            
            if cache_type not in stats['by_type']:
                stats['by_type'][cache_type] = {'count': 0, 'size_mb': 0}
            
            stats['by_type'][cache_type]['count'] += 1
            stats['by_type'][cache_type]['size_mb'] += size_mb
            stats['total_size_mb'] += size_mb
        
        return stats
    
    def print_cache_stats(self):
        """Print cache statistics"""
        stats = self.get_cache_stats()
        
        print(f"\n📊 Cache Statistics:")
        print(f"   - Total entries: {stats['total_entries']}")
        print(f"   - Total size: {stats['total_size_mb']:.1f} MB")
        
        for cache_type, type_stats in stats['by_type'].items():
            print(f"   - {cache_type}: {type_stats['count']} entries, {type_stats['size_mb']:.1f} MB")
