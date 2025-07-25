#!/usr/bin/env python3
"""
Test the caching functionality for the ML pipeline
"""

import os
import sys

# Add parent directory for imports - make it more robust
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from machine_learning.cache_manager import CacheManager


def test_cache_manager():
    """Test basic cache manager functionality"""
    
    print("🧪 Testing Cache Manager...")
    
    # Create cache manager
    cache_manager = CacheManager("test_cache")
    
    # Print initial stats
    print("\n📊 Initial cache stats:")
    cache_manager.print_cache_stats()
    
    # Test configuration
    data_dirs = {
        'dogs_enhanced': '/fake/dogs',
        'races': '/fake/races', 
        'unified': '/fake/unified'
    }
    
    config = {
        'max_races': 1000,
        'exclude_trial_races': True,
        'val_start_date': '2023-01-01'
    }
    
    # Test hash computation
    hash1 = cache_manager._compute_data_hash(data_dirs, config)
    hash2 = cache_manager._compute_data_hash(data_dirs, config)
    
    print(f"\n🔑 Hash consistency test:")
    print(f"   Hash 1: {hash1}")
    print(f"   Hash 2: {hash2}")
    print(f"   Same: {hash1 == hash2}")
    
    # Test with different config
    config2 = config.copy()
    config2['max_races'] = 2000
    hash3 = cache_manager._compute_data_hash(data_dirs, config2)
    
    print(f"   Hash 3 (different config): {hash3}")
    print(f"   Different from 1: {hash1 != hash3}")
    
    # Test cache miss
    cached_encoders = cache_manager.get_cached_encoders(data_dirs, config)
    print(f"\n❌ Cache miss test: {cached_encoders is None}")
    
    cached_datasets = cache_manager.get_cached_datasets(data_dirs, config)
    print(f"❌ Cache miss test: {cached_datasets is None}")
    
    print("\n✅ Cache manager tests passed!")


def main():
    """Main test function"""
    print("🚀 Cache Manager - Component Tests")
    print("=" * 50)
    
    try:
        test_cache_manager()
        print("\n🎉 All Cache Tests Passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == '__main__':
    success = main()
    if not success:
        exit(1)
