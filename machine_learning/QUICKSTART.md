# Quick Start Guide for Coworkers

## 🚀 Getting Started

### 1. **Check Your Environment**
```bash
# Test Python and packages
python machine_learning/check_requirements.py

# Quick model test
python machine_learning/test_model.py
```

### 2. **View Cache Status**
```bash
# See what's cached
python machine_learning/train.py --cache_stats
```

### 3. **Training Options**

#### First Time (builds cache):
```bash
# Small test run
python machine_learning/train.py --data_dir data --max_races 100 --epochs 1

# Full training (takes ~10 min first time)
python machine_learning/train.py --data_dir data --epochs 50 --batch_size 32
```

#### Subsequent Runs (uses cache):
```bash
# Same command, but loads cache (~30 seconds)
python machine_learning/train.py --data_dir data --epochs 50 --batch_size 32
```

### 4. **Cache Management**
```bash
# Force rebuild everything
python machine_learning/train.py --rebuild_cache --data_dir data --epochs 50

# Clear all cached data
python machine_learning/train.py --clear_cache

# Just show cache info
python machine_learning/train.py --cache_stats
```

## 🎯 Key Benefits

- **Fast iteration**: First run builds cache, subsequent runs load in seconds
- **Better feedback**: Progress bars show exactly what's happening
- **Robust imports**: Works regardless of where you run it from
- **Parameter tracking**: See model size and complexity clearly

## 🔧 Troubleshooting

**Import errors?**
- Run `python machine_learning/check_requirements.py`
- Install missing packages: `pip install -r requirements_ai.txt`

**Slow startup?**
- First run builds cache (normal)
- Use `--max_races 100` for quick testing

**Want to start fresh?**
- Use `--clear_cache` to rebuild everything

## 📊 What's New

- ✅ **Progress bars** for all data processing
- ✅ **Smart caching** saves processed data
- ✅ **Parameter counting** shows model complexity
- ✅ **Robust imports** work anywhere
- ✅ **Better error messages** with clear fixes

The cache system means you only wait once for data processing, then everything is fast!
