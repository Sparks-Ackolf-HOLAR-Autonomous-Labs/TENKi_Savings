import gc
import psutil
import os
from typing import Dict, Any
import json

def get_memory_usage() -> Dict[str, float]:
    """Get current memory usage statistics"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    
    return {
        'rss_mb': memory_info.rss / 1024 / 1024,  # Resident Set Size in MB
        'vms_mb': memory_info.vms / 1024 / 1024,  # Virtual Memory Size in MB
        'percent': process.memory_percent(),       # Percentage of system memory
        'available_mb': psutil.virtual_memory().available / 1024 / 1024
    }

def cleanup_memory():
    """Force garbage collection and clean up memory"""
    # Force garbage collection
    collected = gc.collect()
    
    # Get memory usage after cleanup
    memory_after = get_memory_usage()
    
    return {
        'objects_collected': collected,
        'memory_after_mb': memory_after['rss_mb']
    }

def monitor_memory_usage(threshold_mb: float = 512.0) -> bool:
    """Check if memory usage exceeds threshold (default lowered to 512MB)"""
    memory_info = get_memory_usage()
    return memory_info['rss_mb'] > threshold_mb

def optimize_data_structures(data: Any, max_size: int = 100) -> Any:
    """Optimize data structures by limiting their size (reduced default)"""
    if isinstance(data, list) and len(data) > max_size:
        # Keep only the most recent entries
        return data[-max_size:]
    elif isinstance(data, dict) and len(data) > max_size:
        # Keep only the most recent entries (assumes ordered dict)
        items = list(data.items())
        return dict(items[-max_size:])
    return data

def limit_modeling_data(data: list, max_rounds: int = 10) -> list:
    """Limit data used for modeling to recent rounds only"""
    if len(data) <= max_rounds:
        return data
    return data[-max_rounds:]

def _safe_get_value(obj, key: str, default=None):
    """Safely get value from dict or object attribute"""
    if hasattr(obj, 'get') and callable(getattr(obj, 'get')):
        # Dictionary-like object
        return obj.get(key, default)
    elif hasattr(obj, key):
        # Object with attributes
        return getattr(obj, key, default)
    else:
        return default

def select_most_useful_data(data: list, max_entries: int = 50, 
                          strategy: str = 'information_diversity') -> list:
    """
    Dynamically select the most useful historical data instead of just the latest.
    
    Args:
        data: List of historical data entries
        max_entries: Maximum number of entries to return
        strategy: Selection strategy ('information_diversity', 'performance_spread', 'temporal_balanced')
    
    Returns:
        Selected subset of most useful data
    """
    if len(data) <= max_entries:
        return data
    
    import random
    
    if strategy == 'information_diversity':
        return _select_by_information_diversity(data, max_entries)
    elif strategy == 'performance_spread':
        return _select_by_performance_spread(data, max_entries)
    elif strategy == 'temporal_balanced':
        return _select_temporal_balanced(data, max_entries)
    else:
        # Fallback to recent data
        return data[-max_entries:]

def _select_by_information_diversity(data: list, max_entries: int) -> list:
    """Select data points that maximize information diversity"""
    if not data:
        return []
    
    # Score each entry based on information content
    scored_data = []
    for i, entry in enumerate(data):
        score = 0.0
        
        # Recent data gets bonus (but not exclusive)
        recency_bonus = (i / len(data)) * 0.3
        
        # Diversity bonus - prefer entries that are different from neighbors
        if i > 0 and i < len(data) - 1:
            prev_entry = data[i-1]
            next_entry = data[i+1]
            
            # Check for performance variance
            curr_mu = _safe_get_value(entry, 'agent_trueskill_mu', 0)
            prev_mu = _safe_get_value(prev_entry, 'agent_trueskill_mu', 0)
            next_mu = _safe_get_value(next_entry, 'agent_trueskill_mu', 0)
            
            if curr_mu is not None and prev_mu is not None and next_mu is not None:
                
                # Higher variance means more informative
                variance = abs(curr_mu - prev_mu) + abs(curr_mu - next_mu)
                score += variance * 0.4
        
        # Information gain bonus
        uncertainty = _safe_get_value(entry, 'uncertainty', 0)
        info_gain = _safe_get_value(entry, 'information_gain', 0)
        if uncertainty is not None and info_gain is not None:
            # High info gain and low uncertainty are valuable
            score += info_gain * 0.2 - uncertainty * 0.1
        
        score += recency_bonus
        scored_data.append((score, i, entry))
    
    # Sort by score and take top entries
    scored_data.sort(key=lambda x: x[0], reverse=True)
    return [entry for _, _, entry in scored_data[:max_entries]]

def _select_by_performance_spread(data: list, max_entries: int) -> list:
    """Select data to maximize performance spread representation"""
    if not data:
        return []
    
    # Extract performance metrics
    performances = []
    for i, entry in enumerate(data):
        mu = _safe_get_value(entry, 'agent_trueskill_mu', 0)
        sigma = _safe_get_value(entry, 'agent_trueskill_sigma', 0)
        performances.append((mu if mu is not None else 0, i, entry))
    
    # Sort by performance
    performances.sort(key=lambda x: x[0])
    
    # Select entries spread across performance range
    selected_indices = set()
    step = max(1, len(performances) // max_entries)
    
    for i in range(0, len(performances), step):
        if len(selected_indices) < max_entries:
            selected_indices.add(performances[i][1])
    
    # Add some recent entries to ensure temporal relevance
    recent_count = max_entries // 4
    for i in range(max(0, len(data) - recent_count), len(data)):
        if len(selected_indices) < max_entries:
            selected_indices.add(i)
    
    return [data[i] for i in sorted(selected_indices)]

def _select_temporal_balanced(data: list, max_entries: int) -> list:
    """Select data with temporal balance - more recent entries but with historical context"""
    if not data:
        return []
    
    selected = []
    
    # Recent 60% of entries
    recent_count = int(max_entries * 0.6)
    selected.extend(data[-recent_count:])
    
    # Historical 40% spread evenly
    historical_count = max_entries - recent_count
    if historical_count > 0 and len(data) > recent_count:
        historical_data = data[:-recent_count]
        step = max(1, len(historical_data) // historical_count)
        
        for i in range(0, len(historical_data), step):
            if len(selected) < max_entries:
                selected.append(historical_data[i])
    
    return selected

def save_data_in_batches(data: list, filename: str, batch_size: int = 100):
    """Save large data in batches to avoid memory issues"""
    if not data:
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    
    # Save data in batches
    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        batch_filename = f"{filename[:-5]}_batch_{i//batch_size + 1}.json"
        
        with open(batch_filename, 'w') as f:
            json.dump(batch, f, indent=2)
        
        # Clear the batch from memory
        del batch

def load_data_in_batches(filename_pattern: str) -> list:
    """Load data that was saved in batches"""
    data = []
    batch_num = 1
    
    while True:
        batch_filename = filename_pattern.replace('.json', f'_batch_{batch_num}.json')
        if not os.path.exists(batch_filename):
            break
        
        with open(batch_filename, 'r') as f:
            batch_data = json.load(f)
            data.extend(batch_data)
        
        batch_num += 1
    
    return data

class MemoryMonitor:
    """Context manager for monitoring memory usage"""
    
    def __init__(self, operation_name: str = "Operation"):
        self.operation_name = operation_name
        self.start_memory = None
        self.end_memory = None
    
    def __enter__(self):
        self.start_memory = get_memory_usage()
        print(f"Starting {self.operation_name} - Memory: {self.start_memory['rss_mb']:.1f} MB")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_memory = get_memory_usage()
        memory_delta = self.end_memory['rss_mb'] - self.start_memory['rss_mb']
        print(f"Finished {self.operation_name} - Memory: {self.end_memory['rss_mb']:.1f} MB (Delta{memory_delta:+.1f} MB)")
        
        # Warn if memory usage is high (lowered threshold)
        if self.end_memory['rss_mb'] > 800:
            print(f"WARNING: High memory usage detected: {self.end_memory['rss_mb']:.1f} MB")