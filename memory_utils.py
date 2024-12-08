# memory_utils.py
import os
import psutil
from datetime import datetime
from threading import local
from io import StringIO

# Thread-local storage for run-specific data
_thread_local = local()

def init_memory_log():
    """Initialize a new memory log buffer for this run"""
    _thread_local.memory_log = StringIO()    

def get_memory_log() -> str:
    """Get the current memory log contents"""
    if hasattr(_thread_local, 'memory_log'):
        return _thread_local.memory_log.getvalue()
    return ""

def get_run_timestamp():
    """Generate or retrieve timestamp for the current run."""
    if not hasattr(_thread_local, 'timestamp'):
        _thread_local.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return _thread_local.timestamp

def reset_run_timestamp():
    """Reset the timestamp for a new run."""
    if hasattr(_thread_local, 'timestamp'):
        del _thread_local.timestamp

def monitor_memory(stage_name, state=None):
    """
    Capture detailed memory statistics at each stage of the graph workflow
    """
    if not hasattr(_thread_local, 'memory_log'):
        init_memory_log()

    process = psutil.Process(os.getpid())

    # Collect memory stats
    memory_stats = {
        'timestamp': datetime.now().isoformat(),
        'stage': stage_name,
        'memory_usage_mb': process.memory_info().rss / 1024 / 1024,
        'memory_percent': process.memory_percent(),
        'system_memory': {
            'total': psutil.virtual_memory().total / 1024 / 1024,
            'available': psutil.virtual_memory().available / 1024 / 1024,
            'used_percent': psutil.virtual_memory().percent
        }
    }

    # If state is provided, only log safe properties
    if state:
        try:
            memory_stats['state_info'] = {
                'num_docs': len(state.get('docs', [])) if 'docs' in state else 0,
                'num_relevant_docs': len(state.get('relevant_docs', [])) if 'relevant_docs' in state else 0,
                'query_length': len(state.get('query', '')) if 'query' in state else 0,
                'summary_length': len(state.get('generated_summary', '')) if 'generated_summary' in state else 0,
            }
        except Exception as e:
            print(f"Warning: Could not log some state information: {str(e)}")

        # Write to memory buffer
    _thread_local.memory_log.write(f"\n{'='*50}\n")
    _thread_local.memory_log.write(f"Memory Stats for {stage_name}\n")
    _thread_local.memory_log.write(f"{'='*50}\n")
    for key, value in memory_stats.items():
        if isinstance(value, dict):
            _thread_local.memory_log.write(f"{key}:\n")
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, float):
                    _thread_local.memory_log.write(f"  {sub_key}: {sub_value:.2f} MB\n")
                else:
                    _thread_local.memory_log.write(f"  {sub_key}: {sub_value}\n")
        elif isinstance(value, float):
            _thread_local.memory_log.write(f"{key}: {value:.2f} MB\n")
        else:
            _thread_local.memory_log.write(f"{key}: {value}\n")

    ## for local testing
    # # Ensure log directory exists
    # log_dir = "memory_logs"
    # if not os.path.exists(log_dir):
    #     os.makedirs(log_dir)

    # # Get or create log file for this run
    # log_file = os.path.join(log_dir, f"memory_log_{get_run_timestamp()}.txt")
    
    # # Append to the run's log file
    # with open(log_file, 'a') as f:
    #     f.write(f"\n{'='*50}\n")
    #     f.write(f"Memory Stats for {stage_name}\n")
    #     f.write(f"{'='*50}\n")
    #     for key, value in memory_stats.items():
    #         if isinstance(value, dict):
    #             f.write(f"{key}:\n")
    #             for sub_key, sub_value in value.items():
    #                 if isinstance(sub_value, float):
    #                     f.write(f"  {sub_key}: {sub_value:.2f} MB\n")
    #                 else:
    #                     f.write(f"  {sub_key}: {sub_value}\n")
    #         elif isinstance(value, float):
    #             f.write(f"{key}: {value:.2f} MB\n")
    #         else:
    #             f.write(f"{key}: {value}\n")