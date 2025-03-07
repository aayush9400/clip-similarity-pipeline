import os
import numpy as np
import pandas as pd
import psutil
from datetime import datetime
import logging
from pathlib import Path
import re
import json


# Configure logging
logger = logging.getLogger(__name__)

class DataLoader:
    """Class for loading and managing data."""
    
    def __init__(self, csv_path, image_dir=None):
        self.csv_path = csv_path
        self.image_dir = image_dir or os.path.dirname(csv_path)
    
    def load_data(self, required_columns=None):
        """Load entire data from CSV with validation"""
        df = pd.read_csv(self.csv_path)
        
        # Validate required columns if specified
        if required_columns:
            missing = [col for col in required_columns if col not in df.columns]
            if missing:
                raise ValueError(f"CSV missing required columns: {', '.join(missing)}")
        
        return df
    
    def iter_batches(self, batch_size=10000, required_columns=None):
        """Iterate through CSV file in batches for memory efficiency."""
        # validate required columns exist 
        if required_columns:
            df_sample = pd.read_csv(self.csv_path, nrows=1)
            missing = [col for col in required_columns if col not in df_sample.columns]
            if missing:
                raise ValueError(f"CSV missing required columns: {', '.join(missing)}")
        
        # Iterate through the CSV in chunks
        for chunk in pd.read_csv(self.csv_path, chunksize=batch_size):
            yield chunk
    
    def get_local_image_path(self, url):
        """Convert URL to local image path"""
        # Extract the identifier between /users/ and /generations/
        match = re.search(r'/users/([^/]+)/generations/', url)
        if match and self.image_dir:
            image_id = match.group(1)
            return os.path.join(self.image_dir, f"{image_id}.png")
        return url
    
    def prepare_image_paths(self, df, url_column='url', path_column='image_path'):
        """Prepare image paths from URLs"""
        df[path_column] = df[url_column].apply(self.get_local_image_path)
        return df
    
    def save_data(self, df, output_path):
        df.to_csv(output_path, index=False)
        logger.info(f"Data saved to {output_path}")
        
    def append_data(self, df, output_path, header_first=False):
        """Append dataframe to existing CSV or create a new one"""
        write_header = header_first or not os.path.exists(output_path)
        df.to_csv(output_path, mode='a', header=write_header, index=False)


class MetricsTracker:
    """Class for tracking and recording performance metrics."""
    
    def __init__(self):
        self.metrics = {
            'start_time': None,
            'end_time': None,
            'total_images': 0,
            'processed_images': 0,
            'skipped_images': 0,
            'peak_memory': 0,
            'batch_times': [],
            'avg_processing_time_per_batch': 0,
        }
    
    def start(self):
        self.metrics['start_time'] = datetime.now()
    
    def finish(self):
        self.metrics['end_time'] = datetime.now()
        if self.metrics['batch_times']:
            self.metrics['avg_processing_time_per_batch'] = np.mean(self.metrics['batch_times'])
    
    def update_counts(self, total=None, processed=None, skipped=None):
        if total is not None:
            self.metrics['total_images'] = total
        if processed is not None:
            self.metrics['processed_images'] += processed
        if skipped is not None:
            self.metrics['skipped_images'] += skipped
    
    def record_batch_time(self, batch_time):
        self.metrics['batch_times'].append(batch_time)
    
    def update_memory_usage(self):
        current_memory = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)  # MB
        self.metrics['peak_memory'] = max(self.metrics['peak_memory'], current_memory)
    
    def save_metrics(self, filepath):
        # Calculate total runtime
        total_runtime = (self.metrics['end_time'] - self.metrics['start_time']).total_seconds()
        
        with open(filepath, 'w') as f:
            f.write("=== Image-Text Similarity Pipeline Metrics ===\n\n")
            f.write(f"Start time: {self.metrics['start_time']}\n")
            f.write(f"End time: {self.metrics['end_time']}\n")
            f.write(f"Total runtime: {total_runtime:.2f} seconds\n\n")
            
            f.write(f"Total images in dataset: {self.metrics['total_images']}\n")
            f.write(f"Successfully processed images: {self.metrics['processed_images']}\n")
            f.write(f"Skipped images: {self.metrics['skipped_images']}\n\n")
            
            f.write(f"Average processing time per batch: {self.metrics['avg_processing_time_per_batch']:.2f} seconds\n")
            f.write(f"Peak memory usage: {self.metrics['peak_memory']:.2f} MB\n")
            
            # Calculate throughput
            if total_runtime > 0:
                throughput = self.metrics['processed_images'] / total_runtime
                f.write(f"Throughput: {throughput:.2f} images/second\n")
        
        logger.info(f"Performance metrics saved to {filepath}")


class Pipeline:
    """Base class for data processing pipelines."""
    
    def __init__(self, csv_path):
        self.csv_path = csv_path
        self.data_loader = DataLoader(csv_path)
        self.metrics = MetricsTracker()
        self.checkpoint_interval = 100  # Save checkpoint every 100 batches
    
    def _get_output_filepath(self, suffix="_processed"):
        input_path = Path(self.csv_path)
        return str(input_path.parent / f"{input_path.stem}{suffix}{input_path.suffix}")
    
    def _get_checkpoint_filepath(self):
        """Get path for checkpointing progress"""
        input_path = Path(self.csv_path)
        return str(input_path.parent / f"{input_path.stem}_checkpoint.json")
    
    def save_checkpoint(self, current_position, processed_count):
        """Save processing checkpoint"""
        checkpoint_data = {
            "position": current_position,
            "processed_count": processed_count,
            "timestamp": datetime.now().isoformat()
        }
        with open(self._get_checkpoint_filepath(), 'w') as f:
            json.dump(checkpoint_data, f)
        logger.info(f"Checkpoint saved at position {current_position}")
    
    def load_checkpoint(self):
        """Load processing checkpoint if exists"""
        checkpoint_path = self._get_checkpoint_filepath()
        if os.path.exists(checkpoint_path):
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                logger.info(f"Resuming from checkpoint at position {checkpoint_data['position']}")
                return checkpoint_data
            except Exception as e:
                logger.warning(f"Failed to load checkpoint: {e}")
        return None
    
    def _save_metrics(self, prefix="metrics"):
        metrics_path = Path(self.csv_path).parent / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        self.metrics.save_metrics(metrics_path)
