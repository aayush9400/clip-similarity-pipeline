import os
import time
import re
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import gc
import logging
from pipeline.models import ClipModel
from pipeline.utils import Pipeline

logger = logging.getLogger(__name__)


class SimilarityPipeline(Pipeline):
    """Pipeline for computing similarity between images and their text captions"""
    
    def __init__(self, csv_path, batch_size=64, device=None, model_name="ViT-B/32"):
        """
        Args:
            csv_path (str): Path to the CSV file containing image paths and captions
            batch_size (int): Number of images to process in each batch
            device (str): Device to use for computation ('cuda', 'cpu', 'mps')
            model_name (str): Name of the model to use
            image_dir (str): Directory containing local images
        """
        super().__init__(csv_path)
        self.batch_size = batch_size
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.image_dir = os.path.dirname(csv_path)
        
        logger.info(f"Initializing pipeline with device={self.device}, batch_size={self.batch_size}")
        
        hf_model_name = self._convert_to_hf_model_name(model_name)
        logger.info(f"Using model: {model_name} (Hugging Face: {hf_model_name})")
        
        # Initialize model
        self.model = ClipModel(model_name=hf_model_name, device=self.device)
    
    def _convert_to_hf_model_name(self, model_name):
        """Map model name to Hugging Face model name"""
        model_map = {
            "ViT-B/32": "openai/clip-vit-base-patch32",
            "ViT-B/16": "openai/clip-vit-base-patch16",
            "ViT-L/14": "openai/clip-vit-large-patch14",
        }
        return model_map.get(model_name, model_name)
    
    def _extract_local_image_path(self, url):
        """Extract local image path from URL"""
        # Extract the identifier between /users/ and /generations/
        match = re.search(r'/users/([^/]+)/generations/', url)
        if match and self.image_dir:
            image_id = match.group(1)
            return os.path.join(self.image_dir, f"{image_id}.png")
        return url
    
    def load_data(self):
        """Load and prepare data from CSV file"""
        logger.info(f"Loading data from {self.csv_path}")
        df = pd.read_csv(self.csv_path)
        
        # Verify the CSV has the expected columns
        if 'url' not in df.columns or 'caption' not in df.columns:
            raise ValueError("CSV must contain 'url' and 'caption' columns")
        
        # Map URLs to local file paths if image_dir is provided
        if self.image_dir:
            logger.info(f"Using local images from {self.image_dir}")
            df['image_path'] = df['url'].apply(self._extract_local_image_path)
        else:
            df['image_path'] = df['url']
            
        return df
    
    def process_batch(self, batch_df):
        """
        Args:
            batch_df (pd.DataFrame): Dataframe containing batch of image paths and captions
        
        Returns:
            list: List of similarity scores
            float: Batch processing time
        """
        batch_start_time = time.time()
        results = []
        
        processed_count = 0
        skipped_count = 0
        
        for _, row in batch_df.iterrows():
            image_path = row['image_path']
            caption = row['caption']
            
            # Preprocess image
            image_tensor = self.model.preprocess_image(image_path)
            if image_tensor is None:
                results.append(None)
                skipped_count += 1
                continue
            
            # Compute similarity
            similarity = self.model.compute_similarity(image_tensor, caption)
            results.append(similarity)
            processed_count += 1
        
        batch_time = time.time() - batch_start_time
        
        # Update metrics
        self.metrics.update_counts(processed=processed_count, skipped=skipped_count)
        self.metrics.record_batch_time(batch_time)
        
        return results
    
    def process_in_batches(self):
        """Process the CSV file in batches using streaming"""
        # Get total count without loading entire dataset
        with open(self.csv_path, 'r') as f:
            total_count = sum(1 for _ in f) - 1  # Subtract header row
        
        self.metrics.update_counts(total=total_count)
        logger.info(f"Processing approximately {total_count} images in batches of {self.batch_size}")
        
        # Process data in streaming chunks
        output_path = self._get_output_filepath(suffix="_with_similarities")
        header_written = False
        
        for chunk_df in self.data_loader.iter_batches(batch_size=self.batch_size, 
                                                    required_columns=['url', 'caption']):
            # Map URLs to local file paths if needed
            if 'image_path' not in chunk_df.columns:
                chunk_df = self.data_loader.prepare_image_paths(chunk_df)
            
            # Process batch
            similarities = self.process_batch(chunk_df)
            chunk_df['similarity'] = similarities
            
            # Append results to output file
            self.data_loader.append_data(
                chunk_df[['url', 'caption', 'similarity']], 
                output_path, 
                header_first=not header_written
            )
            header_written = True
            
            # Track memory usage and cleanup
            self.metrics.update_memory_usage()
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        logger.info(f"Average batch processing time: {self.metrics.metrics['avg_processing_time_per_batch']:.2f} seconds")
        return output_path
    
    def run(self):
        """
        Run the complete pipeline.
        
        Returns:
            str: Path to the output CSV file
        """
        self.metrics.start()
        
        # Process all images in streaming fashion
        output_path = self.process_in_batches()
        
        self.metrics.finish()
        self._save_metrics(prefix="similarity_metrics")
        
        logger.info(f"Pipeline completed. Results saved to {output_path}")
        return output_path
