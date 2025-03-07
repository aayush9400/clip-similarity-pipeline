import torch
from PIL import Image
import logging
import os
from transformers import AutoProcessor, CLIPModel

# Configure logging
logger = logging.getLogger(__name__)


class ClipModel:
    """
    CLIP model wrapper that handles image processing, text processing, 
    and similarity computation.
    """
    
    def __init__(self, model_name="openai/clip-vit-base-patch32", device=None, batch_size=32, use_half_precision=True):
        """
        Initialize the CLIP model.
        
        Args:
            model_name (str): Name of the CLIP model variant to use
            device (str): Device to use for computation ('cuda' or 'cpu')
            batch_size (int): Batch size for efficient processing
            use_half_precision (bool): Whether to use FP16 for faster processing
        """
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = CLIPModel.from_pretrained(model_name)
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model.to(self.device)
        self.batch_size = batch_size
        self.target_size = 224
            
        # Use half-precision for efficiency with large datasets
        if use_half_precision and self.device == 'cuda':
            self.model = self.model.half()
            
        logger.info(f"Loaded CLIP model {model_name} on {self.device} with input size {self.target_size}px")
        logger.info(f"Batch size: {batch_size}, Half precision: {use_half_precision}")
    
    def preprocess_with_padding(self, image):
        """
        Preprocess image with padding to preserve entire content.
        
        Args:
            image (PIL.Image): PIL image
            
        Returns:
            torch.Tensor: Preprocessed image tensor
        """
        # Calculate aspect ratio-preserving resize
        width, height = image.size
        aspect_ratio = width / height
        
        if width > height:
            new_width = self.target_size
            new_height = int(self.target_size / aspect_ratio)
        else:
            new_height = self.target_size
            new_width = int(self.target_size * aspect_ratio)
        
        # Resize the image while maintaining aspect ratio
        resized_img = image.resize((new_width, new_height), Image.LANCZOS)
        
        # Create a new square black image
        padded_img = Image.new("RGB", (self.target_size, self.target_size), color=(0, 0, 0))
        
        # Paste the resized image centered in the padded image
        paste_x = (self.target_size - new_width) // 2
        paste_y = (self.target_size - new_height) // 2
        padded_img.paste(resized_img, (paste_x, paste_y))
        
        # Use the CLIP processor to normalize properly
        return self.processor(images=padded_img, return_tensors="pt")["pixel_values"][0]
    
    def preprocess_image(self, image_path):
        """
        Load and preprocess an image.
        
        Args:
            image_path (str): Path to the image
        
        Returns:
            Preprocessed image tensor or None if loading fails
        """
        try:
            if not os.path.exists(image_path):
                logger.warning(f"Image not found: {image_path}")
                return None
                
            image = Image.open(image_path).convert("RGB")
            return self.preprocess_with_padding(image)
        except Exception as e:
            logger.warning(f"Error processing image {image_path}: {str(e)}")
            return None
    
    def preprocess_image_batch(self, image_paths):
        """
        Process a batch of images efficiently.
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            torch.Tensor: Batch of preprocessed images, None entries for failed images
        """
        processed_images = []
        valid_indices = []
        
        for i, path in enumerate(image_paths):
            try:
                if not os.path.exists(path):
                    logger.warning(f"Image not found: {path}")
                    processed_images.append(None)
                    continue
                    
                image = Image.open(path).convert("RGB")
                processed = self.preprocess_with_padding(image)
                processed_images.append(processed)
                valid_indices.append(i)
            except Exception as e:
                logger.warning(f"Error processing image {path}: {str(e)}")
                processed_images.append(None)
        
        # Stack only valid tensors
        if valid_indices:
            valid_tensors = [processed_images[i] for i in valid_indices]
            if valid_tensors:
                return torch.stack(valid_tensors).to(self.device), valid_indices
        
        return None, []
    
    def encode_image_batch(self, image_paths):
        """
        Efficiently encode a batch of images to feature vectors.
        
        Args:
            image_paths (list): List of image paths
            
        Returns:
            tuple: (features, valid_indices) where features is a tensor of normalized features
                  and valid_indices maps the output rows to input indices
        """
        total = len(image_paths)
        all_features = []
        all_valid_indices = []
        
        # Process in batches for memory efficiency
        for i in range(0, total, self.batch_size):
            batch_paths = image_paths[i:i+self.batch_size]
            batch_tensors, valid_indices = self.preprocess_image_batch(batch_paths)
            
            if batch_tensors is not None and len(valid_indices) > 0:
                try:
                    with torch.no_grad():
                        # Convert to half precision if model is in half precision
                        if next(self.model.parameters()).dtype == torch.float16:
                            batch_tensors = batch_tensors.half()
                            
                        # Get image features from the vision model
                        features = self.model.get_image_features(pixel_values=batch_tensors)
                        normalized_features = features / features.norm(dim=-1, keepdim=True)
                        
                        # Map back to original indices
                        all_features.append(normalized_features.cpu())
                        all_valid_indices.extend([i + idx for idx in valid_indices])
                        
                    # Explicitly clear tensors to help garbage collection
                    del batch_tensors
                    del features
                    del normalized_features
                    torch.cuda.empty_cache() if torch.cuda.is_available() else None
                except RuntimeError as e:
                    if 'out of memory' in str(e) and self.batch_size > 1:
                        logger.warning("GPU OOM error, reducing batch size and retrying")
                        # Try with a smaller batch size
                        self.batch_size = max(1, self.batch_size // 2)
                        # Clear cache and retry this batch
                        torch.cuda.empty_cache() if torch.cuda.is_available() else None
                        return self.encode_image_batch(image_paths)
                    else:
                        logger.error(f"Error processing batch: {str(e)}")
                        raise
                    
            # Report progress
            if (i + self.batch_size) % 1000 == 0 or (i + self.batch_size) >= total:
                logger.info(f"Processed {min(i + self.batch_size, total)}/{total} images")
        
        if all_features:
            return torch.cat(all_features), all_valid_indices
        return None, []
    
    def encode_image(self, image_tensor):
        """
        Encode a single image to a feature vector.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
        
        Returns:
            torch.Tensor: Normalized image features
        """
        with torch.no_grad():
            # Ensure the tensor has batch dimension
            if image_tensor.dim() == 3:
                image_tensor = image_tensor.unsqueeze(0)
            
            # Move to device and convert to half precision if needed
            image_tensor = image_tensor.to(self.device)
            if next(self.model.parameters()).dtype == torch.float16:
                image_tensor = image_tensor.half()
                
            image_features = self.model.get_image_features(pixel_values=image_tensor)
            return image_features / image_features.norm(dim=-1, keepdim=True)
    
    def preprocess_text(self, text):
        """
        Preprocess a text string.
        
        Args:
            text (str): Text to preprocess
        
        Returns:
            Tokenized text
        """
        inputs = self.processor(text=[text], truncation=True, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items() if k != 'pixel_values'}
    
    def preprocess_text_batch(self, texts):
        """
        Process a batch of texts efficiently.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            dict: Batch of tokenized texts
        """
        inputs = self.processor(text=texts, truncation=True, padding=True, return_tensors="pt")
        return {k: v.to(self.device) for k, v in inputs.items() if k != 'pixel_values'}
    
    def encode_text(self, text):
        """
        Encode text to a feature vector.
        
        Args:
            text (str): Text to encode
        
        Returns:
            torch.Tensor: Normalized text features
        """
        with torch.no_grad():
            text_inputs = self.preprocess_text(text)
            
            text_features = self.model.get_text_features(**text_inputs)
            return text_features / text_features.norm(dim=-1, keepdim=True)
    
    def encode_text_batch(self, texts):
        """
        Efficiently encode a batch of texts to feature vectors.
        
        Args:
            texts (list): List of text strings
            
        Returns:
            torch.Tensor: Tensor of normalized features
        """
        total = len(texts)
        all_features = []
        
        # Process in batches for memory efficiency
        for i in range(0, total, self.batch_size):
            batch_texts = texts[i:i+self.batch_size]
            batch_inputs = self.preprocess_text_batch(batch_texts)
            
            with torch.no_grad():
                features = self.model.get_text_features(**batch_inputs)
                normalized_features = features / features.norm(dim=-1, keepdim=True)
                all_features.append(normalized_features.cpu())
                
            # Report progress for large batches
            if (i + self.batch_size) % 10000 == 0 or (i + self.batch_size) >= total:
                logger.info(f"Processed {min(i + self.batch_size, total)}/{total} texts")
        
        if all_features:
            return torch.cat(all_features)
        return None
    
    def compute_similarity(self, image_tensor, text):
        """
        Compute similarity between image and text.
        
        Args:
            image_tensor (torch.Tensor): Preprocessed image tensor
            text (str): Text caption
        
        Returns:
            float: Similarity score
        """
        with torch.no_grad():
            image_features = self.encode_image(image_tensor)
            text_features = self.encode_text(text)
            
            # Compute similarity
            similarity = (image_features @ text_features.T).item()
            
            return similarity
    
    def batch_compute_similarity(self, image_tensors, texts):
        """
        Compute similarities for a batch of image-text pairs efficiently.
        
        Args:
            image_tensors (list or tensor): List of preprocessed image tensors or batched tensor
            texts (list): List of text captions
        
        Returns:
            list: List of similarity scores
        """
        with torch.no_grad():
            # Handle case where image_tensors is already a batch tensor
            if isinstance(image_tensors, torch.Tensor) and image_tensors.dim() == 4:
                image_features = self.encode_image(image_tensors)
            else:
                # Stack individual tensors into a batch
                stacked_tensors = torch.stack(image_tensors).to(self.device)
                image_features = self.encode_image(stacked_tensors)
            
            # Encode all texts
            text_features = self.encode_text_batch(texts)
            
            # Compute all similarities at once
            similarities = (image_features @ text_features.T).cpu().numpy()
            
            # Extract diagonal for paired similarities
            return [similarities[i, i] for i in range(min(similarities.shape))]
