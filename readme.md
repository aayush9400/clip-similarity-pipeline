# CLIP-Based Image-Text Similarity Pipeline

## Project Overview

This project provides a scalable pipeline for computing similarity metrics between images and their corresponding text captions using OpenAI's CLIP (Contrastive Language-Image Pre-training) model. The solution is designed to efficiently process large datasets while tracking performance metrics.

## Features

- Load image-caption pairs from CSV files
- Compute similarity scores between images and captions using CLIP
- Process data in memory-efficient batches for large datasets
- Support for GPU acceleration (CUDA)
- Checkpoint functionality for resuming interrupted processing
- Comprehensive performance metrics tracking
- Containerized deployment with Docker

## Installation

### Prerequisites

- Python 3.8+
- PyTorch 2.0+
- CUDA-capable GPU (recommended for large datasets)

### Option 1: Local Installation

1. Clone the repository:
   ```
   git clone https://github.com/jaayush12/clip-similarity-pipeline.git
   cd clip-similarity-pipeline
   ```

2. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

### Option 2: Docker Container (Recommended for Production)

1. Build the Docker image:
   ```
   docker build -t clip-similarity-pipeline .
   ```

2. Run the container:
   ```
   docker run -it --gpus all \
     -v /path/to/your/data:/data \
     clip-similarity-pipeline \
     --csv /data/your_input.csv \
     --batch-size 32 \
     --device cuda
   ```

## Usage

### Command Line Interface
```bash
python main.py --csv /path/to/your/data.csv --batch-size 64 --device cuda
```


### Command Line Arguments

- `--csv`: Path to CSV file with image paths (or valid urls) and captions (required)
- `--batch-size`: Number of images to process in each batch 
- `--device`: Computation device (`cpu`, `cuda`, or `mps`)
- `--model`: CLIP model variant to use (default: "ViT-B/32")
- `--checkpoint-interval`: Save checkpoint every N batches
- `--resume`: Resume from last checkpoint if available
- `--half-precision`: Use half precision for faster processing

### Input Format

The input CSV should contain at least two columns:
- `url`: Path or URL to the image
- `caption`: The text caption for the corresponding image

### Output

The pipeline produces:
1. A CSV file with original columns plus a new `similarity` column
2. A metrics file tracking processing time, throughput, and memory usage

## Design Considerations

### Architecture

The pipeline is designed with a modular architecture:

- **Pipeline Layer**: `SimilarityPipeline` class orchestrates the overall process
- **Model Layer**: `ClipModel` provides high-level interfaces for CLIP operations
- **Utility Layer**: `DataLoader` and `MetricsTracker` handle I/O and metrics

This separation of concerns allows for easy maintenance and extension.

### Scalability

Several design choices enable scaling to large datasets:

1. **Streaming Processing**: Data is loaded and processed in chunks to minimize memory usage
2. **Checkpointing**: Saves progress regularly to resume interrupted processing
3. **Adaptive Batch Size**: Automatically reduces batch size on OOM errors
4. **Half-Precision**: Option to use FP16 for faster processing with minimal accuracy loss
5. **Custom Aspect Ratio Handling**: Preserves image content through proper padding

### Performance Optimizations

- Efficient tensor operations using PyTorch
- GPU memory management with explicit garbage collection
- Batched processing for both images and text
- Normalized feature vectors for accurate similarity computation

## Metrics Collection and Analysis

The pipeline records several performance metrics:

- Processing time per batch and total runtime
- Memory usage
- Images processed per second (throughput)
- Success/failure counts

These metrics help identify bottlenecks and optimize the pipeline for production use.

## Future Improvements

- Multi-GPU support for parallel processing
- Integration with distributed computing frameworks (e.g., Dask, Ray)
- Additional similarity models beyond CLIP
- Improve image preprocessing to downsample more efficiently and accurately