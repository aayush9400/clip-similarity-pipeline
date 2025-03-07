import argparse
import logging
from pipeline.similarity import SimilarityPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='CLIP-based image-text similarity pipeline')
    parser.add_argument('--csv', required=True, help='Path to CSV file with image paths and captions')
    parser.add_argument('--batch-size', type=int, default=64, help='Batch size for processing')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default=None, help='Computation device')
    parser.add_argument('--model', default="ViT-B/32", help='CLIP model variant to use')
    parser.add_argument('--checkpoint-interval', type=int, default=100, help='Save checkpoint every N batches')
    parser.add_argument('--resume', action='store_true', help='Resume from last checkpoint if available')
    parser.add_argument('--half-precision', action='store_true', default=True, help='Use half precision for faster processing')
    return parser.parse_args()


if __name__ == "__main__":
    # Parse command line arguments
    args = parse_args()
    
    # Create and run the pipeline
    pipeline = SimilarityPipeline(
        csv_path=args.csv,
        batch_size=args.batch_size,
        device=args.device,
        model_name=args.model,
    )
    
    # Configure checkpointing
    pipeline.checkpoint_interval = args.checkpoint_interval
    
    # Run pipeline
    output_path = pipeline.run()
    print(f"Pipeline completed successfully. Results saved to: {output_path}")