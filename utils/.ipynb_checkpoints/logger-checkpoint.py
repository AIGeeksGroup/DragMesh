"""
Logging utilities for TensorBoard, Weights & Biases, and file logging.
"""
import os
import logging
from datetime import datetime
from typing import Dict, Optional, Any


class Logger:
    """Unified logger for TensorBoard, W&B, and file logging."""

    def __init__(self, log_dir: str, use_tensorboard: bool = True, use_wandb: bool = False,
                 wandb_project: Optional[str] = None, wandb_config: Optional[Dict] = None,
                 use_file_logging: bool = True):
        """
        Initialize logger.

        Args:
            log_dir: Directory for logs
            use_tensorboard: Whether to use TensorBoard
            use_wandb: Whether to use Weights & Biases
            wandb_project: W&B project name
            wandb_config: W&B configuration dict
            use_file_logging: Whether to log to file
        """
        self.use_tensorboard = use_tensorboard
        self.use_wandb = use_wandb
        self.use_file_logging = use_file_logging
        self.log_dir = log_dir

        # Initialize file logger
        if self.use_file_logging:
            self._init_file_logger()

        # Initialize TensorBoard
        if self.use_tensorboard:
            try:
                from torch.utils.tensorboard import SummaryWriter
                self.tb_writer = SummaryWriter(log_dir=log_dir)
                self.log_info(f"✓ TensorBoard logging to: {log_dir}")
            except ImportError:
                self.log_warning("⚠ TensorBoard not available. Install with: pip install tensorboard")
                self.use_tensorboard = False
                self.tb_writer = None
        else:
            self.tb_writer = None

        # Initialize Weights & Biases
        if self.use_wandb:
            try:
                import wandb
                wandb.init(project=wandb_project, config=wandb_config, dir=log_dir)
                self.log_info(f"✓ Weights & Biases logging to project: {wandb_project}")
            except ImportError:
                self.log_warning("⚠ W&B not available. Install with: pip install wandb")
                self.use_wandb = False
            except Exception as e:
                self.log_warning(f"⚠ W&B initialization failed: {e}")
                self.use_wandb = False

    def _init_file_logger(self):
        """Initialize file logger."""
        os.makedirs(self.log_dir, exist_ok=True)

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = os.path.join(self.log_dir, f'training_{timestamp}.log')

        # Configure logger
        self.file_logger = logging.getLogger('training_logger')
        self.file_logger.setLevel(logging.INFO)

        # Remove existing handlers to avoid duplicates
        self.file_logger.handlers = []

        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)

        # Console handler (optional, for debugging)
        # console_handler = logging.StreamHandler()
        # console_handler.setLevel(logging.INFO)

        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)

        self.file_logger.addHandler(file_handler)
        # self.file_logger.addHandler(console_handler)

        print(f"✓ File logging to: {log_file}")

    def log_info(self, message: str):
        """Log info message to file."""
        if self.use_file_logging:
            self.file_logger.info(message)

    def log_warning(self, message: str):
        """Log warning message to file."""
        if self.use_file_logging:
            self.file_logger.warning(message)
        else:
            print(message)

    def log_error(self, message: str):
        """Log error message to file."""
        if self.use_file_logging:
            self.file_logger.error(message)
        else:
            print(message)

    def log_scalar(self, tag: str, value: float, step: int):
        """Log a scalar value."""
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_scalar(tag, value, step)

        if self.use_wandb:
            import wandb
            wandb.log({tag: value}, step=step)

    def log_scalars(self, main_tag: str, tag_scalar_dict: Dict[str, float], step: int):
        """Log multiple scalar values."""
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.add_scalars(main_tag, tag_scalar_dict, step)

        if self.use_wandb:
            import wandb
            log_dict = {f"{main_tag}/{k}": v for k, v in tag_scalar_dict.items()}
            wandb.log(log_dict, step=step)

    def log_metrics(self, metrics: Dict[str, Any], step: int, prefix: str = ""):
        """
        Log multiple metrics at once.

        Args:
            metrics: Dictionary of metric names and values
            step: Global step number
            prefix: Prefix for metric names (e.g., 'train/', 'val/')
        """
        for key, value in metrics.items():
            tag = f"{prefix}{key}" if prefix else key
            self.log_scalar(tag, value, step)

    def log_epoch_summary(self, epoch: int, train_metrics: Dict[str, float],
                         val_metrics: Dict[str, float], extra_info: Optional[Dict[str, Any]] = None):
        """
        Log epoch summary to file.

        Args:
            epoch: Epoch number
            train_metrics: Training metrics
            val_metrics: Validation metrics
            extra_info: Additional info to log (e.g., lr, kl_weight)
        """
        if not self.use_file_logging:
            return

        summary_lines = [
            "",
            "="*80,
            f"Epoch {epoch} Summary",
            "="*80
        ]

        # Training metrics
        summary_lines.append("\nTraining Metrics:")
        for key, value in train_metrics.items():
            if isinstance(value, float):
                summary_lines.append(f"  {key}: {value:.6f}")
            else:
                summary_lines.append(f"  {key}: {value}")

        # Validation metrics
        summary_lines.append("\nValidation Metrics:")
        for key, value in val_metrics.items():
            if isinstance(value, float):
                summary_lines.append(f"  {key}: {value:.6f}")
            else:
                summary_lines.append(f"  {key}: {value}")

        # Extra info
        if extra_info:
            summary_lines.append("\nAdditional Info:")
            for key, value in extra_info.items():
                if isinstance(value, float):
                    summary_lines.append(f"  {key}: {value:.6f}")
                else:
                    summary_lines.append(f"  {key}: {value}")

        summary_lines.append("="*80)
        summary_lines.append("")

        # Log all lines
        for line in summary_lines:
            self.file_logger.info(line)

    def close(self):
        """Close all loggers."""
        if self.use_tensorboard and self.tb_writer is not None:
            self.tb_writer.close()

        if self.use_wandb:
            import wandb
            wandb.finish()

        if self.use_file_logging:
            self.log_info("Training session ended.")
            # Close file handlers
            for handler in self.file_logger.handlers[:]:
                handler.close()
                self.file_logger.removeHandler(handler)


def create_logger(output_dir: str, use_tensorboard: bool = True, use_wandb: bool = False,
                  wandb_project: Optional[str] = None, wandb_config: Optional[Dict] = None,
                  use_file_logging: bool = True) -> Logger:
    """
    Create a logger instance.

    Args:
        output_dir: Output directory for logs
        use_tensorboard: Whether to use TensorBoard
        use_wandb: Whether to use W&B
        wandb_project: W&B project name
        wandb_config: W&B configuration dict
        use_file_logging: Whether to log to file

    Returns:
        Logger instance
    """
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    return Logger(
        log_dir=log_dir,
        use_tensorboard=use_tensorboard,
        use_wandb=use_wandb,
        wandb_project=wandb_project,
        wandb_config=wandb_config,
        use_file_logging=use_file_logging
    )
