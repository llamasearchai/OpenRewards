"""
Comprehensive monitoring and experiment tracking for the reward modeling platform.
Includes metrics collection, experiment tracking, performance monitoring, and alerting.
"""

import time
import psutil
import torch
import numpy as np
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
import threading
from collections import defaultdict, deque
import pickle
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)

@dataclass
class SystemMetrics:
    """System performance metrics."""
    timestamp: float
    cpu_usage: float
    memory_usage: float
    memory_available: float
    gpu_usage: Optional[float] = None
    gpu_memory_usage: Optional[float] = None
    gpu_memory_total: Optional[float] = None
    disk_usage: float = 0.0
    network_io: Optional[Dict[str, int]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class TrainingMetrics:
    """Training-specific metrics."""
    epoch: int
    step: int
    loss: float
    learning_rate: float
    grad_norm: Optional[float] = None
    accuracy: Optional[float] = None
    reward_gap: Optional[float] = None
    throughput: Optional[float] = None  # samples per second
    timestamp: float = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = time.time()
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

@dataclass
class ExperimentConfig:
    """Experiment configuration."""
    experiment_id: str
    model_name: str
    dataset_name: str
    training_args: Dict[str, Any]
    model_config: Dict[str, Any]
    created_at: str
    tags: List[str] = None
    description: str = ""
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []

class MetricsCollector:
    """Collects and manages various metrics."""
    
    def __init__(self, collection_interval: float = 1.0):
        self.collection_interval = collection_interval
        self.system_metrics = deque(maxlen=1000)
        self.training_metrics = deque(maxlen=10000)
        self.custom_metrics = defaultdict(deque)
        self._collecting = False
        self._collection_thread = None
        
        # GPU availability check
        self.gpu_available = torch.cuda.is_available()
        if self.gpu_available:
            try:
                import pynvml
                pynvml.nvmlInit()
                self.nvml_available = True
                self.gpu_count = pynvml.nvmlDeviceGetCount()
            except ImportError:
                self.nvml_available = False
                self.gpu_count = torch.cuda.device_count()
        else:
            self.nvml_available = False
            self.gpu_count = 0
    
    def start_collection(self):
        """Start automatic metrics collection."""
        if not self._collecting:
            self._collecting = True
            self._collection_thread = threading.Thread(target=self._collect_loop, daemon=True)
            self._collection_thread.start()
            logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop automatic metrics collection."""
        self._collecting = False
        if self._collection_thread:
            self._collection_thread.join(timeout=5.0)
        logger.info("Stopped metrics collection")
    
    def _collect_loop(self):
        """Main collection loop."""
        while self._collecting:
            try:
                metrics = self.collect_system_metrics()
                self.system_metrics.append(metrics)
            except Exception as e:
                logger.error(f"Error collecting metrics: {e}")
            
            time.sleep(self.collection_interval)
    
    def collect_system_metrics(self) -> SystemMetrics:
        """Collect current system metrics."""
        # CPU and memory
        cpu_percent = psutil.cpu_percent(interval=None)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        # Network I/O
        try:
            network = psutil.net_io_counters()
            network_io = {
                "bytes_sent": network.bytes_sent,
                "bytes_recv": network.bytes_recv,
                "packets_sent": network.packets_sent,
                "packets_recv": network.packets_recv
            }
        except Exception:
            network_io = None
        
        # GPU metrics
        gpu_usage = None
        gpu_memory_usage = None
        gpu_memory_total = None
        
        if self.gpu_available:
            try:
                if self.nvml_available:
                    gpu_usage, gpu_memory_usage, gpu_memory_total = self._get_gpu_metrics_nvml()
                else:
                    gpu_usage, gpu_memory_usage, gpu_memory_total = self._get_gpu_metrics_torch()
            except Exception as e:
                logger.warning(f"Could not collect GPU metrics: {e}")
        
        return SystemMetrics(
            timestamp=time.time(),
            cpu_usage=cpu_percent,
            memory_usage=memory.percent,
            memory_available=memory.available / (1024**3),  # GB
            gpu_usage=gpu_usage,
            gpu_memory_usage=gpu_memory_usage,
            gpu_memory_total=gpu_memory_total,
            disk_usage=disk.percent,
            network_io=network_io
        )
    
    def _get_gpu_metrics_nvml(self) -> tuple:
        """Get GPU metrics using NVIDIA ML."""
        import pynvml
        
        total_gpu_usage = 0
        total_memory_used = 0
        total_memory_total = 0
        
        for i in range(self.gpu_count):
            handle = pynvml.nvmlDeviceGetHandleByIndex(i)
            
            # GPU utilization
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            total_gpu_usage += util.gpu
            
            # Memory info
            memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
            total_memory_used += memory.used
            total_memory_total += memory.total
        
        avg_gpu_usage = total_gpu_usage / self.gpu_count
        total_memory_used_gb = total_memory_used / (1024**3)
        total_memory_total_gb = total_memory_total / (1024**3)
        
        return avg_gpu_usage, total_memory_used_gb, total_memory_total_gb
    
    def _get_gpu_metrics_torch(self) -> tuple:
        """Get GPU metrics using PyTorch."""
        total_memory_used = 0
        total_memory_total = 0
        
        for i in range(self.gpu_count):
            memory_used = torch.cuda.memory_allocated(i)
            memory_total = torch.cuda.get_device_properties(i).total_memory
            total_memory_used += memory_used
            total_memory_total += memory_total
        
        total_memory_used_gb = total_memory_used / (1024**3)
        total_memory_total_gb = total_memory_total / (1024**3)
        
        # PyTorch doesn't provide GPU utilization, so return None
        return None, total_memory_used_gb, total_memory_total_gb
    
    def log_training_metrics(self, metrics: TrainingMetrics):
        """Log training metrics."""
        self.training_metrics.append(metrics)
    
    def log_custom_metric(self, name: str, value: Union[float, int, str], timestamp: Optional[float] = None):
        """Log custom metric."""
        if timestamp is None:
            timestamp = time.time()
        
        self.custom_metrics[name].append({
            "value": value,
            "timestamp": timestamp
        })
    
    def get_latest_system_metrics(self) -> Optional[SystemMetrics]:
        """Get latest system metrics."""
        return self.system_metrics[-1] if self.system_metrics else None
    
    def get_training_metrics_window(self, window_size: int = 100) -> List[TrainingMetrics]:
        """Get recent training metrics."""
        return list(self.training_metrics)[-window_size:]
    
    def get_metrics_summary(self, time_window: Optional[float] = None) -> Dict[str, Any]:
        """Get summary of all metrics."""
        current_time = time.time()
        cutoff_time = current_time - time_window if time_window else 0
        
        # Filter metrics by time window
        recent_system = [m for m in self.system_metrics if m.timestamp >= cutoff_time]
        recent_training = [m for m in self.training_metrics if m.timestamp >= cutoff_time]
        
        summary = {
            "system": {
                "count": len(recent_system),
                "latest": recent_system[-1].to_dict() if recent_system else None,
                "avg_cpu": np.mean([m.cpu_usage for m in recent_system]) if recent_system else 0,
                "avg_memory": np.mean([m.memory_usage for m in recent_system]) if recent_system else 0,
            },
            "training": {
                "count": len(recent_training),
                "latest": recent_training[-1].to_dict() if recent_training else None,
                "avg_loss": np.mean([m.loss for m in recent_training]) if recent_training else 0,
            },
            "custom": {
                name: {
                    "count": len(values),
                    "latest": values[-1] if values else None
                }
                for name, values in self.custom_metrics.items()
            }
        }
        
        return summary
    
    def export_metrics(self, filepath: str, format: str = "json"):
        """Export metrics to file."""
        data = {
            "system_metrics": [m.to_dict() for m in self.system_metrics],
            "training_metrics": [m.to_dict() for m in self.training_metrics],
            "custom_metrics": dict(self.custom_metrics),
            "exported_at": datetime.now().isoformat()
        }
        
        filepath = Path(filepath)
        
        if format == "json":
            with open(filepath, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        elif format == "pickle":
            with open(filepath, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Metrics exported to {filepath}")

class ExperimentTracker:
    """Tracks experiments and their metadata."""
    
    def __init__(self, experiment_id: str, db_path: Optional[str] = None):
        self.experiment_id = experiment_id
        self.db_path = db_path or "experiments.db"
        self.metrics_collector = MetricsCollector()
        self.experiment_config = None
        self.start_time = time.time()
        
        # Initialize database
        self._init_database()
        
        # Start metrics collection
        self.metrics_collector.start_collection()
    
    def _init_database(self):
        """Initialize SQLite database for experiment tracking."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create experiments table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                id TEXT PRIMARY KEY,
                config TEXT,
                status TEXT,
                created_at TEXT,
                completed_at TEXT,
                metrics TEXT
            )
        """)
        
        # Create metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_metrics (
                experiment_id TEXT,
                metric_name TEXT,
                metric_value REAL,
                timestamp REAL,
                FOREIGN KEY (experiment_id) REFERENCES experiments (id)
            )
        """)
        
        conn.commit()
        conn.close()
    
    def set_config(self, config: ExperimentConfig):
        """Set experiment configuration."""
        self.experiment_config = config
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO experiments (id, config, status, created_at)
            VALUES (?, ?, ?, ?)
        """, (
            self.experiment_id,
            json.dumps(asdict(config)),
            "running",
            datetime.now().isoformat()
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Experiment {self.experiment_id} configuration set")
    
    def log_metric(self, name: str, value: float, step: Optional[int] = None):
        """Log a metric for this experiment."""
        timestamp = time.time()
        
        # Log to metrics collector
        self.metrics_collector.log_custom_metric(f"exp_{name}", value, timestamp)
        
        # Save to database
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO experiment_metrics (experiment_id, metric_name, metric_value, timestamp)
            VALUES (?, ?, ?, ?)
        """, (self.experiment_id, name, value, timestamp))
        
        conn.commit()
        conn.close()
    
    def log_training_metrics(self, metrics: TrainingMetrics):
        """Log training metrics."""
        self.metrics_collector.log_training_metrics(metrics)
        
        # Also log individual metrics to database
        for key, value in metrics.to_dict().items():
            if isinstance(value, (int, float)):
                self.log_metric(f"training_{key}", value)
    
    def complete_experiment(self, final_metrics: Optional[Dict[str, float]] = None):
        """Mark experiment as completed."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        final_metrics_json = json.dumps(final_metrics) if final_metrics else None
        
        cursor.execute("""
            UPDATE experiments 
            SET status = ?, completed_at = ?, metrics = ?
            WHERE id = ?
        """, (
            "completed",
            datetime.now().isoformat(),
            final_metrics_json,
            self.experiment_id
        ))
        
        conn.commit()
        conn.close()
        
        # Stop metrics collection
        self.metrics_collector.stop_collection()
        
        logger.info(f"Experiment {self.experiment_id} completed")
    
    def get_progress(self) -> Dict[str, Any]:
        """Get experiment progress information."""
        elapsed_time = time.time() - self.start_time
        latest_training = self.metrics_collector.get_training_metrics_window(1)
        
        progress = {
            "experiment_id": self.experiment_id,
            "elapsed_time": elapsed_time,
            "status": "running",
            "current_step": latest_training[0].step if latest_training else 0,
            "current_epoch": latest_training[0].epoch if latest_training else 0,
            "latest_loss": latest_training[0].loss if latest_training else None
        }
        
        return progress
    
    def get_latest_metrics(self) -> Dict[str, Any]:
        """Get latest metrics for this experiment."""
        return self.metrics_collector.get_metrics_summary(time_window=3600)  # Last hour
    
    def get_logs(self) -> List[Dict[str, Any]]:
        """Get experiment logs."""
        # This would typically read from a log file or database
        # For now, return training metrics as logs
        training_metrics = self.metrics_collector.get_training_metrics_window(100)
        
        logs = []
        for metric in training_metrics:
            logs.append({
                "timestamp": metric.timestamp,
                "level": "INFO",
                "message": f"Step {metric.step}: Loss {metric.loss:.4f}, LR {metric.learning_rate:.2e}",
                "data": metric.to_dict()
            })
        
        return logs
    
    def export_experiment(self, output_dir: str):
        """Export complete experiment data."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Export configuration
        if self.experiment_config:
            with open(output_path / "config.json", "w") as f:
                json.dump(asdict(self.experiment_config), f, indent=2)
        
        # Export metrics
        self.metrics_collector.export_metrics(output_path / "metrics.json")
        
        # Export logs
        logs = self.get_logs()
        with open(output_path / "logs.json", "w") as f:
            json.dump(logs, f, indent=2, default=str)
        
        logger.info(f"Experiment data exported to {output_path}")

class PerformanceMonitor:
    """Monitors model performance and detects issues."""
    
    def __init__(self, alert_thresholds: Optional[Dict[str, float]] = None):
        self.alert_thresholds = alert_thresholds or {
            "cpu_usage": 90.0,
            "memory_usage": 90.0,
            "gpu_memory_usage": 95.0,
            "loss_spike": 2.0,  # Factor increase
            "training_stall": 300.0  # Seconds without progress
        }
        
        self.alerts = []
        self.last_training_step = 0
        self.last_training_time = time.time()
    
    def check_system_health(self, metrics: SystemMetrics) -> List[str]:
        """Check system health and return alerts."""
        alerts = []
        
        if metrics.cpu_usage > self.alert_thresholds["cpu_usage"]:
            alerts.append(f"High CPU usage: {metrics.cpu_usage:.1f}%")
        
        if metrics.memory_usage > self.alert_thresholds["memory_usage"]:
            alerts.append(f"High memory usage: {metrics.memory_usage:.1f}%")
        
        if (metrics.gpu_memory_usage and 
            metrics.gpu_memory_total and
            (metrics.gpu_memory_usage / metrics.gpu_memory_total * 100) > self.alert_thresholds["gpu_memory_usage"]):
            gpu_percent = metrics.gpu_memory_usage / metrics.gpu_memory_total * 100
            alerts.append(f"High GPU memory usage: {gpu_percent:.1f}%")
        
        return alerts
    
    def check_training_health(self, metrics: TrainingMetrics, previous_metrics: List[TrainingMetrics]) -> List[str]:
        """Check training health and return alerts."""
        alerts = []
        
        # Check for loss spikes
        if len(previous_metrics) > 0:
            recent_losses = [m.loss for m in previous_metrics[-10:]]
            avg_loss = np.mean(recent_losses)
            
            if metrics.loss > avg_loss * self.alert_thresholds["loss_spike"]:
                alerts.append(f"Loss spike detected: {metrics.loss:.4f} vs avg {avg_loss:.4f}")
        
        # Check for training stalls
        current_time = time.time()
        if metrics.step == self.last_training_step:
            if current_time - self.last_training_time > self.alert_thresholds["training_stall"]:
                alerts.append(f"Training appears stalled at step {metrics.step}")
        else:
            self.last_training_step = metrics.step
            self.last_training_time = current_time
        
        return alerts
    
    def log_alert(self, alert: str, severity: str = "warning"):
        """Log an alert."""
        alert_entry = {
            "timestamp": time.time(),
            "severity": severity,
            "message": alert
        }
        
        self.alerts.append(alert_entry)
        logger.warning(f"ALERT [{severity.upper()}]: {alert}")
    
    def get_recent_alerts(self, time_window: float = 3600) -> List[Dict[str, Any]]:
        """Get recent alerts within time window."""
        current_time = time.time()
        cutoff_time = current_time - time_window
        
        return [alert for alert in self.alerts if alert["timestamp"] >= cutoff_time]

class Visualizer:
    """Creates visualizations for monitoring data."""
    
    @staticmethod
    def plot_training_metrics(metrics: List[TrainingMetrics], output_path: str):
        """Plot training metrics over time."""
        if not metrics:
            return
        
        steps = [m.step for m in metrics]
        losses = [m.loss for m in metrics]
        learning_rates = [m.learning_rate for m in metrics]
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
        
        # Loss plot
        ax1.plot(steps, losses, 'b-', alpha=0.7)
        ax1.set_xlabel('Step')
        ax1.set_ylabel('Loss')
        ax1.set_title('Training Loss')
        ax1.grid(True)
        
        # Learning rate plot
        ax2.plot(steps, learning_rates, 'r-', alpha=0.7)
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Learning Rate')
        ax2.set_title('Learning Rate Schedule')
        ax2.set_yscale('log')
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    @staticmethod
    def plot_system_metrics(metrics: List[SystemMetrics], output_path: str):
        """Plot system metrics over time."""
        if not metrics:
            return
        
        timestamps = [datetime.fromtimestamp(m.timestamp) for m in metrics]
        cpu_usage = [m.cpu_usage for m in metrics]
        memory_usage = [m.memory_usage for m in metrics]
        gpu_usage = [m.gpu_usage for m in metrics if m.gpu_usage is not None]
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # CPU usage
        axes[0, 0].plot(timestamps, cpu_usage, 'b-', alpha=0.7)
        axes[0, 0].set_title('CPU Usage (%)')
        axes[0, 0].set_ylabel('Usage %')
        axes[0, 0].grid(True)
        
        # Memory usage
        axes[0, 1].plot(timestamps, memory_usage, 'g-', alpha=0.7)
        axes[0, 1].set_title('Memory Usage (%)')
        axes[0, 1].set_ylabel('Usage %')
        axes[0, 1].grid(True)
        
        # GPU usage (if available)
        if gpu_usage:
            gpu_timestamps = [datetime.fromtimestamp(m.timestamp) for m in metrics if m.gpu_usage is not None]
            axes[1, 0].plot(gpu_timestamps, gpu_usage, 'r-', alpha=0.7)
            axes[1, 0].set_title('GPU Usage (%)')
            axes[1, 0].set_ylabel('Usage %')
            axes[1, 0].grid(True)
        
        # GPU memory usage
        gpu_memory = [m.gpu_memory_usage for m in metrics if m.gpu_memory_usage is not None]
        if gpu_memory:
            gpu_mem_timestamps = [datetime.fromtimestamp(m.timestamp) for m in metrics if m.gpu_memory_usage is not None]
            axes[1, 1].plot(gpu_mem_timestamps, gpu_memory, 'm-', alpha=0.7)
            axes[1, 1].set_title('GPU Memory Usage (GB)')
            axes[1, 1].set_ylabel('Memory (GB)')
            axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()

# Utility functions
def create_experiment_tracker(experiment_id: str, config: Dict[str, Any]) -> ExperimentTracker:
    """Create and configure an experiment tracker."""
    tracker = ExperimentTracker(experiment_id)
    
    exp_config = ExperimentConfig(
        experiment_id=experiment_id,
        model_name=config.get("model_name", "unknown"),
        dataset_name=config.get("dataset_name", "unknown"),
        training_args=config.get("training_args", {}),
        model_config=config.get("model_config", {}),
        created_at=datetime.now().isoformat(),
        tags=config.get("tags", []),
        description=config.get("description", "")
    )
    
    tracker.set_config(exp_config)
    return tracker

def load_experiment_data(experiment_id: str, db_path: str = "experiments.db") -> Optional[Dict[str, Any]]:
    """Load experiment data from database."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute("SELECT * FROM experiments WHERE id = ?", (experiment_id,))
    result = cursor.fetchone()
    
    if result:
        columns = [desc[0] for desc in cursor.description]
        experiment_data = dict(zip(columns, result))
        
        # Load metrics
        cursor.execute(
            "SELECT metric_name, metric_value, timestamp FROM experiment_metrics WHERE experiment_id = ?",
            (experiment_id,)
        )
        metrics = cursor.fetchall()
        experiment_data["metrics"] = [
            {"name": name, "value": value, "timestamp": timestamp}
            for name, value, timestamp in metrics
        ]
        
        conn.close()
        return experiment_data
    
    conn.close()
    return None 