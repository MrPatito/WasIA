from typing import Dict, Any, Optional, List
import time
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import psutil
import numpy as np
from collections import deque

logger = logging.getLogger(__name__)

@dataclass
class PerformanceMetric:
    name: str
    value: float
    timestamp: datetime
    metadata: Optional[Dict[str, Any]] = None

class PerformanceMonitor:
    def __init__(
        self,
        window_size: int = 1000,
        alert_threshold: float = 0.9
    ):
        """
        Initialize performance monitor.
        
        Args:
            window_size: Number of metrics to keep in memory
            alert_threshold: Threshold for resource usage alerts
        """
        self.metrics: Dict[str, deque] = {}
        self.window_size = window_size
        self.alert_threshold = alert_threshold
        
    def record_metric(
        self,
        name: str,
        value: float,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """Record a performance metric."""
        if name not in self.metrics:
            self.metrics[name] = deque(maxlen=self.window_size)
            
        metric = PerformanceMetric(
            name=name,
            value=value,
            timestamp=datetime.now(),
            metadata=metadata
        )
        
        self.metrics[name].append(metric)
        
        # Check for alerts
        self._check_alerts(metric)
        
    def get_metrics(
        self,
        name: str,
        time_window: Optional[timedelta] = None
    ) -> List[PerformanceMetric]:
        """Get metrics for a specific name."""
        if name not in self.metrics:
            return []
            
        if not time_window:
            return list(self.metrics[name])
            
        cutoff_time = datetime.now() - time_window
        return [
            m for m in self.metrics[name]
            if m.timestamp >= cutoff_time
        ]
        
    def get_stats(
        self,
        name: str,
        time_window: Optional[timedelta] = None
    ) -> Dict[str, float]:
        """Get statistics for a metric."""
        metrics = self.get_metrics(name, time_window)
        if not metrics:
            return {}
            
        values = [m.value for m in metrics]
        return {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'p50': np.percentile(values, 50),
            'p95': np.percentile(values, 95),
            'p99': np.percentile(values, 99)
        }
        
    def monitor_system(self):
        """Record system performance metrics."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=1)
            self.record_metric(
                'cpu_usage',
                cpu_percent,
                {'unit': 'percent'}
            )
            
            # Memory usage
            memory = psutil.virtual_memory()
            self.record_metric(
                'memory_usage',
                memory.percent,
                {
                    'unit': 'percent',
                    'total': memory.total,
                    'available': memory.available
                }
            )
            
            # Disk usage
            disk = psutil.disk_usage('/')
            self.record_metric(
                'disk_usage',
                disk.percent,
                {
                    'unit': 'percent',
                    'total': disk.total,
                    'free': disk.free
                }
            )
            
            # Network I/O
            net_io = psutil.net_io_counters()
            self.record_metric(
                'network_bytes_sent',
                net_io.bytes_sent,
                {'unit': 'bytes'}
            )
            self.record_metric(
                'network_bytes_recv',
                net_io.bytes_recv,
                {'unit': 'bytes'}
            )
            
        except Exception as e:
            logger.error(f"Error monitoring system: {str(e)}")
            
    def _check_alerts(self, metric: PerformanceMetric):
        """Check for performance alerts."""
        try:
            if metric.name == 'cpu_usage' and metric.value > self.alert_threshold * 100:
                logger.warning(
                    f"High CPU usage: {metric.value}% "
                    f"at {metric.timestamp}"
                )
                
            elif metric.name == 'memory_usage' and metric.value > self.alert_threshold * 100:
                logger.warning(
                    f"High memory usage: {metric.value}% "
                    f"at {metric.timestamp}"
                )
                
            elif metric.name == 'disk_usage' and metric.value > self.alert_threshold * 100:
                logger.warning(
                    f"High disk usage: {metric.value}% "
                    f"at {metric.timestamp}"
                )
                
        except Exception as e:
            logger.error(f"Error checking alerts: {str(e)}")
            
    def get_system_health(self) -> Dict[str, Any]:
        """Get overall system health metrics."""
        try:
            return {
                'cpu': self.get_stats('cpu_usage', timedelta(minutes=5)),
                'memory': self.get_stats('memory_usage', timedelta(minutes=5)),
                'disk': self.get_stats('disk_usage', timedelta(minutes=5)),
                'network': {
                    'sent': self.get_stats('network_bytes_sent', timedelta(minutes=5)),
                    'received': self.get_stats('network_bytes_recv', timedelta(minutes=5))
                }
            }
        except Exception as e:
            logger.error(f"Error getting system health: {str(e)}")
            return {}
