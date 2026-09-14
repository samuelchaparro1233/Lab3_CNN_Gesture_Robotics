"""
Temporal Command Filter with Stability Verification and Safety Logic
Laboratorio 3: CNN para Reconocimiento de Gestos
Universidad Militar Nueva Granada - Inteligencia Artificial
"""

import time
from collections import deque, Counter
from typing import Optional, Tuple, Dict, Any, List


class CommandFilter:
    """
    Temporal filter that converts noisy raw frame-by-frame CNN predictions
    into stable, safe, discrete robotic actuation commands.
    
    Features:
    - Sliding temporal window buffer (default N=10 frames).
    - Statistical mode & confidence threshold gating (default >=0.85 in >=8/10 frames).
    - Refractory cooldown timer between consecutive commands (default 1.5s).
    - Dedicated safety logic for Class 0 (Logical Stop / Inhibit).
    - Full telemetry logging of raw vs filtered commands.
    """
    
    def __init__(
        self,
        confidence_threshold: float = 0.85,
        window_size: int = 10,
        min_stable_count: int = 8,
        cooldown_seconds: float = 1.5,
        idle_class: int = 0
    ):
        self.confidence_threshold = confidence_threshold
        self.window_size = window_size
        self.min_stable_count = min_stable_count
        self.cooldown_seconds = cooldown_seconds
        self.idle_class = idle_class
        
        # Internal state
        self.prediction_buffer = deque(maxlen=window_size)
        self.confidence_buffer = deque(maxlen=window_size)
        self.last_command_time = 0.0
        self.last_accepted_command: Optional[int] = None
        self.is_inhibited: bool = False
        
        # Diagnostic statistics
        self.total_frames_processed = 0
        self.total_commands_accepted = 0
        self.rejected_low_confidence = 0
        self.rejected_instability = 0
        self.rejected_cooldown = 0
        self.class_accepted_counts = {c: 0 for c in range(5)}
        self.history_log: List[Dict[str, Any]] = []

    def reset(self):
        """Clears the temporal buffer and resets cooldown timer."""
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        self.last_command_time = 0.0
        self.last_accepted_command = None
        self.is_inhibited = False

    def update(
        self,
        predicted_class: int,
        confidence: float,
        timestamp: Optional[float] = None
    ) -> Tuple[Optional[int], Dict[str, Any]]:
        """
        Updates the filter with a new raw CNN inference result.
        
        Args:
            predicted_class (int): Raw class predicted by CNN (0 to 4).
            confidence (float): Softmax confidence score for predicted class (0.0 to 1.0).
            timestamp (float, optional): Timestamp in seconds. Defaults to time.time().
            
        Returns:
            Tuple[Optional[int], Dict[str, Any]]:
                - accepted_command: Integer command if accepted this frame, otherwise None.
                - diagnostics: Dictionary containing detailed filter metrics and status.
        """
        if timestamp is None:
            timestamp = time.time()
            
        self.total_frames_processed += 1
        self.prediction_buffer.append(predicted_class)
        self.confidence_buffer.append(confidence)
        
        # Default diagnostic payload
        diag: Dict[str, Any] = {
            "timestamp": timestamp,
            "raw_class": predicted_class,
            "confidence": confidence,
            "accepted_command": None,
            "buffer_length": len(self.prediction_buffer),
            "dominant_class": None,
            "dominant_count": 0,
            "avg_confidence": 0.0,
            "cooldown_remaining": max(0.0, self.cooldown_seconds - (timestamp - self.last_command_time)),
            "filter_status": "BUFFERING",
            "is_inhibited": self.is_inhibited
        }
        
        # Check if buffer has reached minimum required frames
        if len(self.prediction_buffer) < self.window_size:
            diag["filter_status"] = f"BUFFERING ({len(self.prediction_buffer)}/{self.window_size})"
            self.history_log.append(diag)
            return None, diag
            
        # Statistical analysis of current window
        counts = Counter(self.prediction_buffer)
        dominant_class, dominant_count = counts.most_common(1)[0]
        
        # Calculate average confidence for the dominant class instances
        dominant_confs = [
            c for p, c in zip(self.prediction_buffer, self.confidence_buffer)
            if p == dominant_class
        ]
        avg_conf = sum(dominant_confs) / len(dominant_confs) if dominant_confs else 0.0
        
        diag["dominant_class"] = dominant_class
        diag["dominant_count"] = dominant_count
        diag["avg_confidence"] = avg_conf
        
        # Safety Check: Class 0 (Logical Stop / Inhibit)
        if dominant_class == self.idle_class and dominant_count >= self.min_stable_count and avg_conf >= self.confidence_threshold:
            self.is_inhibited = True
            diag["is_inhibited"] = True
            diag["filter_status"] = "IDLE / LOGICAL_STOP_ACTIVE"
            diag["accepted_command"] = self.idle_class
            self.last_accepted_command = self.idle_class
            self.last_command_time = timestamp
            self.class_accepted_counts[self.idle_class] += 1
            self.history_log.append(diag)
            return self.idle_class, diag
            
        # If currently inhibited, clear inhibition once a valid, stable non-idle class is presented
        if self.is_inhibited:
            if dominant_class != self.idle_class and dominant_count >= self.min_stable_count and avg_conf >= self.confidence_threshold:
                self.is_inhibited = False
                diag["is_inhibited"] = False
            else:
                diag["filter_status"] = "INHIBITED_WAITING_FOR_STABLE_GESTURE"
                self.history_log.append(diag)
                return None, diag
                
        # Condition 1: Stability Threshold (Count in window)
        if dominant_count < self.min_stable_count:
            self.rejected_instability += 1
            diag["filter_status"] = f"REJECTED_UNSTABLE ({dominant_count}/{self.min_stable_count})"
            self.history_log.append(diag)
            return None, diag
            
        # Condition 2: Confidence Threshold
        if avg_conf < self.confidence_threshold:
            self.rejected_low_confidence += 1
            diag["filter_status"] = f"REJECTED_LOW_CONFIDENCE ({avg_conf:.2f} < {self.confidence_threshold:.2f})"
            self.history_log.append(diag)
            return None, diag
            
        # Condition 3: Cooldown refractory timer
        time_since_last = timestamp - self.last_command_time
        if time_since_last < self.cooldown_seconds:
            self.rejected_cooldown += 1
            remaining = self.cooldown_seconds - time_since_last
            diag["filter_status"] = f"REJECTED_COOLDOWN ({remaining:.2f}s remaining)"
            self.history_log.append(diag)
            return None, diag
            
        # All conditions satisfied: ACCEPT COMMAND
        self.last_command_time = timestamp
        self.last_accepted_command = dominant_class
        self.total_commands_accepted += 1
        self.class_accepted_counts[dominant_class] += 1
        
        diag["accepted_command"] = dominant_class
        diag["filter_status"] = f"ACCEPTED_COMMAND_{dominant_class}"
        self.history_log.append(diag)
        
        # Clear buffer to require fresh stability for next command
        self.prediction_buffer.clear()
        self.confidence_buffer.clear()
        
        return dominant_class, diag

    def get_summary_statistics(self) -> Dict[str, Any]:
        """Returns cumulative diagnostic report for experimental evaluation."""
        return {
            "total_frames_processed": self.total_frames_processed,
            "total_commands_accepted": self.total_commands_accepted,
            "rejected_low_confidence": self.rejected_low_confidence,
            "rejected_instability": self.rejected_instability,
            "rejected_cooldown": self.rejected_cooldown,
            "class_accepted_counts": self.class_accepted_counts,
            "acceptance_rate_percent": (
                (self.total_commands_accepted / max(1, self.total_frames_processed)) * 100
            )
        }


if __name__ == "__main__":
    # Unit test and demonstration of temporal filter
    cf = CommandFilter(confidence_threshold=0.85, window_size=10, min_stable_count=8, cooldown_seconds=1.5)
    print("Testing CommandFilter with simulated noisy sequence...")
    
    # 1. Unstable sequence
    for i in range(10):
        cls = 1 if i % 2 == 0 else 2
        cmd, diag = cf.update(cls, 0.90, timestamp=i * 0.1)
    print("Unstable result:", diag["filter_status"], "| Accepted:", cmd)
    
    # 2. Stable sequence class 2
    for i in range(10):
        cmd, diag = cf.update(2, 0.92, timestamp=1.0 + i * 0.1)
    print("Stable result:", diag["filter_status"], "| Accepted:", cmd)
    
    # 3. Attempt command during cooldown
    cmd, diag = cf.update(2, 0.95, timestamp=2.2)
    print("Cooldown attempt:", diag["filter_status"], "| Accepted:", cmd)
    
    print("\nSummary:", cf.get_summary_statistics())
