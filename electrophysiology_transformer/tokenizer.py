"""
Simple tokenizer for electrophysiology data.
Handles clipping and scaling of voltage and current data.
"""
import numpy as np
import torch


class ElectrophysiologyTokenizer:
    """
    Tokenizer for patch-clamp electrophysiology data.
    
    This tokenizer performs:
    - Clipping to reasonable physiological ranges
    - Optional scaling/normalization
    """
    
    def __init__(
        self,
        voltage_clip=(-120.0, 40.0),  # mV - typical physiological range
        current_clip=(-4000.0, 4000.0),  # pA - typical current range
        normalize=True,
    ):
        """
        Args:
            voltage_clip: tuple (min, max) for voltage clipping in mV
            current_clip: tuple (min, max) for current clipping in pA
            normalize: whether to normalize to [-1, 1] range
        """
        self.voltage_clip = voltage_clip
        self.current_clip = current_clip
        self.normalize = normalize
        
    def clip_voltage(self, voltage):
        """Clip voltage to physiological range."""
        return np.clip(voltage, self.voltage_clip[0], self.voltage_clip[1])
    
    def clip_current(self, current):
        """Clip current to reasonable range."""
        return np.clip(current, self.current_clip[0], self.current_clip[1])
    
    def normalize_voltage(self, voltage):
        """Normalize voltage to [-1, 1] range."""
        v_min, v_max = self.voltage_clip
        return 2 * (voltage - v_min) / (v_max - v_min) - 1
    
    def denormalize_voltage(self, voltage_normalized):
        """Denormalize voltage from [-1, 1] range back to mV."""
        v_min, v_max = self.voltage_clip
        return (voltage_normalized + 1) * (v_max - v_min) / 2 + v_min
    
    def normalize_current(self, current):
        """Normalize current to [-1, 1] range."""
        c_min, c_max = self.current_clip
        return 2 * (current - c_min) / (c_max - c_min) - 1
    
    def denormalize_current(self, current_normalized):
        """Denormalize current from [-1, 1] range back to pA."""
        c_min, c_max = self.current_clip
        return (current_normalized + 1) * (c_max - c_min) / 2 + c_min
    
    def __call__(self, voltages, currents):
        """
        Process voltage and current data.
        
        Args:
            voltages: numpy array or list of voltage values
            currents: numpy array or list of current values
            
        Returns:
            tuple of processed (voltages, currents)
        """
        # Convert to numpy if needed
        if isinstance(voltages, list):
            voltages = np.array(voltages, dtype=np.float32)
        if isinstance(currents, list):
            currents = np.array(currents, dtype=np.float32)
            
        # Clip
        voltages = self.clip_voltage(voltages)
        currents = self.clip_current(currents)
        
        # Normalize if requested
        if self.normalize:
            voltages = self.normalize_voltage(voltages)
            currents = self.normalize_current(currents)
            
        return voltages, currents
    
    def decode_voltage(self, voltage_normalized):
        """Decode voltage from normalized values back to mV."""
        if self.normalize:
            return self.denormalize_voltage(voltage_normalized)
        else:
            return voltage_normalized
            
    def decode_current(self, current_normalized):
        """Decode current from normalized values back to pA."""
        if self.normalize:
            return self.denormalize_current(current_normalized)
        else:
            return current_normalized
