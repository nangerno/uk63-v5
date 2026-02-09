"""
Two-Phase Training Strategy
Phase 1: Fast convergence (60% time) - Aggressive settings, early checkpoint
Phase 2: Quality refinement (40% time) - Conservative settings, fine-tune from Phase 1
"""
import os
import json
import re
from typing import Dict, Optional
from datetime import datetime, timedelta, timezone


class TwoPhaseTrainingStrategy:
    """Implements two-phase training to beat competitors by finishing early with better quality"""
    
    def __init__(self, hours_to_complete: float, task_type: str):
        self.hours_to_complete = hours_to_complete
        self.task_type = task_type
        self.phase1_time_ratio = 0.55  # 55% for Phase 1 (slightly less than 60% to be safe)
        self.phase2_time_ratio = 0.45  # 45% for Phase 2
        
    def get_phase1_config(self, base_config: Dict, model_name: str) -> Dict:
        """Phase 1: Fast convergence with aggressive settings
        Note: This method is kept for compatibility but config is now built in text_trainer.py
        """
        config = base_config.copy()
        
        # Aggressive learning rate (1.5x base LR for faster convergence)
        if "learning_rate" in config:
            config["learning_rate"] = config["learning_rate"] * 1.5
        
        # Larger batch size if possible (faster training)
        if "batch_size" in config:
            config["batch_size"] = int(config["batch_size"] * 1.2)
        
        # Fewer warmup steps (start training faster)
        if "warmup_steps" in config:
            config["warmup_steps"] = max(10, int(config["warmup_steps"] * 0.6))
        else:
            config["warmup_steps"] = 15
        
        # Mark as Phase 1
        config["training_phase"] = "phase1"
        config["skip_evaluation"] = True
        
        return config
    
    def get_phase2_config(self, base_config: Dict, phase1_checkpoint: str, model_name: str) -> Dict:
        """Phase 2: Quality refinement with conservative settings
        Note: This method is kept for compatibility but config is now built in text_trainer.py
        """
        config = base_config.copy()
        
        # Start from Phase 1 checkpoint
        config["resume_from_checkpoint"] = phase1_checkpoint
        
        # Conservative learning rate (0.5x base LR for fine-tuning)
        if "learning_rate" in config:
            config["learning_rate"] = config["learning_rate"] * 0.5
        
        # Smaller batch size for better generalization
        if "batch_size" in config:
            config["batch_size"] = max(2, int(config["batch_size"] * 0.8))
        
        # More warmup steps for stable training
        if "warmup_steps" in config:
            config["warmup_steps"] = int(config["warmup_steps"] * 1.5)
        else:
            config["warmup_steps"] = 35
        
        # Mark as Phase 2
        config["training_phase"] = "phase2"
        config["skip_evaluation"] = False
        
        return config
    
    def get_early_checking_step(self, total_steps: int, phase: str = "phase1") -> int:
        """Get early checking step - much earlier than competitor's 100 steps"""
        if phase == "phase1":
            # Check at 30-50 steps (vs competitor's 100)
            # Use 5% of total steps or 40 steps, whichever is smaller
            # But ensure at least 30 steps for meaningful training
            checking_step = min(50, max(30, int(total_steps * 0.05)))
        else:  # phase2
            # Check earlier in phase 2 too, but not as early
            # Use 8% of total steps or 50 steps, whichever is smaller
            checking_step = min(60, max(40, int(total_steps * 0.08)))
        
        return checking_step
    
    def calculate_time_allocation(self) -> Dict[str, float]:
        """Calculate time allocation for each phase"""
        total_seconds = self.hours_to_complete * 3600
        
        # Reserve 5 minutes for final evaluation and upload
        reserve_time = 5 * 60
        available_time = total_seconds - reserve_time
        
        phase1_time = available_time * self.phase1_time_ratio
        phase2_time = available_time * self.phase2_time_ratio
        
        return {
            "phase1_seconds": phase1_time,
            "phase2_seconds": phase2_time,
            "reserve_seconds": reserve_time,
            "total_seconds": total_seconds
        }
    
    def get_phase_end_time(self, start_time: datetime, phase: str) -> str:
        """Calculate end time for specific phase"""
        time_allocation = self.calculate_time_allocation()
        
        if phase == "phase1":
            phase_duration = timedelta(seconds=time_allocation["phase1_seconds"])
        else:  # phase2
            phase_duration = timedelta(seconds=time_allocation["phase2_seconds"])
        
        end_time = start_time + phase_duration
        return end_time.strftime("%Y-%m-%d %H:%M:%S")
    
    def modify_train_cmd_for_phase(self, base_cmd: str, phase_config: Dict, 
                                  output_dir: str, request_path: str) -> str:
        """Modify training command for specific phase"""
        cmd = base_cmd
        
        # Replace learning rate
        if "learning_rate" in phase_config:
            cmd = re.sub(r'--learning_rate\s+\S+', 
                        f'--learning_rate {phase_config["learning_rate"]}', cmd)
            if '--learning_rate' not in cmd:
                cmd += f' --learning_rate {phase_config["learning_rate"]}'
        
        # Replace batch size
        if "batch_size" in phase_config:
            cmd = re.sub(r'--per_device_train_batch_size\s+\S+', 
                        f'--per_device_train_batch_size {phase_config["batch_size"]}', cmd)
            if '--per_device_train_batch_size' not in cmd:
                cmd += f' --per_device_train_batch_size {phase_config["batch_size"]}'
        
        # Replace warmup steps
        if "warmup_steps" in phase_config:
            cmd = re.sub(r'--warmup_steps\s+\S+', 
                        f'--warmup_steps {phase_config["warmup_steps"]}', cmd)
            if '--warmup_steps' not in cmd:
                cmd += f' --warmup_steps {phase_config["warmup_steps"]}'
        
        # Replace gradient accumulation
        if "gradient_accumulation_steps" in phase_config:
            cmd = re.sub(r'--gradient_accumulation_steps\s+\S+', 
                        f'--gradient_accumulation_steps {phase_config["gradient_accumulation_steps"]}', cmd)
            if '--gradient_accumulation_steps' not in cmd:
                cmd += f' --gradient_accumulation_steps {phase_config["gradient_accumulation_steps"]}'
        
        # Replace output dir
        cmd = re.sub(r'--output_dir\s+\S+', f'--output_dir {output_dir}', cmd)
        
        # Replace request path
        cmd = re.sub(r'--request_path\s+\S+', f'--request_path {request_path}', cmd)
        
        # Add resume from checkpoint if Phase 2
        if "resume_from_checkpoint" in phase_config and phase_config["resume_from_checkpoint"]:
            cmd = re.sub(r'--resume_from_checkpoint\s+\S+', 
                        f'--resume_from_checkpoint {phase_config["resume_from_checkpoint"]}', cmd)
            if '--resume_from_checkpoint' not in cmd:
                cmd += f' --resume_from_checkpoint {phase_config["resume_from_checkpoint"]}'
        
        # Add LoRA configs if present
        if "lora_r" in phase_config:
            cmd = re.sub(r'--lora_r\s+\S+', f'--lora_r {phase_config["lora_r"]}', cmd)
            if '--lora_r' not in cmd:
                cmd += f' --lora_r {phase_config["lora_r"]}'
        
        if "lora_alpha" in phase_config:
            cmd = re.sub(r'--lora_alpha\s+\S+', f'--lora_alpha {phase_config["lora_alpha"]}', cmd)
            if '--lora_alpha' not in cmd:
                cmd += f' --lora_alpha {phase_config["lora_alpha"]}'
        
        return cmd


def extract_value_from_cmd(cmd: str, arg_name: str) -> Optional[str]:
    """Extract value from command string"""
    match = re.search(f"(?P<p>--{arg_name}(\\s+)(?P<value>[^\\s]+))(\\s+)", cmd)
    if match:
        return match.group("value")
    return None


def replace_args_in_cmd(cmd: str, arg_name: str, arg_value: str) -> str:
    """Replace argument in command string"""
    match = re.search(f"(?P<p>--{arg_name}(\\s+)([^\\s]+))(\\s+)", cmd)
    if match:
        left_index = match.start("p")
        right_index = match.end("p")
        return cmd[:left_index] + f" --{arg_name} {arg_value} " + cmd[right_index:]
    else:
        # Add if not present
        return cmd + f" --{arg_name} {arg_value}"
