from transformers import GenerationConfig
import datetime
from datetime import timezone
from transformers import (
    TrainerCallback,
    TrainerState,
    TrainerControl,
)
import os
from typing import Callable, Optional, Dict
import shutil
import json
from transformers.trainer_utils import is_main_process
import wandb
import torch
import torch.nn as nn
from state_manager import get_state, set_state
import numpy as np
MAX_TRIES = 9


MIS_MATCH_VOCAB_SIZE_MODELS = [
    'NousResearch/Nous-Capybara-7B-V1',
    'berkeley-nest/Starling-LM-7B-alpha',
    'NousResearch/Hermes-2-Theta-Llama-3-8B',
    'MNC-Jihun/Mistral-7B-AO-u0.5-b2-ver0.4'
]

ERROR_GENERATION_CONFIG_MODELS = [
    "lmsys/vicuna-7b-v1.5", 
    "lmsys/vicuna-13b-v1.5",
    "NousResearch/Nous-Hermes-llama-2-7b", 
    "defog/llama-3-sqlcoder-8b"
]

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))

print(f"LOCAL_RANK: {LOCAL_RANK} in customized_trainer.py", flush=True)


class EMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.9999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.backup:
                param.data = self.backup[name]
        self.backup = {}


class SWA:
    """Stochastic Weight Averaging"""
    def __init__(self, model, start_step=0, swa_freq=1):
        self.model = model
        self.start_step = start_step
        self.swa_freq = swa_freq
        self.swa_model = {}
        self.swa_n = 0
        self.enabled = False

    def update(self, step):
        if step >= self.start_step and step % self.swa_freq == 0:
            self.enabled = True
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    if name not in self.swa_model:
                        self.swa_model[name] = param.data.clone()
                    else:
                        self.swa_model[name] = (self.swa_n * self.swa_model[name] + param.data) / (self.swa_n + 1)
            self.swa_n += 1

    def apply_swa(self):
        if not self.enabled or self.swa_n == 0:
            return False
        for name, param in self.model.named_parameters():
            if param.requires_grad and name in self.swa_model:
                param.data = self.swa_model[name].clone()
        return True


class AdvancedCheckpointSelector:
    """Multi-criteria checkpoint selection"""
    def __init__(self):
        self.checkpoints = []
        
    def add_checkpoint(self, step, eval_loss, train_loss, gradient_norm=None):
        score = self._calculate_score(eval_loss, train_loss, gradient_norm)
        self.checkpoints.append({
            'step': step,
            'eval_loss': eval_loss,
            'train_loss': train_loss,
            'gradient_norm': gradient_norm,
            'score': score
        })
        
    def _calculate_score(self, eval_loss, train_loss, gradient_norm):
        """Calculate composite score - lower is better"""
        if eval_loss is None or train_loss is None:
            return float('inf')
        
        # Base score: eval loss (most important)
        score = eval_loss
        
        # Penalty for overfitting (large train/eval gap)
        if train_loss > 0:
            overfit_ratio = eval_loss / train_loss if train_loss > 0 else 1.0
            if overfit_ratio > 1.5:  # Significant overfitting
                score *= 1.1
        
        # Bonus for stable gradients
        if gradient_norm is not None:
            if 0.1 < gradient_norm < 10.0:  # Reasonable gradient norm
                score *= 0.98
        
        return score
    
    def get_best_checkpoint(self):
        if not self.checkpoints:
            return None
        best = min(self.checkpoints, key=lambda x: x['score'])
        return best
    
    def get_all_checkpoints(self):
        return sorted(self.checkpoints, key=lambda x: x['score'])


class CustomEvalSaveCallback(TrainerCallback):
    def __init__(
        self,
        function_when_to_evaluate: Callable,
        submission_dir: str,
        output_dir: str,
        original_model_name: str,
        max_steps: int = -1,
        checking_step: int = 100,
        total_steps_all_epochs: int = -1,
        end_time: str = "",
        checking_mode: str = "none",
        use_ema: bool = True,
        use_swa: bool = True,
        ema_decay: float = 0.9999,
        swa_start_ratio: float = 0.75,
    ):
        self.function_when_to_evaluate = function_when_to_evaluate
        self.submission_dir = submission_dir
        self.current_best_loss = None
        self.best_checkpoint_info = None
        self.update_best_checkpoint = False
        self.output_dir = output_dir
        self.original_model_name = original_model_name
        self.max_steps = max_steps
        self.has_checkpoint = False
        self.save_only = False
        self.checking_step = checking_step
        self.total_steps_all_epochs = total_steps_all_epochs
        self.checking_mode = checking_mode
        self.end_time = end_time
        self.use_ema = use_ema
        self.use_swa = use_swa
        self.ema = None
        self.swa = None
        self.checkpoint_selector = AdvancedCheckpointSelector()
        self.ema_decay = ema_decay
        self.swa_start_ratio = swa_start_ratio
        self.loss_history = []
        self.gradient_norms = []
        
    def compute_loss(self, state: TrainerState, metrics):
        return metrics.get("eval_loss", None)

    def on_train_begin(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Initialize EMA and SWA"""
        if model is not None and is_main_process(LOCAL_RANK):
            if self.use_ema:
                self.ema = EMA(model, decay=self.ema_decay)
                print(f"EMA initialized with decay={self.ema_decay}", flush=True)
            if self.use_swa:
                swa_start_step = int(self.total_steps_all_epochs * self.swa_start_ratio) if self.total_steps_all_epochs > 0 else 0
                self.swa = SWA(model, start_step=swa_start_step, swa_freq=1)
                print(f"SWA initialized starting at step {swa_start_step}", flush=True)

    def on_step_end(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        """Update EMA and SWA, track metrics"""
        if model is not None and is_main_process(LOCAL_RANK):
            if self.ema is not None:
                self.ema.update()
            if self.swa is not None:
                self.swa.update(state.global_step)
        
        # Track loss history for better selection
        if state.log_history:
            last_log = state.log_history[-1]
            if "loss" in last_log:
                self.loss_history.append(last_log["loss"])
                # Keep only last 100 losses for trend analysis
                if len(self.loss_history) > 100:
                    self.loss_history.pop(0)
        
        # Track gradient norms if available
        if model is not None:
            try:
                total_norm = 0.0
                for p in model.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** (1. / 2)
                self.gradient_norms.append(total_norm)
                if len(self.gradient_norms) > 100:
                    self.gradient_norms.pop(0)
            except:
                pass
        
        if state.global_step == self.checking_step and self.checking_mode == "first_time":
            my_state = get_state()
            if "train" not in my_state:
                print(f"Warning: 'train' key not found in state at step {state.global_step}", flush=True)
                return control
            if "start_time" not in my_state["train"]:
                print(f"Warning: 'start_time' key not found in state['train'] at step {state.global_step}", flush=True)
                return control
            if "start_train_time" not in my_state["train"]:
                print(f"Warning: 'start_train_time' key not found in state['train'] at step {state.global_step}", flush=True)
                return control
            start_time_obj = datetime.datetime.strptime(my_state["train"]["start_time"], "%Y-%m-%d %H:%M:%S")
            start_train_time_obj = datetime.datetime.strptime(my_state["train"]["start_train_time"], "%Y-%m-%d %H:%M:%S")
            
            log_content = f"Checking the model at step: {state.global_step}"
            now = datetime.datetime.now()
            preparation_time = (start_train_time_obj - start_time_obj).total_seconds()
            log_content += f"\nPreparation time: {preparation_time}"
            time_so_far = (now - start_time_obj).total_seconds()
            log_content += f"\nTime so far: {time_so_far}"
            time_for_one_step = (now - start_train_time_obj).total_seconds() / self.checking_step
            log_content += f"\nTime for one step: {time_for_one_step}"
            log_content += f"\nTotal steps all epochs: {self.total_steps_all_epochs}"
            total_remaining_training_time = time_for_one_step * (self.total_steps_all_epochs - state.global_step)
            log_content += f"\nTotal remaining training time: {total_remaining_training_time}"
            end_time_obj = datetime.datetime.strptime(self.end_time, "%Y-%m-%d %H:%M:%S")
            total_remaining_time = (end_time_obj - now).total_seconds()
            log_content += f"\nTotal remaining time: {total_remaining_time}"
            
            max_var_time_sofar = 3 * 60
            n = (total_remaining_time - (time_so_far + total_remaining_training_time + 12 * 60)) / (time_so_far + max_var_time_sofar)
            n = int(n)
            my_state["check_details"] = {
                "now": str(now.strftime("%Y-%m-%d %H:%M:%S")),
                "start_time": str(start_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "start_train_time": str(start_train_time_obj.strftime("%Y-%m-%d %H:%M:%S")),
                "checking_step": self.checking_step,
                "checking_mode": self.checking_mode,
                "estimation_of_steps": n,
                "preparation_time": preparation_time,
                "time_so_far": time_so_far,
                "time_for_one_step": time_for_one_step,
                "total_remaining_training_time": total_remaining_training_time,
                "total_remaining_time": total_remaining_time,
                "end_time": self.end_time,
            }
            if n > 0:
                log_content += f"\nEstimated number of steps to complete the training: {n}"
                control.should_training_stop = True
                control.should_save = False
                args.save_strategy = "no"
                last_log = state.log_history[-1]
                my_state["train"]["current_loss"] = last_log["loss"]
                my_state["mode"] = "continue"
                if n > MAX_TRIES:
                    n = MAX_TRIES
                log_content += f"\nFinal number: {n + 1}"
                my_state["next_runs"] = n + 1
            else:
                print(f"Time is not enough so we will finish the training", flush=True)
                my_state["mode"] = "finish"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
                print(log_content, flush=True)            
            return control
    
        elif state.global_step == self.checking_step and self.checking_mode == "second_time":
            log_content = f"Checking the model at step: {state.global_step} where check_mode=second_time"            
            my_state = get_state()
            if "train" not in my_state:
                print(f"Warning: 'train' key not found in state at step {state.global_step}", flush=True)
                return control
            current_loss = state.log_history[-1]["loss"]
            my_state["train"]["current_loss"] = current_loss
                
            control.should_training_stop = True

            current_is_the_best = False
            if "runs" not in my_state or len(my_state["runs"]) == 0:
                print(f"Warning: 'runs' key not found or empty in state at step {state.global_step}", flush=True)
                return control
            current_min_loss = min([run["current_loss"] for run in my_state["runs"]])
            if current_loss <= current_min_loss:
                if len(my_state["runs"]) + 1 == my_state["next_runs"]:
                    print(f"Current loss: {my_state['train']['current_loss']} is less than or equal to: {current_min_loss}", flush=True)
                    current_is_the_best = True
                    
            if current_is_the_best:
                control.should_training_stop = False
                my_state["mode"] = "finish"
            else:
                control.should_save = False
                args.save_strategy = "no"
            
            if is_main_process(LOCAL_RANK):
                set_state(my_state)
        
        when_to_eval = self.function_when_to_evaluate(state.global_step)
        if when_to_eval["eval"]:
            print(f"Evaluating the model at step: {state.global_step} the reason: {when_to_eval['reason']}", flush=True)
            control.should_evaluate = True
            control.should_save = True
            if when_to_eval["reason"] == "end_time":
                if not self.has_checkpoint:
                    print(f"No checkpoint found, just save the model at step: {state.global_step}", flush=True)
                    control.should_evaluate = False
                    self.save_only = True
        return control

    def on_evaluate(
        self, args, state: TrainerState, control: TrainerControl, metrics, **kwargs
    ):
        self.save_only = False
        eval_loss = self.compute_loss(state, metrics)
        if state.global_step < 2:
            return 
        print(f"GO INTO CUSTOMIZED EVALUATE AT STEP: {state.global_step}", flush=True)
        
        # Get training loss
        train_loss = None
        if state.log_history:
            last_log = state.log_history[-1]
            train_loss = last_log.get("loss", None)
        
        # Get gradient norm
        gradient_norm = None
        if self.gradient_norms:
            gradient_norm = np.mean(self.gradient_norms[-10:])  # Average of last 10
        
        # Add to checkpoint selector
        self.checkpoint_selector.add_checkpoint(
            state.global_step, 
            eval_loss, 
            train_loss, 
            gradient_norm
        )
        
        # Get best checkpoint from selector
        best_checkpoint = self.checkpoint_selector.get_best_checkpoint()
        
        if best_checkpoint and (self.best_checkpoint_info is None or 
                               best_checkpoint['step'] == state.global_step):
            print(f"Updating the best checkpoint info at step: {state.global_step} with eval_loss: {eval_loss}, score: {best_checkpoint['score']:.6f}", flush=True)
            self.best_checkpoint_info = {
                "loss": eval_loss,
                "step": state.global_step,
                "score": best_checkpoint['score'],
                "train_loss": train_loss,
                "gradient_norm": gradient_norm
            }
            self.update_best_checkpoint = True
        else:
            if self.best_checkpoint_info is not None:
                print(f"At step: {state.global_step} The eval_loss: {eval_loss} score: {best_checkpoint['score'] if best_checkpoint else 'N/A'} is not better than current best: {self.best_checkpoint_info.get('score', self.best_checkpoint_info['loss'])}, update_best_checkpoint={self.update_best_checkpoint}", flush=True)

    def on_save(self, args, state: TrainerState, control: TrainerControl, model=None, **kwargs):
        if state.global_step == self.max_steps and self.max_steps != -1:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            control.should_training_stop = True
        
        self.has_checkpoint = True
        
        if not is_main_process(LOCAL_RANK):
            return 
            
        if self.save_only:
            print(f"Only save the model at step: {state.global_step}, no evaluation", flush=True)
            current_step = state.global_step
            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)
                
            shutil.copytree(
                os.path.join(self.output_dir, f"checkpoint-{current_step}"),
                self.submission_dir
            )
            self.update_best_checkpoint = False
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{current_step},no_eval")
            self.save_only = False
            return 
            
        if (
            self.update_best_checkpoint
            and is_main_process(LOCAL_RANK)
        ):
            print(f"Copy the best checkpoint to the submission directory at step: {state.global_step}", flush=True)
            if os.path.exists(self.submission_dir):
                shutil.rmtree(self.submission_dir)
            best_eval_loss = self.best_checkpoint_info["loss"]
            best_step = self.best_checkpoint_info["step"]
            
            # Use best checkpoint (EMA/SWA will be applied at final save if needed)
            best_model_path = os.path.join(self.output_dir, f"checkpoint-{best_step}")
            model_type = "checkpoint"
            
            # If we have EMA/SWA and this is near the end, try to use them
            if model is not None and state.global_step >= self.total_steps_all_epochs * 0.9:
                # Try SWA first (more stable for final model)
                if self.swa is not None and self.swa.enabled:
                    try:
                        # Save current weights
                        current_weights = {}
                        for name, param in model.named_parameters():
                            if param.requires_grad:
                                current_weights[name] = param.data.clone()
                        
                        # Apply SWA
                        if self.swa.apply_swa():
                            swa_path = os.path.join(self.output_dir, f"checkpoint-{best_step}-swa")
                            if not os.path.exists(swa_path):
                                os.makedirs(swa_path, exist_ok=True)
                            
                            # Copy base checkpoint first
                            if os.path.exists(best_model_path):
                                for item in os.listdir(best_model_path):
                                    s = os.path.join(best_model_path, item)
                                    d = os.path.join(swa_path, item)
                                    if os.path.isdir(s):
                                        shutil.copytree(s, d, dirs_exist_ok=True)
                                    else:
                                        shutil.copy2(s, d)
                            
                            # Save SWA weights
                            import torch
                            swa_state_dict = {}
                            for name, param in model.named_parameters():
                                if param.requires_grad and name in self.swa.swa_model:
                                    swa_state_dict[name] = self.swa.swa_model[name].clone()
                            
                            if swa_state_dict:
                                torch.save(swa_state_dict, os.path.join(swa_path, "swa_weights.pt"))
                                best_model_path = swa_path
                                model_type = "swa"
                                print(f"Using SWA model with {self.swa.swa_n} averaged checkpoints", flush=True)
                            
                            # Restore original weights
                            for name, param in model.named_parameters():
                                if param.requires_grad and name in current_weights:
                                    param.data = current_weights[name]
                        else:
                            # Restore original weights
                            for name, param in model.named_parameters():
                                if param.requires_grad and name in current_weights:
                                    param.data = current_weights[name]
                    except Exception as e:
                        print(f"Failed to use SWA: {e}", flush=True)
                        import traceback
                        traceback.print_exc()
            
            if not os.path.exists(best_model_path):
                best_model_path = os.path.join(self.output_dir, f"checkpoint-{best_step}")
            
            shutil.copytree(best_model_path, self.submission_dir)
            self.update_best_checkpoint = False
            
            # Save metadata
            with open(os.path.join(self.submission_dir, "loss.txt"), "w") as f:
                f.write(f"{best_step},{best_eval_loss},{model_type}")
            
            # Save detailed info
            with open(os.path.join(self.submission_dir, "checkpoint_info.json"), "w") as f:
                json.dump(self.best_checkpoint_info, f, indent=2)


class GRPOCustomEvalSaveCallback(CustomEvalSaveCallback):
    def compute_loss(self, state: TrainerState, metrics):
        eval_loss = None
        if state.log_history:
            last_log_entry = state.log_history[-1]
            eval_loss = last_log_entry.get("eval_reward", None)
            print(f"choose eval_loss ({eval_loss}) as eval_reward from: last_log_entry: {last_log_entry}; \n metrics: {metrics}", flush=True)
        else:
            print(f"state.log_history is empty", flush=True)
            
        if eval_loss is not None:
            eval_loss = - eval_loss
            
        return eval_loss
    
    def penalize_eval_loss(self, eval_loss: float):
        if eval_loss < 0:
            return eval_loss / 3
        else:
            return eval_loss * 3


def check_remaining_time_less_than_minutes(end_time: str, minutes: int) -> bool: 
    end_time = datetime.datetime.strptime(end_time, "%Y-%m-%d %H:%M:%S")
    end_time = end_time.replace(tzinfo=timezone.utc)
    now = datetime.datetime.now(timezone.utc)
    time_diff = end_time - now
    result =  time_diff.total_seconds() < minutes * 60
    if result:
        print(f"*** current time: {now} end_time: {end_time} time_diff: {time_diff}", flush=True)
    return result


class WhenToEvalHandler:
    def __init__(self, end_time: str, save_before_remaining_time: int = 3, periodic_save_steps: int = -1, steps_per_epoch: int = -1, max_steps: int = -1):
        self.save_before_remaining_time = save_before_remaining_time
        self.run_eval = False
        self.end_time = end_time
        self.periodic_save_steps = periodic_save_steps
        self.steps_per_epoch = steps_per_epoch
        self.max_steps = max_steps

    def __call__(self, global_step: int) -> dict:
        if self.steps_per_epoch != -1 and global_step % self.steps_per_epoch == 0 and global_step > 1:
            return {"eval": True, "reason": "epoch"}
        
        if self.periodic_save_steps != -1 and global_step % self.periodic_save_steps == 0 and global_step > 1:
            return {"eval": True, "reason": "periodic"}
        
        if self.save_before_remaining_time > 0 and not self.run_eval:
            if check_remaining_time_less_than_minutes(self.end_time, self.save_before_remaining_time):
                print(f"***ALERT: The time is about to run out need to eval & save the model", flush=True)
                self.run_eval = True
                return {"eval": True, "reason": "end_time"}
        
        if self.max_steps != -1 and global_step == self.max_steps:
            print(f"Stop training because of max steps: {self.max_steps}", flush=True)
            return {"eval": True, "reason": "max_step"}

        return {"eval": False, "reason": "none"}


def set_generation_config(model_name, model):
    try:
        if model_name in ERROR_GENERATION_CONFIG_MODELS:
            model.generation_config = GenerationConfig(temperature=None, top_p=None)
    except:
        print(f"Error setting generation config for model {model_name}")
        pass


def resize_if_needed(model_name, model, token_nums):
    try:
        if model_name in MIS_MATCH_VOCAB_SIZE_MODELS:
            model.resize_token_embeddings(token_nums)
    except:
        print(f"Error resizing token embeddings for model {model_name}")
        pass


def init_wandb(train_request: Dict):
    return True
    task_id = train_request["task_id"]
    expected_repo_name = train_request["expected_repo_name"]
    os.environ["WANDB_MODE"] = "offline"
    os.environ["WANDB_DIR"] = train_request["wandb_log_dir"]
    os.environ["WANDB_RUN_ID"] = f"{task_id}_{expected_repo_name}"
    os.environ["WANDB_NAME"] = f"{task_id}_{expected_repo_name}"
    if is_main_process(LOCAL_RANK):
        os.makedirs(train_request["wandb_log_dir"], exist_ok=True)
    return True
