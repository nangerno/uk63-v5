#!/usr/bin/env python3
"""
Standalone script for text model training (InstructText, DPO, and GRPO)
"""

import argparse
import asyncio
import json
import os
import shutil
import copy
import subprocess
import sys
import uuid
import re
import time 
from datetime import datetime, timezone, timedelta

import yaml
from transformers import AutoTokenizer
from state_manager import get_state, set_state
import numpy as np


script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(script_dir)
sys.path.append(project_root)

import train_cst
from core.config.config_handler import create_dataset_entry
from core.config.config_handler import save_config
from core.config.config_handler import update_flash_attention
from core.dataset_utils import adapt_columns_for_dpo_dataset
from core.dataset_utils import adapt_columns_for_grpo_dataset
from core.models.utility_models import DpoDatasetType
from core.models.utility_models import FileFormat
from core.models.utility_models import GrpoDatasetType
from core.models.utility_models import InstructTextDatasetType
from core.models.utility_models import TaskType
import training_paths as train_paths
from instruct_config import get_training_json as get_instruct_training_json
from dpo_config import get_training_json as get_dpo_training_json
from grpo_config import get_training_json as get_grpo_training_json
import pathlib
from transformers import AutoConfig
import lr_utils
from two_phase_trainer import TwoPhaseTrainingStrategy, extract_value_from_cmd, replace_args_in_cmd

def run_cmd_with_log(cmd: str, log_file_path: str, env_vars: dict = None):
    # print(f"Running command: {cmd}", flush=True)
    with open(log_file_path, "w") as log_file:
        # Prepare environment variables
        process_env = os.environ.copy()
        if env_vars:
            process_env.update(env_vars)

        # Run the command, capturing stdout and stderr
        process = subprocess.Popen(
            cmd,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=process_env,
        )

        # Stream output to both console and log file
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
            log_file.flush()

        # Wait for the process to complete
        return_code = process.wait()

        # Log the return code
        log_file.write(f"\nProcess completed with return code: {return_code}\n")


def replace_args_in_cmd(cmd: str, arg_name: str, arg_value: str):
    match = re.search(f"(?P<p>--{arg_name}(\s+)([^\s]+))(\s+)", cmd)
    if match:
        left_index = match.start("p")
        right_index = match.end("p")
        return cmd[:left_index] + f" --{arg_name} {arg_value} " + cmd[right_index:]
    else:
        return None


def extract_value_from_cmd(cmd: str, arg_name: str):
    match = re.search(f"(?P<p>--{arg_name}(\s+)(?P<value>[^\s]+))(\s+)", cmd)
    if match:
        return match.group("value")
    else:
        return None


def get_model_architecture(model_name: str) -> str:
    try:
        config = AutoConfig.from_pretrained(model_name)
        architectures = config.architectures
        if len(architectures) > 1:
            return "Multiple architectures"
        return architectures[0].strip().lower()
    except Exception as e:
        if "model type `gpt_oss`" in str(e):
            return "GptOssForCausalLM"
        return "Unknown"


def is_openai_model(model_name: str) -> bool:
    architecture = get_model_architecture(model_name)
    if architecture.lower() == "gptossforcausallm":
        return True
    return False


OOM_ERROR = "torch.OutOfMemoryError: CUDA out of memory"
VLLM_OOM_ERROR = "ValueError: No available memory for the cache blocks"


def get_error_type(log_path: str):
    with open(log_path, "r") as f:
        text = f.read()
    if OOM_ERROR in text:
        return OOM_ERROR
    elif VLLM_OOM_ERROR in text:
        return VLLM_OOM_ERROR
    else:
        return None


def extract_output_dir(train_cmd: str) -> str:
    match = re.search(r"--output_dir\s+(.*?)\s+", train_cmd)
    if match:
        return match.group(1)
    else:
        return None


def run_training(
    train_cmd: str,
    log_path: str,
    task_id: str,
    retries: int,
    task_type: str,
    expected_repo_name: str,
):
    for i in range(retries):
        print(
            f"************* Training attempt {i+1}/{retries} for task {task_id}*************",
            flush=True,
        )
        if i > 0:  # there was something wrong so we will reduce the batch_size
            # first check if the training is OOM
            if os.path.exists(log_path):
                error_type = get_error_type(log_path)
                if error_type == OOM_ERROR:
                    current_batch_size = extract_value_from_cmd(
                        train_cmd, "per_device_train_batch_size"
                    )
                    current_batch_size = int(current_batch_size)
                    if current_batch_size > 1:
                        new_batch_size = current_batch_size // 2
                        print(
                            f"Reducing batch size from {current_batch_size} to {new_batch_size}",
                            flush=True,
                        )
                        train_cmd = replace_args_in_cmd(
                            train_cmd,
                            "per_device_train_batch_size",
                            str(new_batch_size),
                        )
                        # print(f"New train command: {train_cmd}", flush=True)
                    else:
                        print(f"batch size is 1, cannot reduce further", flush=True)
                        if task_type == TaskType.GRPOTASK.value:
                            # disable vllm
                            train_cmd = replace_args_in_cmd(
                                train_cmd, "use_vllm", "False"
                            )
                            # print(f"disable VLLM {train_cmd}", flush=True)
                elif error_type == VLLM_OOM_ERROR:
                    if task_type == TaskType.GRPOTASK.value:
                        print(f"VLLM OOM error, disable VLLM", flush=True)
                        train_cmd = replace_args_in_cmd(train_cmd, "use_vllm", "False")

        # empty the log file if it exists
        if os.path.exists(log_path):
            with open(log_path, "w") as f:
                f.write("STARTING TRAINING")

        training_env_vars = {
            "WANDB_MODE": "offline",
            "WANDB_RUN_ID": f"{task_id}_{expected_repo_name}",
            "WANDB_NAME": f"{task_id}_{expected_repo_name}",
        }

        run_cmd_with_log(train_cmd, log_path, env_vars=training_env_vars)
        # check if training is successfully here so we can break the loop; if output_dir contains file: "successs.txt" return true
        output_dir = extract_value_from_cmd(train_cmd, "output_dir")
        if os.path.exists(os.path.join(output_dir, "success.txt")):
            return True
        time.sleep(5)
    return False


def patch_wandb_symlinks(base_dir: str):
    for root, _, files in os.walk(base_dir):
        for name in files:
            full_path = os.path.join(root, name)

            if os.path.islink(full_path):
                target_path = os.readlink(full_path)

                print(f"Symlink: {full_path} → {target_path}")
                try:
                    os.unlink(full_path)
                except Exception as e:
                    print(f"Failed to unlink {full_path}: {e}")
                    continue

                if os.path.exists(target_path):
                    print("Copying real file")
                    try:
                        shutil.copy(target_path, full_path)
                    except Exception as e:
                        print(f"Failed to copy: {e}")
                else:
                    print("Target not found, creating dummy")
                    pathlib.Path(full_path).touch()


def delete_poor_checkpoints(train_runs: list[dict]):
    lowest_loss = min([run["current_loss"] for run in train_runs])
    for run in train_runs:
        if run["current_loss"] > lowest_loss:
            if os.path.exists(run["output_dir"]):
                print(f"Deleting checkpoint {run['output_dir']} with loss {run['current_loss']}", flush=True)
                shutil.rmtree(run["output_dir"])


def get_log_scale(task_type: str):
    log_scale_map = {
        TaskType.INSTRUCTTEXTTASK.value: 0.18,
        TaskType.DPOTASK.value: 0.18,
        TaskType.GRPOTASK.value: 0.2,
        TaskType.CHATTASK.value: 0.18,
    }
    return log_scale_map[task_type]


def main():
    print("---STARTING TEXT TRAINING SCRIPT---", flush=True)
    parser = argparse.ArgumentParser(description="Text Model Training Script")
    parser.add_argument("--task-id", required=True, help="Task ID")
    parser.add_argument("--model", required=True, help="Model name or path")
    parser.add_argument(
        "--dataset", required=True, help="Dataset path or HF dataset name"
    )
    parser.add_argument(
        "--dataset-type", required=True, help="JSON string of dataset type config"
    )
    parser.add_argument(
        "--task-type",
        required=True,
        choices=["InstructTextTask", "DpoTask", "GrpoTask", "ChatTask"],
        help="Type of task",
    )
    parser.add_argument(
        "--file-format",
        required=False,
        choices=["csv", "json", "hf", "s3"],
        help="File format",
        default="s3",
    )
    parser.add_argument(
        "--hours-to-complete",
        type=float,
        required=True,
        help="Number of hours to complete the task",
    )
    parser.add_argument("--expected-repo-name", help="Expected repository name")
    parser.add_argument(
        "--max-data-size",
        type=int,
        help="Max data size to use for training",
        default=-1,
    )
    parser.add_argument(
        "--max-steps", type=int, help="Max steps to use for training", default=-1
    )
    parser.add_argument("--retries", type=int, help="Number of retries", default=5)
    parser.add_argument(
        "--min-steps", type=int, help="Min steps to use for training", default=100
    )

    parser.add_argument(
        "--reg-ratio", type=float, help="Reg ratio to use for training", default=1.24383
    )

    args = parser.parse_args()
    original_model_name = args.model
    original_task_type = args.task_type

    for directory in train_cst.AXOLOTL_DIRECTORIES.values():
        os.makedirs(directory, exist_ok=True)
    try:
        dataset_type_dict = json.loads(args.dataset_type)
    except Exception as e:
        sys.exit(f"Error creating dataset type object: {e}")

    dataset_path = train_paths.get_text_dataset_path(args.task_id)
    submission_dir = train_paths.get_checkpoints_output_path(
        args.task_id, args.expected_repo_name
    )
    print(f"submission_dir: {submission_dir}", flush=True)
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir, exist_ok=True)

    output_dir = f"/workspace/scripts/soutputs/{args.task_id}"
    os.makedirs(output_dir, exist_ok=True)

    end_time = datetime.now(timezone.utc) + timedelta(
        hours=args.hours_to_complete - 3 / 60
    )  # assume that 3 minutes to go this far
    end_time = end_time.strftime("%Y-%m-%d %H:%M:%S")
    print("end_time: ", end_time, flush=True)

    ds_folder = "datasets"
    os.makedirs(ds_folder, exist_ok=True)
    request_path = os.path.join(ds_folder, f"training_request_{args.task_id}.json")
    model_path = str(train_paths.get_text_base_model_path(original_model_name))

    is_openai = False
    if is_openai_model(original_model_name):
        print("Upgrading python packages for openai model", flush=True)
        run_cmd_with_log(
            "pip uninstall -y transformers && pip install transformers==4.55.0",
            os.path.join(ds_folder, f"upgrade_transformers.log"),
        )
        # upgrade deepspeed
        run_cmd_with_log(
            "pip uninstall -y deepspeed && pip install deepspeed==0.17.4",
            os.path.join(ds_folder, f"upgrade_deepspeed.log"),
        )
        # install kernel
        run_cmd_with_log(
            "pip install kernels==0.9.0", os.path.join(ds_folder, f"install_kernel.log")
        )
        is_openai = True

    train_info = {
        "model_name": original_model_name,
        "model_path": model_path,
        "task_id": args.task_id,
        "dataset": dataset_path,
        "hours_to_complete": args.hours_to_complete,
        "expected_repo_name": args.expected_repo_name,
        "end_time": end_time,
        "dataset_type": dataset_type_dict,
        "submission_dir": submission_dir,
        "output_dir": output_dir,
        "adjust_batch_size": True,
        "request_path": request_path,
        "max_data_size": args.max_data_size,
        "max_steps": args.max_steps,
        "wandb_log_dir": train_cst.WANDB_LOGS_DIR,
        "min_steps": args.min_steps,
        "is_openai": is_openai,
        "reg_ratio": args.reg_ratio,
        "find_lk_lr": True,
        "checking_mode": "first_time",
    }

    if (
        args.task_type == TaskType.INSTRUCTTEXTTASK.value
        or args.task_type == TaskType.CHATTASK.value
    ):
        train_info = get_instruct_training_json(train_info)
        tokenize_cmd = (
            f"/workspace/axo_py/bin/python tokenize_instruct.py {request_path}"
        )
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.DPOTASK.value:
        train_info = get_dpo_training_json(train_info)
        tokenize_cmd = f"python tokenize_dpo.py {request_path}"
        train_cmd = train_info["run_cmd"]

    elif args.task_type == TaskType.GRPOTASK.value:
        train_info = get_grpo_training_json(train_info)
        tokenize_cmd = f"python tokenize_grpo.py {request_path}"
        train_cmd = train_info["run_cmd"]
    else:
        raise ValueError(f"Task type {args.task_type} not supported")

    
    with open(request_path, "w") as f:
        json.dump(train_info, f, indent=4, ensure_ascii=False)

    run_cmd_with_log(
        tokenize_cmd, os.path.join(ds_folder, f"tokenize_{args.task_id}.log")
    )

    original_train_cmd = train_cmd
    train_success = False
    
    # Initialize Two-Phase Training Strategy
    two_phase_strategy = TwoPhaseTrainingStrategy(args.hours_to_complete, args.task_type)
    
    # Skip two-phase for GRPO tasks (they have different evaluation)
    use_two_phase = args.task_type != TaskType.GRPOTASK.value
    
    if use_two_phase:
        print("=" * 80, flush=True)
        print("TWO-PHASE TRAINING STRATEGY ENABLED", flush=True)
        print("Phase 1: Fast convergence (55% time) - Aggressive settings, early checkpoint", flush=True)
        print("Phase 2: Quality refinement (45% time) - Conservative settings, fine-tune from Phase 1", flush=True)
        print("=" * 80, flush=True)
        
        # Phase 1: Fast Convergence
        print("\n>>> STARTING PHASE 1: FAST CONVERGENCE", flush=True)
        phase1_start_time = datetime.now(timezone.utc)
        phase1_end_time_str = two_phase_strategy.get_phase_end_time(phase1_start_time, "phase1")
        
        # Extract config from train_info for Phase 1
        # train_info has train_request dict, we need to work with that
        phase1_train_info = copy.deepcopy(train_info)
        phase1_train_request = phase1_train_info.get("train_request", {})
        
        # Extract learning rate from command - store for Phase 2
        base_lr_str = extract_value_from_cmd(original_train_cmd, "learning_rate")
        if base_lr_str:
            base_lr = float(base_lr_str)
        else:
            base_lr = 4e-5  # Default fallback
        
        # Extract batch size from command - store for Phase 2
        base_batch_size_str = extract_value_from_cmd(original_train_cmd, "per_device_train_batch_size")
        base_batch_size = int(base_batch_size_str) if base_batch_size_str else None
        
        # Extract gradient accumulation - store for Phase 2
        base_grad_accum_str = extract_value_from_cmd(original_train_cmd, "gradient_accumulation_steps")
        base_grad_accum = int(base_grad_accum_str) if base_grad_accum_str else None
        
        # Create Phase 1 config dict for command modification
        phase1_config = {
            "learning_rate": base_lr * 1.5,  # Aggressive LR
            "warmup_steps": max(10, int(phase1_train_request.get("warmup_steps", 35) * 0.6)),
        }
        
        # Add batch size if extracted
        if base_batch_size:
            phase1_config["batch_size"] = int(base_batch_size * 1.2)
        
        # Add gradient accumulation if extracted
        if base_grad_accum:
            phase1_config["gradient_accumulation_steps"] = base_grad_accum
        
        phase1_output_dir = os.path.join(output_dir, "phase1")
        os.makedirs(phase1_output_dir, exist_ok=True)
        
        # Modify train command for Phase 1
        phase1_train_cmd = two_phase_strategy.modify_train_cmd_for_phase(
            original_train_cmd, phase1_config, phase1_output_dir, request_path
        )
        
        # Update train_request for Phase 1
        phase1_train_request["end_time"] = phase1_end_time_str
        phase1_train_request["checking_mode"] = "first_time"
        phase1_train_request["skip_evaluation"] = True  # Skip eval to save time
        
        # Calculate early checking step (30-50 steps vs competitor's 100)
        # Estimate total steps - use checking_step as reference or default
        default_checking = phase1_train_request.get("checking_step", 70)
        estimated_total_steps = max(1000, default_checking * 20)  # Rough estimate
        early_checking_step = two_phase_strategy.get_early_checking_step(estimated_total_steps, "phase1")
        phase1_train_request["checking_step"] = early_checking_step
        print(f"Phase 1 checking step: {early_checking_step} (vs competitor's 100)", flush=True)
        
        phase1_request_path = os.path.join(ds_folder, f"training_request_{args.task_id}_phase1.json")
        with open(phase1_request_path, "w") as f:
            json.dump(phase1_train_info, f, indent=4, ensure_ascii=False)
        
        phase1_train_cmd = replace_args_in_cmd(phase1_train_cmd, "request_path", phase1_request_path)
        
        # Run Phase 1 training
        phase1_log_path = os.path.join(ds_folder, f"train_{args.task_id}_phase1.log")
        phase1_success = run_training(
            phase1_train_cmd,
            phase1_log_path,
            args.task_id,
            args.retries,
            args.task_type,
            args.expected_repo_name,
        )
        
        if not phase1_success:
            print("Phase 1 training failed, falling back to standard training", flush=True)
            use_two_phase = False
        else:
            # Find Phase 1 checkpoint
            phase1_checkpoint = None
            if os.path.exists(phase1_output_dir):
                # Look for latest checkpoint
                try:
                    checkpoints = [d for d in os.listdir(phase1_output_dir) if d.startswith("checkpoint-")]
                    if checkpoints:
                        # Get checkpoint with highest step number
                        checkpoint_steps = []
                        for c in checkpoints:
                            try:
                                step_num = int(c.split("-")[1])
                                checkpoint_steps.append((step_num, c))
                            except (ValueError, IndexError):
                                continue
                        
                        if checkpoint_steps:
                            latest_step, latest_checkpoint = max(checkpoint_steps, key=lambda x: x[0])
                            phase1_checkpoint = os.path.join(phase1_output_dir, latest_checkpoint)
                            print(f"Phase 1 checkpoint found: {phase1_checkpoint} (step {latest_step})", flush=True)
                        else:
                            # Use output_dir as checkpoint if parsing failed
                            phase1_checkpoint = phase1_output_dir
                            print(f"Using Phase 1 output_dir as checkpoint: {phase1_checkpoint}", flush=True)
                    else:
                        # Check if success.txt exists (training completed)
                        if os.path.exists(os.path.join(phase1_output_dir, "success.txt")):
                            phase1_checkpoint = phase1_output_dir
                            print(f"Phase 1 completed, using output_dir as checkpoint: {phase1_checkpoint}", flush=True)
                        else:
                            print(f"No checkpoints found in {phase1_output_dir}", flush=True)
                except Exception as e:
                    print(f"Error finding Phase 1 checkpoint: {e}", flush=True)
                    # Try using output_dir as fallback
                    if os.path.exists(phase1_output_dir):
                        phase1_checkpoint = phase1_output_dir
            
            if phase1_checkpoint and os.path.exists(phase1_checkpoint):
                # Phase 2: Quality Refinement
                print("\n>>> STARTING PHASE 2: QUALITY REFINEMENT", flush=True)
                phase2_start_time = datetime.now(timezone.utc)
                phase2_end_time_str = two_phase_strategy.get_phase_end_time(phase2_start_time, "phase2")
                
                # Create Phase 2 config - conservative settings
                phase2_train_info = copy.deepcopy(train_info)
                phase2_train_request = phase2_train_info.get("train_request", {})
                
                # Phase 2 config - conservative settings
                phase2_config = {
                    "learning_rate": base_lr * 0.5,  # Conservative LR (half of base)
                    "warmup_steps": int(phase2_train_request.get("warmup_steps", 35) * 1.5),
                    "resume_from_checkpoint": phase1_checkpoint,
                }
                
                # Smaller batch size for Phase 2
                if base_batch_size:
                    phase2_config["batch_size"] = max(2, int(base_batch_size * 0.8))
                
                # Adjust gradient accumulation
                if base_grad_accum:
                    phase2_config["gradient_accumulation_steps"] = base_grad_accum
                
                phase2_output_dir = submission_dir  # Final output goes to submission_dir
                os.makedirs(phase2_output_dir, exist_ok=True)
                
                # Modify train command for Phase 2
                phase2_train_cmd = two_phase_strategy.modify_train_cmd_for_phase(
                    original_train_cmd, phase2_config, phase2_output_dir, request_path
                )
                
                # Update train_request for Phase 2
                phase2_train_request["end_time"] = phase2_end_time_str
                phase2_train_request["checking_mode"] = "none"  # No early stopping in Phase 2
                phase2_train_request["skip_evaluation"] = False  # Full evaluation in Phase 2
                
                # Calculate checking step for Phase 2 (still earlier than 100)
                phase2_checking_step = two_phase_strategy.get_early_checking_step(estimated_total_steps, "phase2")
                phase2_train_request["checking_step"] = phase2_checking_step
                print(f"Phase 2 checking step: {phase2_checking_step}", flush=True)
                
                phase2_request_path = os.path.join(ds_folder, f"training_request_{args.task_id}_phase2.json")
                with open(phase2_request_path, "w") as f:
                    json.dump(phase2_train_info, f, indent=4, ensure_ascii=False)
                
                phase2_train_cmd = replace_args_in_cmd(phase2_train_cmd, "request_path", phase2_request_path)
                
                # Run Phase 2 training
                phase2_log_path = os.path.join(ds_folder, f"train_{args.task_id}_phase2.log")
                train_success = run_training(
                    phase2_train_cmd,
                    phase2_log_path,
                    args.task_id,
                    args.retries,
                    args.task_type,
                    args.expected_repo_name,
                )
            else:
                print("Phase 1 checkpoint not found, falling back to standard training", flush=True)
                use_two_phase = False
    
    # Fallback to original logic if two-phase failed or GRPO task
    if not use_two_phase:
        print("Using standard training strategy", flush=True)
        state = get_state()
        state = {}
        set_state(state) # reset first
        state["mode"] = "initial"
        set_state(state)
        
        count = 0
        while True:
            state = get_state()
            train_cmd = original_train_cmd
            c_train_info = copy.deepcopy(train_info)
            final_output_dir = None
            
            if args.task_type == TaskType.GRPOTASK.value:
                state["mode"] = "finish"
                c_train_info["train_request"]["checking_mode"] = "none"
            else:
                if state["mode"] == "initial":
                    c_train_info["train_request"]["checking_mode"] = "first_time"
                elif state["mode"] == "continue":
                    c_train_info["train_request"]["checking_mode"] = "second_time"
                    n_runs = state["next_runs"]
                    if "lrs" not in state:
                        current_lr = float(state["train"]["lr"])
                        state["lrs"] = lr_utils.extend_learning_rates(current_lr, n_runs, log_range=get_log_scale(args.task_type))
                        assert len(state["lrs"]) == n_runs
                        state["runs"] = []
                    set_state(state)
                    state["runs"].append(state["train"].copy())
                    delete_poor_checkpoints(state["runs"])
                    if len(state["runs"]) < n_runs:
                        index = len(state["runs"])
                        train_cmd = replace_args_in_cmd(train_cmd, "learning_rate", str(state["lrs"][index]))
                    else:
                        c_train_info["train_request"]["checking_mode"] = "none"
                        index = np.argmin([run["current_loss"] for run in state["runs"]])
                        print(f"BL;{index};{state['runs'][index]['current_loss']}; {state['lrs'][index]}", flush=True)
                        train_cmd = state["runs"][index]["train_cmd"]
                        final_output_dir = state["runs"][index]["output_dir"]
                        state["mode"] = "finish"
                else:
                    assert state["mode"] == "finish"
                    break
            
            set_state(state)
            if train_cmd:
                run_output_dir = output_dir + f"_{count}" if not final_output_dir else final_output_dir
                train_cmd = replace_args_in_cmd(train_cmd, "output_dir", run_output_dir)
                current_request_path = os.path.join(ds_folder, f"training_request_{args.task_id}_{count}.json")
                with open(current_request_path, "w") as f:
                    json.dump(c_train_info, f, indent=4, ensure_ascii=False)
                train_cmd = replace_args_in_cmd(train_cmd, "request_path", current_request_path)
                
                state["train"] = {
                    "train_cmd": train_cmd,
                    "log_path": os.path.join(ds_folder, f"train_{args.task_id}.log"),
                    "lr": extract_value_from_cmd(train_cmd, "learning_rate"),
                    "output_dir": run_output_dir
                }
                state["train"]["start_time"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                set_state(state)
                
                log_path = state["train"]["log_path"]
                success = run_training(
                    train_cmd, log_path, args.task_id, args.retries, args.task_type, args.expected_repo_name,
                )
                time.sleep(5)
                if not success:
                    print(f"Training failed for task {args.task_id} at count={count}", flush=True)
                    break
                train_success = success
            count += 1

    if not os.path.exists(submission_dir) or len(os.listdir(submission_dir)) < 2:
        print(f"Training failed for task {args.task_id}", flush=True)
    else:
        print(f"Training successfully done for task {args.task_id}", flush=True)
        train_success = True

    if not train_success:
        print(f"Training failed for task {args.task_id}", flush=True)
        # add noise to the model
        add_noise_cmd = f"python add_random_noise.py {model_path} {submission_dir}"
        run_cmd_with_log(
            add_noise_cmd, os.path.join(ds_folder, f"add_noise_{args.task_id}.log")
        )

    patch_wandb_symlinks(train_cst.WANDB_LOGS_DIR)


if __name__ == "__main__":
    main()
