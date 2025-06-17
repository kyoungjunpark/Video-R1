import json
import wandb

# === Step 1: Load the trainer_state.json ===
with open('chekpoint-2000/trainer_state.json', 'r') as f:
    trainer_state = json.load(f)

# === Step 2: Initialize wandb ===
wandb.init(project="CoT-0612", name="trainer-state-upload", resume="allow")

# === Step 3: Upload log_history entries step-by-step ===
log_history = trainer_state.get("log_history", [])

for entry in log_history:
    # Ensure 'step' exists for wandb logging
    step = entry.get("step")
    if step is None:
        continue

    # Flatten nested reward keys if needed
    flat_entry = {}
    for k, v in entry.items():
        if isinstance(v, dict):
            for subk, subv in v.items():
                flat_entry[f"{k}/{subk}"] = subv
        else:
            flat_entry[k] = v

    # Log to wandb at this step
    wandb.log(flat_entry, step=step)

# === Step 4: Upload trainer-level metadata as config ===
wandb.config.update({
    "best_metric": trainer_state.get("best_metric"),
    "best_model_checkpoint": trainer_state.get("best_model_checkpoint"),
    "epoch": trainer_state.get("epoch"),
    "global_step": trainer_state.get("global_step"),
    "eval_steps": trainer_state.get("eval_steps"),
    "is_hyper_param_search": trainer_state.get("is_hyper_param_search"),
    "is_local_process_zero": trainer_state.get("is_local_process_zero"),
    "is_world_process_zero": trainer_state.get("is_world_process_zero"),
})
