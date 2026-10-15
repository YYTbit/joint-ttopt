import numpy as np


def synchronize_devices():
    import torch
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            torch.cuda.synchronize(device)


def memory_snapshot(reset=False):
    import torch
    result = {}
    if torch.cuda.is_available():
        for device in range(torch.cuda.device_count()):
            if reset:
                torch.cuda.reset_peak_memory_stats(device)
            result[str(device)] = {"torch_peak_allocated_bytes": torch.cuda.max_memory_allocated(device),
                                   "torch_peak_reserved_bytes": torch.cuda.max_memory_reserved(device)}
    return result


def derive_seed(seed, namespace, version=0, index=0, repeat=0):
    return int(np.random.SeedSequence([seed, namespace, version, index, repeat]).generate_state(1)[0])


def trajectory_work(trajectory, updating_agents=(), replay=True):
    updating = set(updating_agents)
    replayed = [decision for decision in trajectory.decisions if decision.agent in updating] if replay else []
    return {"joint_episodes": 1,
            "environment_steps": len(next(iter(trajectory.rewards.values()))),
            "action_forward_calls": len(trajectory.decisions),
            "action_forward_tokens": sum(len(decision.input_ids) for decision in trajectory.decisions),
            "gradient_forward_calls": len(replayed),
            "gradient_forward_tokens": sum(len(decision.input_ids) for decision in replayed),
            "gradient_backward_calls": len(replayed),
            "gradient_backward_tokens": sum(len(decision.input_ids) for decision in replayed)}


def add_work(records):
    keys = {key for record in records for key in record}
    return {key: sum(record.get(key, 0) for record in records) for key in sorted(keys)}
