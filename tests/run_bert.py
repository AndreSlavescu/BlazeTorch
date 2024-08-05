import torch
import blazetorch
import time
from typing import List
from transformers import BertModel, BertTokenizer
from transformers.activations import GELUActivation
from torch.profiler import profile, record_function, ProfilerActivity
from argparse import ArgumentParser
import os

torch.manual_seed(0)

def check_output(output, name, device):
    border = "=" * 50
    header = f" {name.upper()} OUTPUT CHECK ON {str(device).upper()} "
    print(f"\n{border}")
    print(f"{header:^{len(border)}}")
    print(f"{border}\n")

    has_nan = False
    for i, tensor in enumerate(output):
        if torch.isnan(tensor).any():
            has_nan = True
            print(f"  ❌ NaN detected in {name} output[{i}]:")
            print(f"     • Min: {tensor.min().item():.6f}")
            print(f"     • Max: {tensor.max().item():.6f}")
        else:
            print(f"  ✅ No NaN in {name} output[{i}]")
            print(f"     • Min: {tensor.min().item():.6f}")
            print(f"     • Max: {tensor.max().item():.6f}")
        print()

    if has_nan:
        print(f"❗ NaN values found in {name} output on {device}.")
    else:
        print(f"✨ All values are valid in {name} output on {device}.")

    print(f"\n{border}\n")

def replace_layers(model, device):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module, device)

        if isinstance(module, GELUActivation):
            setattr(model, n, torch.nn.ReLU(inplace=True).to(device))

def my_profile(f, x: List, iter: int = 1000, name: str = "profile.json", generate_trace: bool = False, device: str = "cpu"):
    start = time.perf_counter()
    for _ in range(iter):
        f(*x)  # optimized run
    end = time.perf_counter()
    avg_time = (end - start) / iter

    if generate_trace:
        if device == "cuda":
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as shape_prof:
                with record_function("model_inference"):
                    f(*x)
            print(shape_prof.key_averages(group_by_input_shape=True).table(sort_by="cuda_time_total", row_limit=15))

            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                f(*x)
            prof.export_chrome_trace(os.path.join('traces', name))
        else:
            with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as shape_prof:
                with record_function("model_inference"):
                    f(*x)
            print(shape_prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15))

            with profile(activities=[ProfilerActivity.CPU]) as prof:
                f(*x)
            prof.export_chrome_trace(os.path.join('traces', name))

    return avg_time

def test_layer(device: torch.device, generate_trace: bool, iter: int, debug: bool):
    try:
        batch_size = 1
        seq_length = 32 
        vocab_size = 30522 

        tokens_tensor = torch.randint(0, vocab_size, (batch_size, seq_length), dtype=torch.long).to(device)
        segments_tensors = torch.randint(0, 2, (batch_size, seq_length), dtype=torch.long).to(device)
        attention_mask = torch.randint(0, 2, (batch_size, seq_length), dtype=torch.long).to(device)

        model = BertModel.from_pretrained("bert-base-uncased")
        model.config.torchscript = True
        model = model.to(device)
        model.eval()
        with torch.no_grad():
            blazetorch.enable()
            trace_jit = torch.jit.trace(model, [tokens_tensor, attention_mask, segments_tensors], check_trace=False, check_tolerance=2)
            trace_jit = torch.jit.freeze(trace_jit.eval())

        inputs = [tokens_tensor, attention_mask, segments_tensors]

        trace_jit(*inputs)  # run optimize and have execution_plan
        ti_optimized = my_profile(trace_jit, inputs, iter, f'optim_bertlayer_trace_{device}.json', generate_trace, device=device)
        print(f'TIME AFTER optimize on {device}: {ti_optimized} [s]')

        optimized_output = trace_jit(*inputs)
        check_output(optimized_output, "Optimized", device=device)

        blazetorch.disable()
        base_output = model(*inputs)
        check_output(base_output, "Base", device=device)

        if isinstance(optimized_output, tuple) and isinstance(base_output, tuple):
            outputs_match = all(torch.allclose(o, b, atol=1e-5, equal_nan=True) for o, b in zip(optimized_output, base_output))
        else:
            outputs_match = torch.allclose(optimized_output, base_output, atol=1e-5, equal_nan=True)

        if outputs_match:
            print(f"\nOutputs match on {device}.")
        else:
            if isinstance(optimized_output, tuple) and isinstance(base_output, tuple):
                max_diff = max(torch.max(torch.abs(o - b)) for o, b in zip(optimized_output, base_output))
            else:
                max_diff = torch.max(torch.abs(optimized_output - base_output))
            raise ValueError(f"Outputs do not match on {device}. Max diff: {max_diff:.6f}")
        
        if debug:
            print("\nDebug Information:")
            print(f"Model architecture:\n{model}")
            print(f"\nOptimized JIT graph:\n{trace_jit.graph}")
            print(f"\nLast executed optimized graph:\n{torch.jit.last_executed_optimized_graph()}")
        
        ti_base = my_profile(model, inputs, iter, f'base_bertlayer_trace_{device}.json', generate_trace, device=device)
        print(f'TIME BEFORE optimize on {device}: {ti_base} [s]')

        speedup = (ti_base - ti_optimized) / ti_base * 100
        print(f'\nOptimized layer inference time on {device}: {ti_optimized:.6f} seconds')
        print(f'Base layer inference time on {device}: {ti_base:.6f} seconds')
        print(f'Speedup achieved on {device}: {speedup:.2f}%')

    except Exception as e:
        print(f"Error in test_layer for device {device}: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    parser = ArgumentParser(description="Profile BERT")
    parser.add_argument('--generate_trace', action='store_true', help='Generate execution trace')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode')
    parser.add_argument('--iter', type=int, default=10, help='Number of iterations for profiling')
    args = parser.parse_args()

    if torch.cuda.is_available():
        test_layer(torch.device('cuda'), args.generate_trace, args.iter, args.debug)
    test_layer(torch.device('cpu'), args.generate_trace, args.iter, args.debug)