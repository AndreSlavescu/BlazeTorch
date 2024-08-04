import torch
import blazetorch
import time
from typing import List
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.activations import GELUActivation
from transformers.models.bert.modeling_bert import BertAttention
from torch.profiler import profile, record_function, ProfilerActivity
from argparse import ArgumentParser
import os

torch.manual_seed(0)

def replace_layers(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module)

        if isinstance(module, GELUActivation):
            setattr(model, n, torch.nn.ReLU(inplace=True))

def my_profile(f, x: List, iter: int = 1000, name: str = "profile.json"):
    start = time.perf_counter()
    for _ in range(iter):
        f(*x)  # optimized run
    end = time.perf_counter()
    avg_time = (end - start) / iter

    if args.generate_trace > 0:
        # Profile with shapes
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as shape_prof:
            with record_function("model_inference"):
                f(*x)
        print(shape_prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15))

        with profile(activities=[ProfilerActivity.CPU]) as prof:
            f(*x)
        prof.export_chrome_trace(os.path.join('traces', name))

    return avg_time

def test_layer():
    x = torch.rand(4, 100, 768)
    config = BertConfig(
        vocab_size_or_config_json_file=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        torchscript=True,
    )
    model = BertAttention(config)
    model.eval()

    iter = 1000
    with torch.no_grad():
        blazetorch.enable()
        trace_jit = torch.jit.trace(model, x, check_trace=False, check_tolerance=2)
        trace_jit = torch.jit.freeze(trace_jit.eval())

        trace_jit(x) # run optimize and have execution_plan
        ti = my_profile(trace_jit, [x], iter, 'optim_bertlayer_trace.json') # run with optim
        print('TIME AFTER optimize: {} [s] '.format(ti))
        
        print(torch.jit.last_executed_optimized_graph())
        blazetorch.disable()
        ti = my_profile(model, [x], iter, 'base_bertlayer_trace.json')
        print('TIME BEFORE optimize: {} [s] '.format(ti))
        print('done')

def test_model():
    enc = BertTokenizer.from_pretrained("bert-base-uncased")
    text = "All that glitters is not gold"
    tokenized_text = enc.tokenize(text)

    masked_index = 6  # Masking "gold" in "All that glitters is not gold"
    tokenized_text[masked_index] = "[MASK]"
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0] * len(indexed_tokens)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    
    config = BertConfig(
        vocab_size_or_config_json_file=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        torchscript=True,
    )
    model = BertModel(config)
    replace_layers(model)
    model.eval()
    iter = 1000

    def run_model(m, inputs):
        outputs = m(*inputs)
        return outputs

    with torch.no_grad():
        # Test with blazetorch optimization
        blazetorch.enable()
        trace_jit = torch.jit.trace(model, [tokens_tensor, segments_tensors], check_trace=False, check_tolerance=2)
        trace_jit = torch.jit.freeze(trace_jit.eval())
        
        # warm-up
        _ = run_model(trace_jit, [tokens_tensor, segments_tensors])
        ti_optimized = my_profile(run_model, [trace_jit, [tokens_tensor, segments_tensors]], iter, 'optim_bertmodel_trace.json')        
        
        # Test without blazetorch optimization
        blazetorch.disable()
        ti_base = my_profile(run_model, [model, [tokens_tensor, segments_tensors]], iter, 'base_bertmodel_trace.json')
        
        speedup = (ti_base - ti_optimized) / ti_base * 100
        optimized_output = run_model(trace_jit, [tokens_tensor, segments_tensors])
        base_output = run_model(model, [tokens_tensor, segments_tensors])
        
        print("Optimized model output:")
        print(optimized_output)
        print("\nBase model output:")
        print(base_output)
        
        outputs_match = all(torch.allclose(o, b, atol=1e-5) for o, b in zip(optimized_output, base_output))
        if outputs_match:
            print("\nOutputs match.")
        else:
            max_diff = max(torch.max(torch.abs(o - b)) for o, b in zip(optimized_output, base_output))
            raise ValueError(f"Outputs do not match. Max diff: {max_diff:.6f}")
        
        print(f'\n\nOptimized model inference time: {ti_optimized:.6f} seconds')
        print(f'Base model inference time: {ti_base:.6f} seconds')
        print(f'Speedup achieved: {speedup:.2f}%')

if __name__ == "__main__":
    parser = ArgumentParser(description="Profile BERT")
    parser.add_argument('--test_layer', action='store_true', help='Test only a single layer')
    parser.add_argument('--generate_trace', action='store_true', help='Generate execution trace')
    args = parser.parse_args()

    if args.test_layer:
        test_layer()
    else:
        test_model()
