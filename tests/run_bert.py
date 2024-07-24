import torch
import blazetorch
import time
from typing import List
from transformers import BertModel, BertTokenizer, BertConfig
from transformers.activations import GELUActivation
from transformers.models.bert.modeling_bert import BertAttention
from torch.profiler import profile, record_function, ProfilerActivity
from argparse import ArgumentParser


parser = ArgumentParser(description="Profile bert")
_ = parser.add_argument
_('--testL', action='store_true', help='test only layer')
_('--genTrace', action='store_true', help='generate trace')
args = parser.parse_args()

torch.manual_seed(0)

def replace_layers(model):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_layers(module)

        if isinstance(module, GELUActivation):
            setattr(model, n, torch.nn.ReLU(inplace=True))

def my_profile(f, x : List, iter = 1000, name="default.json"):
    start = time.time()
    for _ in range(iter):
        _ = f(*x) # run with optim
    end = time.time()

    if args.genTrace > 0:
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as profe:
            with record_function("model_inference"):
                f(*x)
        print(profe.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=15))
        with profile(activities=[ProfilerActivity.CPU]) as prof:
            f(*x)
        prof.export_chrome_trace(name)
        
    return (end - start)/iter

def test_layer():
    x = torch.rand(4, 100, 768) # creating a dummy input
    config = BertConfig(
        vocab_size_or_config_json_file=32000,
        hidden_size=768,
        num_hidden_layers=12,
        num_attention_heads=12,
        intermediate_size=3072,
        torchscript=True,
    )
    model = BertAttention(config)
    # model = BertLayer(config)
    model.eval()

    iter = 1000
    with torch.no_grad():
        blazetorch.enable()
        trace_jit = torch.jit.trace(model, x, check_trace=False, check_tolerance=2)
        trace_jit = torch.jit.freeze(trace_jit.eval())

        trace_jit(x) # run optimize and have execution_plan
        ti = my_profile(trace_jit, [x], iter, 'optim_bertlayer_trace.json')# run with optim
        print('TIME AFTER optimize: {} [s] '.format(ti))
        
        # print(torch.jit.last_executed_optimized_graph())
        blazetorch.disable()
        ti = my_profile(model, [x], iter, 'base_bertlayer_trace.json')#run without optim
        print('TIME BEFORE optimize: {} [s] '.format(ti))
        print('done')

def test_model():
    
    enc = BertTokenizer.from_pretrained("bert-base-uncased")
    # Tokenizing input text
    text = "[CLS] Who was Jim Henson ? [SEP] Jim Henson was a puppeteer [SEP]"
    tokenized_text = enc.tokenize(text)
    # Masking one of the input tokens
    masked_index = 8
    tokenized_text[masked_index] = "[MASK]"
    indexed_tokens = enc.convert_tokens_to_ids(tokenized_text)
    segments_ids = [0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1]
    # Creating a dummy input
    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])
    x = [tokens_tensor, segments_tensors]
    
    # x = torch.rand(4, 100, 768) # creating a dummy input
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
    with torch.no_grad():
        blazetorch.enable()
        trace_jit = torch.jit.trace(model, [tokens_tensor, segments_tensors], check_trace=False, check_tolerance=2)
        trace_jit = torch.jit.freeze(trace_jit.eval())
        trace_jit(tokens_tensor, segments_tensors) # run optimize and have execution_plan

        ti = my_profile(trace_jit, [tokens_tensor, segments_tensors], iter, 'optim_bertmodel_trace.json') # run with optim
        print('optimized model time: {} seconds '.format(ti))
        
        blazetorch.disable()
        ti = my_profile(model, [tokens_tensor, segments_tensors], iter, 'base_bertmodel_trace.json') # run without optim
        print('base model time: {} seconds '.format(ti))
 
        print('done')

if __name__ == "__main__":
    if args.testL:
        test_layer()
    else:
        test_model()
