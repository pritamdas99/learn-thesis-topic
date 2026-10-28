"""Reload a full checkpoint OR a LoRA adapter and generate a response.
Run after 04_huggingface_sft.py. No training occurs here. Only new tokens are decoded.
Exercise: compare greedy generation with temperature/top-p sampling.
"""
import argparse
from pathlib import Path
import torch


def main():
    torch.set_num_threads(1)  # Small examples run faster without large thread pools.
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/sft')
    parser.add_argument('--prompt', default='What is a gradient?')
    args = parser.parse_args()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    if (args.checkpoint/'adapter_config.json').exists():
        from peft import PeftConfig, PeftModel
        config = PeftConfig.from_pretrained(args.checkpoint)
        base = AutoModelForCausalLM.from_pretrained(config.base_model_name_or_path)
        model = PeftModel.from_pretrained(base, args.checkpoint)
    else:
        model = AutoModelForCausalLM.from_pretrained(args.checkpoint)
    # This reloads even a QLoRA-trained adapter on an unquantized CPU base.
    # Large bases may require quantized/GPU loading instead; see 04_huggingface_sft.py.
    model.eval()
    inputs = tokenizer('Question: '+args.prompt+'\nAnswer:', return_tensors='pt')
    with torch.no_grad():
        output = model.generate(**inputs, max_new_tokens=40, do_sample=False,
                                pad_token_id=tokenizer.pad_token_id)
    print(tokenizer.decode(output[0, inputs['input_ids'].shape[1]:], skip_special_tokens=True))

if __name__ == '__main__':
    main()
