"""Actual pretrained causal-LM fine-tuning with full tuning or LoRA.
Optional dependencies: requirements-huggingface.txt. First run downloads a model.
Run: python3 02_finetuning/04_huggingface_sft.py --method lora --steps 20
For a tiny plumbing check: add --model sshleifer/tiny-gpt2.
For NVIDIA QLoRA: add --qlora (install bitsandbytes separately).

This deliberately uses a plain PyTorch loop to expose labels and updates.
The tiny bundled dataset teaches the workflow; it cannot create a capable bot.
Default distilgpt2 is a base completion model, not an instruction/chat model.
For chat models adapt formatting to their chat template and verify the exact
assistant-token mask; generic string concatenation is not a universal template.
"""
import argparse
import json
from pathlib import Path
import torch
from torch.utils.data import DataLoader


def encode_row(row, tokenizer, max_length):
    prefix = tokenizer.encode('Question: '+row['prompt']+'\nAnswer:', add_special_tokens=False)
    answer = tokenizer.encode(' '+row['response'], add_special_tokens=False) + [tokenizer.eos_token_id]
    # Preserve full examples. Silently truncating a response can train bad EOS behavior.
    if len(prefix)+len(answer) > max_length:
        raise ValueError('Example exceeds --max-length; shorten it or increase the limit.')
    return {'input_ids': prefix+answer, 'labels': [-100]*len(prefix)+answer}


def collate(rows, pad_id):
    size = max(len(row['input_ids']) for row in rows)
    batch = {'input_ids': [], 'labels': [], 'attention_mask': []}
    for row in rows:
        length = len(row['input_ids'])
        batch['input_ids'].append(row['input_ids']+[pad_id]*(size-length))
        batch['labels'].append(row['labels']+[-100]*(size-length))
        # Mask by position, not token equality: PAD may share the EOS ID.
        batch['attention_mask'].append([1]*length+[0]*(size-length))
    return {key: torch.tensor(value) for key, value in batch.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='distilgpt2')
    parser.add_argument('--method', choices=['full', 'lora'], default='lora')
    parser.add_argument('--steps', type=int, default=20)
    parser.add_argument('--max-length', type=int, default=128)
    parser.add_argument('--data', type=Path, default=Path(__file__).parent/'data/train.jsonl')
    parser.add_argument('--validation', type=Path, default=Path(__file__).parent/'data/validation.jsonl')
    parser.add_argument('--output', type=Path, default=Path(__file__).resolve().parents[1]/'outputs/sft')
    parser.add_argument('--qlora', action='store_true')
    args = parser.parse_args()
    if args.steps < 1:
        parser.error('--steps must be positive')
    if args.qlora and (args.method != 'lora' or not torch.cuda.is_available()):
        parser.error('This QLoRA example requires --method lora and an NVIDIA CUDA GPU.')
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    torch.manual_seed(7)
    torch.set_num_threads(2)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.eos_token_id is None:
        raise ValueError('This example requires an EOS token.')
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    kwargs = {}
    if args.qlora:
        compute_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs.update(quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type='nf4',
            bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=compute_dtype),
            device_map={'': 0})
        # NF4 compresses frozen base weights; adapters remain trainable floats.
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)
    if args.qlora:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    else:
        model.to(device)
    if args.method == 'lora':
        from peft import LoraConfig, get_peft_model
        config = LoraConfig(task_type='CAUSAL_LM', r=8, lora_alpha=16,
                            lora_dropout=0.05, target_modules='all-linear')
        model = get_peft_model(model, config)
        model.print_trainable_parameters()
    model.config.use_cache = False  # KV caching is for generation, not this training loop.
    def load(path):
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        if not rows:
            raise ValueError(f'Empty dataset: {path}')
        return rows
    train_rows, val_rows = load(args.data), load(args.validation)
    if {r['prompt'] for r in train_rows} & {r['prompt'] for r in val_rows}:
        raise ValueError('Overlapping prompts between train and validation.')
    batcher = lambda rows: collate(rows, tokenizer.pad_token_id)
    train = DataLoader([encode_row(r, tokenizer, args.max_length) for r in train_rows],
                       batch_size=2, shuffle=True, collate_fn=batcher)
    val = DataLoader([encode_row(r, tokenizer, args.max_length) for r in val_rows],
                     batch_size=2, collate_fn=batcher)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=2e-4 if args.method == 'lora' else 2e-5)
    def evaluate():
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for batch in val:
                batch = {k: v.to(device) for k, v in batch.items()}
                n = batch['labels'][:, 1:].ne(-100).sum().item()
                total += model(**batch).loss.item()*n
                count += n
        return total/count  # Token-weighted mean, not a mean of unequal batches.
    print('validation before:', evaluate())
    iterator = iter(train)
    for step in range(args.steps):
        try:
            batch = next(iterator)
        except StopIteration:
            iterator = iter(train)
            batch = next(iterator)
        model.train()
        batch = {k: v.to(device) for k, v in batch.items()}
        optimizer.zero_grad(set_to_none=True)
        loss = model(**batch).loss  # Model shifts labels internally.
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        if step % 5 == 0:
            print('step:', step, 'train loss:', loss.item())
    print('validation after:', evaluate())
    args.output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    (args.output/'run_config.json').write_text(json.dumps(vars(args), default=str, indent=2))
    print('saved to:', args.output)
    # LoRA output contains adapters; reload with the ORIGINAL base model plus
    # PeftModel.from_pretrained(base, output). Full tuning saves full weights.

if __name__ == '__main__':
    main()
