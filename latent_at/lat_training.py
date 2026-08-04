import os
import wandb
import torch
import argparse
import gc
from datetime import datetime
from contextlib import contextmanager
import json
from dotenv import load_dotenv
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig
from tasks.harmbench.FastHarmBenchEvals import run_attack_evals, run_general_evals

from .lat_datasets import process_generic_chat_dataset, LatentAdversarialTrainingDataCollator
from .lat_methods import ProjectedGradLAT





@contextmanager
def eval_mode(model):
    """Context manager that temporarily sets model to eval() and restores previous mode on exit.

    Usage:
        with eval_mode(model):
            # run evaluation
    """
    # Some wrappers (deepspeed engines, etc.) may not expose `.training` attribute, so be defensive
    try:
        was_training = getattr(model, "training", None)
    except Exception:
        was_training = None
    try:
        # Set eval mode
        print("Setting model to eval mode for evaluation...")
        model.eval()
    except Exception:
        # If model doesn't support eval(), just yield and do nothing
        yield
        return
    try:
        yield
    finally:
        # Restore previous training mode only if we can determine it
        try:
            if was_training is True:
                print("Restoring model to train mode after evaluation...")
                model.train()
            elif was_training is False:
                print("Restoring model to eval mode after evaluation...")
                model.eval()
        except Exception:
            # Best-effort: ignore if restoration fails
            pass

def evaluate_model(model, tokenizer, model_type, cls, cls_tokenizer, cache_dir):
    """Evaluate the model on HarmBench and general capability tasks."""
    with eval_mode(model):
        with torch.no_grad():
            print("Running HarmBench evaluations...")
            harmbench_asr = run_attack_evals(model=model, 
                                             tokenizer=tokenizer,
                                             model_type=model_type,
                                             pretrained_cls="llama",
                                             cls=cls,
                                             cls_tokenizer=cls_tokenizer,
                                             do_sample=False,
                                             cache_dir=cache_dir,
                                             verbose=True,
                                             move_cls_device=True,
                                             move_model_device=True)
            
            harmbench_logs = {f"harmbench/{k}": v for k, v in harmbench_asr.items()}
            print("Running utility evaluations...")
            utility_acc = run_general_evals(model=model,
                                            tokenizer=tokenizer,
                                            model_type=model_type,
                                            evals_to_include=["MMLU", "HellaSwag", "Winogrande", "SciQ", "Lambada"])
            utility_logs = ({f"utility/{k}": v for k, v in utility_acc.items()})
    torch.cuda.empty_cache()
    gc.collect()
    return harmbench_logs, utility_logs


def load_model(model_name):
    model_dtype = torch.bfloat16

    print(f"Loading model {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=model_dtype,
        device_map="cuda"
    )
    print("Model loaded.")

    print("Loading tokenizer...")
    if "Llama-2" in model_name:
        model_type = "llama2"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif "Llama-3" in model_name:
        model_type = "llama3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif "Mistral" in model_name:
        model_type = "mistral"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = "left"
    elif "zephyr" in model_name:
        model_type = "zephyr"    
        tokenizer = AutoTokenizer.from_pretrained("HuggingFaceH4/zephyr-7b-beta")
        tokenizer.pad_token_id = tokenizer.unk_token_id
        tokenizer.padding_side = "left"
    elif "Qwen" in model_name:
        model_type = "qwen3"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    elif "Olmo" in model_name or "OLMo" in model_name:
        model_type = "olmo"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.padding_side = "left"
    else:
        print(model_name)
        raise Exception("Unsupported model type.")
    print("Tokenizer loaded.")
    return model, tokenizer, model_type

def load_data(harmful_dataset, benign_dataset, tokenizer, model_type, system_prompt, batch_size):
    # Normalize system prompt variable used for different model formats
    if model_type == "llama2": # LLama 2 Chat Formatting
        use_tokenizer_template = True
        custom_prompt_template = None
        custom_completion_template = None
    elif model_type == "llama3": # LLama 3 chat formatting
        use_tokenizer_template = False
        custom_prompt_template = f"<|start_header_id|>system<|end_header_id|>\n\n{system_prompt}<|eot_id|>"+"<|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        custom_completion_template="{completion}"
    elif model_type == "qwen3":  # Qwen 3 chat formatting
        use_tokenizer_template = False
        custom_prompt_template = "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        custom_completion_template="{completion}"
    elif model_type == "mistral":  # Mistral Instruct formatting (system folded into the user turn)
        use_tokenizer_template = False
        custom_prompt_template = "[INST] {system_prompt}\n\n{prompt} [/INST]"
        custom_completion_template="{completion}"
    elif model_type == "olmo":  # OLMo ChatML formatting (no <think> block)
        use_tokenizer_template = False
        custom_prompt_template = "<|im_start|>system\n{system_prompt}<|im_end|>\n<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        custom_completion_template="{completion}"
    else:  # Zephyr chat formatting
        # for models like zephyr/mistral we don't use the system prompt templating
        use_tokenizer_template = False
        custom_prompt_template = "<|user|>\n{prompt}</s> \n <|assistant|>\n"
        custom_completion_template="{completion}"
    
    print("Processing harmful dataset for LAT...")
    lat_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset=harmful_dataset,
        adv_column="rejected",
        def_column="chosen",
        split="train",
        use_tokenizer_template=use_tokenizer_template,
        system_prompt=system_prompt,
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template
    )

    lat_dataloader = DataLoader(
        lat_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=2048
        )
    )

    print("Processing benign dataset for supervised finetuning...")
    # interleaving supervised finetuning with LAT stabilizes training
    sft_dataset = process_generic_chat_dataset(
        tokenizer,
        dataset=benign_dataset,
        adv_column="refusal",
        def_column="response",
        split="train",
        use_tokenizer_template=use_tokenizer_template,
        system_prompt=system_prompt,
        custom_prompt_template=custom_prompt_template,
        custom_completion_template=custom_completion_template,
        add_eos_token=True
    )

    sft_dataloader = DataLoader(
        sft_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=LatentAdversarialTrainingDataCollator(
            tokenizer.pad_token_id,
            truncate_length=2048
        )
    )
    return lat_dataloader, sft_dataloader

def get_trainer(model, model_type, lat_dataloader, sft_dataloader, lat_config, project_dir):
    print("Setting up LAT trainer...")
    # Set the attack hyperparameters
    if model_type == 'llama2':  # use llama2-7b
        adv_loss_coefs = lat_config['adv_loss_coefs']
        def_loss_coefs = lat_config['def_loss_coefs']
        inner_learning_rate = 5e-2
        outer_learning_rate = 2e-5
        add_completions_pgd = False
    elif model_type == 'llama3': # use llama3-8b
        adv_loss_coefs = lat_config['adv_loss_coefs']
        def_loss_coefs = lat_config['def_loss_coefs']
        inner_learning_rate = 1e-3
        outer_learning_rate = 8e-5
        add_completions_pgd = True
    elif model_type == 'qwen3': # use qwen3-8b
        adv_loss_coefs = lat_config['adv_loss_coefs']
        def_loss_coefs = lat_config['def_loss_coefs']
        inner_learning_rate = 1e-3
        outer_learning_rate = 8e-5

        add_completions_pgd = True
    elif model_type == 'mistral':  # Mistral-7B-Instruct-v0.3
        adv_loss_coefs = lat_config['adv_loss_coefs']
        def_loss_coefs = lat_config['def_loss_coefs']
        inner_learning_rate = 1e-3
        outer_learning_rate = 8e-5
        add_completions_pgd = True
    elif model_type == 'olmo':  # OLMo-3-7B-Instruct
        adv_loss_coefs = lat_config['adv_loss_coefs']
        def_loss_coefs = lat_config['def_loss_coefs']
        inner_learning_rate = 1e-3
        outer_learning_rate = 8e-5
        add_completions_pgd = True
    else:
        raise Exception(f"No LAT hyperparameters configured for model_type '{model_type}'")

    print("Adversary loss coefs:", adv_loss_coefs)
    print("Defender loss coefs:", def_loss_coefs)

    pgd_trainer = ProjectedGradLAT(
        model=model,  # model
        dataloader=lat_dataloader,  # dataloader for lat
        sft_dataloader=sft_dataloader,  # dataloader for supervised finetuning
        adv_loss_coefs=adv_loss_coefs,  # adversary's loss coefs
        def_loss_coefs=def_loss_coefs,  # model's loss coefs
        pgd_layers=lat_config['pgd_layers'],  # what layers to attack
        pgd_iterations_per_step=lat_config['pgd_iterations_per_step'],  # how many steps of projected gradient descent to do
        model_layers=list(range(0, model.config.num_hidden_layers)),  # model layers to train
        epsilon=lat_config['epsilon'],  # attack l2 constraint
        inner_learning_rate=inner_learning_rate,  # adversary lr
        outer_learning_rate=outer_learning_rate,  # model lr
        model_iterations_per_step=lat_config['model_iterations_per_step'],  # how many times to train on each step
        num_steps=lat_config['num_steps'],  # number of epochs
        max_batch_per_acc=lat_config['max_batch_per_acc'],  # max size of a minibatch
        only_train_lora=True,  # train using low rank adapters
        l2_regularization=lat_config['l2_regularization'],  # coef for l2 weight regularization
        model_layers_module="base_model.model.model.layers",  # where the model layers are
        reinitialize_dev_optim=lat_config['reinitialize_dev_optim'],  # whether to reinitialize optimizer every lat step,
        add_completions_pgd=add_completions_pgd,  # Whether to add PGD over the completion tokens
        N_checkpoints=lat_config['N_checkpoints'],  # number of checkpoints to keep on disk
        checkpoint_dir=project_dir
    )

    return pgd_trainer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default='meta-llama/Meta-Llama-3-8B-Instruct')
    parser.add_argument("--benign_dataset", type=str, required=True)
    parser.add_argument("--harmful_dataset", type=str, required=True)
    parser.add_argument("--cache_dir", type=str, default=os.path.join(os.path.dirname(__file__), 'cache'))
    parser.add_argument("--system_prompt_path", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--project_name", type=str, default="lat_project")
    parser.add_argument("--eval", action='store_true')
    parser.add_argument("--eval_freq", type=int, default=1)
    parser.add_argument("--lat_config_path", type=str, default=os.path.join(os.path.dirname(__file__), 'lat_config.json'))
    parser.add_argument("--wandb-offline", action='store_true')
    parser.add_argument("--timestamp", type=str)
    
    # Optional arguments to override lat_config values
    parser.add_argument("--pgd_iterations_per_step", type=int, default=None)
    parser.add_argument("--model_iterations_per_step", type=int, default=None)
    parser.add_argument("--num_steps", type=int, default=None)
    parser.add_argument("--max_batch_per_acc", type=int, default=None)
    parser.add_argument("--l2_regularization", type=float, default=None)
    parser.add_argument("--reinitialize_dev_optim", type=bool, default=None)
    parser.add_argument("--N_checkpoints", type=int, default=None)
    parser.add_argument("--adv_toward", type=float, default=None, help="Adversarial loss coefficient for 'toward' term")
    parser.add_argument("--adv_away", type=float, default=None, help="Adversarial loss coefficient for 'away' term")
    parser.add_argument("--def_sft", type=float, default=None, help="Defender loss coefficient for 'sft' term")
    parser.add_argument("--def_toward", type=float, default=None, help="Defender loss coefficient for 'toward' term")
    parser.add_argument("--def_away", type=float, default=None, help="Defender loss coefficient for 'away' term")
    parser.add_argument("--pgd_layers", type=str, default=None, help="JSON string with layer indices, e.g. '[\"embedding\", 8, 16, 24, 30]'")
    parser.add_argument("--epsilon", type=float, default=None, help="L2 constraint for PGD attack")
    


    args = parser.parse_args()

    model_name = args.model_name
    benign_dataset = args.benign_dataset
    harmful_dataset = args.harmful_dataset
    cache_dir = args.cache_dir
    system_prompt_path = args.system_prompt_path
    batch_size = args.batch_size
    project_name = args.project_name
    lat_config_path = args.lat_config_path
    evaluate = args.eval
    wandb_offline = args.wandb_offline
    eval_freq = args.eval_freq
    timestamp = args.timestamp

    # Load system prompt from file if provided
    if system_prompt_path is not None:
        with open(system_prompt_path, 'r') as f:
            system_prompt = f.read().strip()
    else:
        print("No system prompt path provided, using empty system prompt.")
        system_prompt = ""

    model, tokenizer, model_type = load_model(model_name=model_name)

    print("Loading data...")
    lat_dataloader, sft_dataloader = load_data(
        benign_dataset=benign_dataset,
        harmful_dataset=harmful_dataset,
        tokenizer=tokenizer,
        model_type=model_type,
        system_prompt=system_prompt,
        batch_size=batch_size,
    )
    print("Data loaded.")
    
    # Load lat config defaults from JSON (can be overridden by CLI args)
    if os.path.exists(lat_config_path):
        with open(lat_config_path, 'r') as f:
            lat_config = json.load(f)
    else:
        raise Exception(f"LAT config file not found at {lat_config_path}")
    
    # Override lat_config with command-line arguments if provided
    if args.pgd_iterations_per_step is not None:
        lat_config['pgd_iterations_per_step'] = args.pgd_iterations_per_step
    if args.model_iterations_per_step is not None:
        lat_config['model_iterations_per_step'] = args.model_iterations_per_step
    if args.num_steps is not None:
        lat_config['num_steps'] = args.num_steps
    if args.max_batch_per_acc is not None:
        lat_config['max_batch_per_acc'] = args.max_batch_per_acc
    if args.l2_regularization is not None:
        lat_config['l2_regularization'] = args.l2_regularization
    if args.reinitialize_dev_optim is not None:
        lat_config['reinitialize_dev_optim'] = args.reinitialize_dev_optim
    if args.N_checkpoints is not None:
        lat_config['N_checkpoints'] = args.N_checkpoints
    
    # Override individual adv loss coefficients
    if args.adv_toward is not None:
        lat_config['adv_loss_coefs']['toward'] = args.adv_toward
    if args.adv_away is not None:
        lat_config['adv_loss_coefs']['away'] = args.adv_away
    
    # Override individual def loss coefficients
    if args.def_sft is not None:
        lat_config['def_loss_coefs']['sft'] = args.def_sft
    if args.def_toward is not None:
        lat_config['def_loss_coefs']['toward'] = args.def_toward
    if args.def_away is not None:
        lat_config['def_loss_coefs']['away'] = args.def_away
    if args.pgd_layers is not None:
        try:
            lat_config['pgd_layers'] = json.loads(args.pgd_layers)
        except json.JSONDecodeError as e:
            raise ValueError(f"Failed to parse pgd_layers JSON: {args.pgd_layers}") from e
    if args.epsilon is not None:
        lat_config['epsilon'] = args.epsilon

    peft_config = LoraConfig(
        r=64,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj"],
    )

    model = get_peft_model(model, peft_config)

    if evaluate:
        print("Loading ais/HarmBench-Llama-2-13b-cls from huggingface")
        try:
            # if that fails, load from huggingface
            cls = AutoModelForCausalLM.from_pretrained("cais/HarmBench-Llama-2-13b-cls",
                                                            dtype=torch.bfloat16,
                                                            device_map="cpu")
            cls_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf")
            cls_tokenizer.pad_token_id = cls_tokenizer.unk_token_id
            cls_tokenizer.padding_side = "left"
            print("HarmBench classifier loaded.")
        except Exception as e:
            print("Failed to load HarmBench classifier:", e)
            raise e
    else:
        cls = None
        cls_tokenizer = None

    if timestamp is None:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
    project_dir = cache_dir+"/"+project_name+"_"+timestamp
    os.makedirs(project_dir, exist_ok=True)

    trainer = get_trainer(
        model=model,
        model_type=model_type,
        lat_dataloader=lat_dataloader,
        sft_dataloader=sft_dataloader,
        lat_config=lat_config,
        project_dir=project_dir
    )

    # Attach callbacks to log all reported metrics to wandb each epoch
    def _to_number(x):
        try:
            return float(x)
        except Exception:
            try:
                return float(x.item())
            except Exception:
                return x

    # Define W&B metrics to use 'epoch' as the step for adv/* and def/*
    def _init_define_metrics(_, __):
        try:
            # set all metrics to use epochs as step
            wandb.define_metric("adv/*", step_metric="epoch")
            wandb.define_metric("def/*", step_metric="epoch")
            wandb.define_metric("harmbench/*", step_metric="epoch")
            wandb.define_metric("utility/*", step_metric="epoch")
        except Exception:
            # If define_metric fails (no active run), ignore — trainer.train will init W&B
            pass
        if False: #Disable evaluation before training (epoch 0)
            if trainer.checkpoint_dir is None:
                harmbench_cache_dir = "cache/harmbench_cache_0"
            else:
                harmbench_cache_dir = trainer.checkpoint_dir+"/harmbench_cache_0"
            harmbench_logs, utility_logs = evaluate_model(model=trainer.model,
                                                        tokenizer=tokenizer,
                                                        model_type=model_type,
                                                        cache_dir=harmbench_cache_dir,
                                                        cls=cls,
                                                        cls_tokenizer=cls_tokenizer)
            wandb.log(harmbench_logs, step=0)
            wandb.log(utility_logs, step=0)
        wandb.log({'epoch': 0}, step=0)

    def post_adv_callback(losses, epoch):
        if losses is None:
            return
        log = {}
        for k, v in losses.items():
            log[k] = _to_number(v)
        if epoch is not None:
            log["epoch"] = int(epoch)+1  # epoch is 0-indexed internally
        # Log with explicit step for consistent x-axis
        if epoch is not None:
            wandb.log(log, step=int(epoch)+1)

    def post_def_callback(losses, epoch):
        if losses is None:
            return
        log = {}
        for k, v in losses.items():
            log[k] = _to_number(v)
        log["epoch"] = int(epoch)+1  # epoch is 0-indexed internally
        if evaluate and ((epoch+1) % eval_freq == 0):
            if trainer.checkpoint_dir is None:
                harmbench_cache_dir = "cache/harmbench_cache"+str(int(epoch)+1)
            else:
                harmbench_cache_dir = trainer.checkpoint_dir+"/harmbench_cache_"+str(int(epoch)+1)
            harmbench_logs, utility_logs = evaluate_model(model=trainer.model,
                                                        tokenizer=tokenizer,
                                                        model_type=model_type,
                                                        cache_dir=harmbench_cache_dir,
                                                        cls=cls,
                                                        cls_tokenizer=cls_tokenizer)
            wandb.log(harmbench_logs, step=int(epoch)+1)
            wandb.log(utility_logs, step=int(epoch)+1)
        wandb.log(log, step=int(epoch)+1)

    trainer.init_callback = _init_define_metrics
    trainer.post_adv_callback = post_adv_callback
    trainer.post_def_callback = post_def_callback

    # Prepare run metadata to store in W&B config
    additional_wandb_kwargs = {
        "id": timestamp,
        "model_name": model_name,
        "benign_dataset": benign_dataset,
        "harmful_dataset": harmful_dataset,
        "system_prompt_path": system_prompt_path,
        "batch_size": batch_size,
        "lat_config": lat_config,
    }

    if wandb_offline:
        additional_wandb_kwargs["mode"] = "offline"

    with open(project_dir+'/parameters.json', 'w', encoding='utf-8') as f:
        json.dump(additional_wandb_kwargs, f, ensure_ascii=False, indent=2)
    print("Starting LAT training...")
    trainer.train(project_name=project_name, name=timestamp, additional_wandb_kwargs=additional_wandb_kwargs)
    trainer.model.save_pretrained(project_dir)

if __name__ == "__main__":
    main()