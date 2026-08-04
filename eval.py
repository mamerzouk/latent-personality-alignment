import os
import argparse
import wandb
import torch
import json
from transformers import AutoModelForCausalLM, AutoTokenizer
from tasks.harmbench.FastHarmBenchEvals import run_attack_evals, run_general_evals

from peft import PeftModel


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project_name", type=str, default="latent-personality-alignment")
    parser.add_argument("--run_id", type=str, required=True)
    parser.add_argument("--epoch", type=str, default=None)


    args = parser.parse_args()

    project_name = args.project_name
    run_id = args.run_id
    epoch = args.epoch
    project_path = "cache/"+project_name+"_"+run_id
    
    if os.path.exists(project_path):
        with open(os.path.join(project_path, "parameters.json"), "r") as f:
            parameters = json.load(f)
    else:
        raise Exception(f"Project path {project_path} does not exist. Please check the project name and run ID.")
    

    model_name = parameters["model_name"]

    if epoch is not None:
        project_path += "/checkpoint_"+epoch

    model_dtype = torch.bfloat16

    print(f"Loading model {model_name}...")
    base_model = AutoModelForCausalLM.from_pretrained(
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


    model = PeftModel.from_pretrained(base_model,
                                      project_path,
                                      device_map="auto"
                                     )


    print("Running HarmBench evaluations...")
    model.eval()
    with torch.no_grad():
        harmbench_asr = run_attack_evals(model=model,
                                            tokenizer=tokenizer,
                                            model_type=model_type,
                                            pretrained_cls="llama",
                                            do_sample=False,
                                            move_cls_device=True,
                                            move_model_device=True,
                                            cache_dir=project_path+"/eval")
        
        harmbench_logs = {f"harmbench/{k}": v for k, v in harmbench_asr.items()}
        print("Running utility evaluations...")
        utility_acc = run_general_evals(model=model,
                                        tokenizer=tokenizer,
                                        model_type=model_type,
                                        evals_to_include=["MMLU", "HellaSwag", "Winogrande", "SciQ", "Lambada"])
        utility_logs = ({f"utility/{k}": v for k, v in utility_acc.items()})


    wandb.init(
        project=project_name,
        id=run_id,
        resume="allow"
    )

    if epoch is not None:
        wandb.define_metric("epoch")
        wandb.define_metric("harmbench/*", step_metric="epoch", step_sync=True)
        wandb.define_metric("utility/*", step_metric="epoch", step_sync=True)
        log_dict = {"epoch": int(epoch)}
    else:
        log_dict = {}        
        wandb.define_metric("harmbench/*")
        wandb.define_metric("utility/*")

    log_dict.update(harmbench_logs)
    log_dict.update(utility_logs)
    
    wandb.log(log_dict)
    wandb.finish()


if __name__ == "__main__":
    main()