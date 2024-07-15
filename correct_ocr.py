import os
import torch
import json
import time
import argparse
import pandas as pd
from datetime import timedelta
from copy import deepcopy
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
)
from langchain.prompts import PromptTemplate
from utils import mkdir, set_logger, log_info
from llm.prompts import *
from llm.dataset import ReportDataset
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", type=str, default="v_0_2_2")
    parser.add_argument("--metadata", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--repo", type=str, required=True)
    parser.add_argument("--cohort", type=str, required=True)
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--log_path", type=str, required=True)
    parser.add_argument("--gpu", type=str, default="a100")

    args = parser.parse_args()
    meta_df = pd.read_csv(args.metadata, sep="\t")

    meta_df = meta_df[meta_df["project_ids"] == args.cohort]

    prompt = eval(args.prompt)

    OUT_PATH = f"{args.out_dir}/{args.cohort}/"
    LOG_PATH = f"{args.log_path}/"
    mkdir(LOG_PATH)
    set_logger(f"{LOG_PATH}/{args.cohort}-{args.repo.split('/')[-1]}.log")

    log_info(args)

    mkdir(f"{OUT_PATH}/jsons")
    mkdir(f"{OUT_PATH}/txts")

    repo = args.repo
    device = "cuda"

    # Initialise model and tokeniser
    config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    if "a100" in args.gpu.lower():
        gpu = "a100"
    else:
        gpu = "other"

    start = time.perf_counter()
    if gpu == "other":
        model = AutoModelForCausalLM.from_pretrained(
            repo, quantization_config=config, device_map="auto"
        )
    elif gpu == "a100":
        log_info("A100 gpu detected using flash attention")
        model = AutoModelForCausalLM.from_pretrained(
            repo,
            quantization_config=config,
            device_map="auto",
            attn_implementation="flash_attention_2",
        )
    tokenizer = AutoTokenizer.from_pretrained(
        repo, trust_remote_code=True, use_fast=True
    )
    end = time.perf_counter()
    log_info(f"Loaded model in {timedelta(seconds=end-start)}")

    # Set up dataset and loader
    dataset = ReportDataset(args.data_dir, tokenizer, cohort_df=meta_df)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=8,
    )

    log_info(f"{len(dataset)} files to parse")

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    for idx, batch in tqdm(enumerate(loader), total=len(loader)):
        pages, max_length, _, file_stem = batch
        if isinstance(file_stem, tuple):
            file_stem = file_stem[0]
        if os.path.exists(f"{OUT_PATH}/jsons/{file_stem}.json"):
            log_info(f"{file_stem} already done, skipping")
        else:
            log_info(f"Processing {file_stem}, {len(pages)} pages")

            def data(prompt=prompt):
                for i, (page_num, page) in enumerate(pages.items()):
                    prompt_ = deepcopy(prompt)
                    prompt_template = PromptTemplate.from_template(
                        template=prompt_[-1]["content"],
                    )
                    prompt_insert = prompt_template.format(page_text=page)
                    prompt_[-1]["content"] = prompt_insert
                    yield tokenizer.apply_chat_template(
                        prompt_, tokenize=False, return_tensors="pt"
                    )

            start = time.perf_counter()
            store_dict = {}
            for i, out in enumerate(
                pipe(
                    data(),
                    max_new_tokens=2 * max_length,
                    temperature=0.0,
                )
            ):
                # ! /INST is hardcoded for mistral rn
                llm_parsed = out[0]["generated_text"].split("[/INST]")[-1]
                if len(llm_parsed) > 2 * len(pages[f"page_{i+1}"][0]):
                    store_dict[f"page_{i+1}"] = pages[f"page_{i+1}"][0]
                elif len(llm_parsed) < 0.5 * len(pages[f"page_{i+1}"][0]):
                    store_dict[f"page_{i+1}"] = pages[f"page_{i+1}"][0]
                else:
                    store_dict[f"page_{i+1}"] = out[0]["generated_text"].split(
                        "[/INST]"
                    )[-1]

            end = time.perf_counter()
            log_info(f"Processed {file_stem} in {timedelta(seconds=end-start)}")

            final_text = "\n".join([x for x in store_dict.values()])

            # save json
            with open(f"{OUT_PATH}/jsons/{file_stem}.json", "w") as f:
                json.dump(store_dict, f, indent=4)

            # save txt
            with open(f"{OUT_PATH}/txts/{file_stem}.txt", "w") as f:
                f.write(final_text)

            log_info(f"Completed {idx + 1}/{len(loader)} files")
        # break
