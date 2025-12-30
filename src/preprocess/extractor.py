#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Process Images with Qwen3-VL Model
This script processes all image files under a folder and maintains the exact folder structure in the output.
"""

import torch
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor, AutoTokenizer
import os
import json
import argparse
import glob
from tqdm import tqdm
import logging
import sys
import time
import re
from json_repair import repair_json
from ollama import chat, ChatResponse
from pathlib import Path

class Qwen3VLModel:
    """
    A wrapper for the Qwen3-VL model that encapsulates model loading and inference.
    """
    def __init__(self, model_name="unsloth/Qwen3-VL-32B-Thinking-unsloth-bnb-4bit"):
        """
        Initializes and loads the Qwen3-VL model, processor, and tokenizer.
        """
        logging.info(f"Loading {model_name} model...")
        logging.info("This may take several minutes on first run (model download)...")
        try:
            self.model = Qwen3VLForConditionalGeneration.from_pretrained(
                model_name,
                dtype=torch.float16,
                attn_implementation="flash_attention_2",
                device_map="auto"
            )
            self.processor = AutoProcessor.from_pretrained(model_name)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.think_token_id = 151668  # Special token for separating thinking from response
            self.max_new_tokens = 6000 if 'Thinking' in model_name else 2048
            logging.info("✅ Model, Processor, and Tokenizer loaded successfully")
        except Exception as e:
            logging.error(f"❌ Error loading model: {e}")
            raise

    def invoke(self, image_path, instruction):
        """
        Performs inference on a single image with a given instruction.
        """
        try:
            logging.info(f"Processing image: {image_path}")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image_path},
                        {"type": "text", "text": instruction},
                    ],
                }
            ]

            inputs = self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt"
            ).to(self.model.device)

            logging.info("✅ Inputs prepared for inference")

            start_time = time.time()
            generated_ids = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"🚀 V-LLM inference took {duration:.2f} seconds.")
            output_ids = generated_ids[0][len(inputs.input_ids[0]):].tolist()
            num_new_tokens = len(output_ids)
            print(f"{num_new_tokens} new tokens generated.")

            try:
                index = len(output_ids) - output_ids[::-1].index(self.think_token_id)
            except ValueError:
                index = 0

            content = self.tokenizer.decode(output_ids[index:], skip_special_tokens=True).strip("\n")
            return content
        except Exception as e:
            logging.error(f"❌ Error during inference: {e}")
            return None

class Ollama:
    """
    A wrapper for a model served by Ollama.
    """
    def __init__(self, model_name="qwen3-vl"):
        """
        Initializes the Ollama wrapper.
        """
        logging.info(f"Using Ollama model: {model_name}")
        self.model_name = model_name
        # No model loading needed here, Ollama server handles it.
        logging.info("✅ Ollama model wrapper initialized successfully")

    def invoke(self, image_path, instruction):
        """
        Performs inference on a single image with a given instruction using Ollama.
        """
        try:
            logging.info(f"Processing image: {image_path} with Ollama model {self.model_name}")
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            start_time = time.time()
            response: ChatResponse = chat(
                model=self.model_name,
                messages=[
                    {
                        'role': 'user',
                        'content': instruction,
                        'images': [image_path]
                    },
                ])
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"🚀 Ollama inference took {duration:.2f} seconds.")

            content = response['message']['content']
            return content
        except Exception as e:
            logging.error(f"❌ Error during Ollama inference: {e}")
            return None

class ImageProcessor:
    """
    Encapsulates the logic for processing images with a Qwen3VLModel.
    """
    def __init__(self, qwen_model, model_name, prompt, retry=1):
        self.qwen_model = qwen_model
        self.model_name = model_name
        self.prompt = prompt
        self.retry = retry

    def _clean_llm_output(self, text):
        if text.startswith("```json"):
            text = text[7:]
        elif text.startswith("```"):
            text = text[3:]
        
        if text.endswith("```json"):
            text = text[:-7]
        elif text.endswith("```"):
            text = text[:-3]
        
        return text.strip()

    def process_single_image(self, image_path, output_path, instruction, force_mode='none'):
        """
        Process a single image file with a configurable number of retries and final repair logic.
        """
        if os.path.exists(output_path):
            if force_mode == 'hard':
                logging.info(f"🔄 Force reprocessing {image_path} as --hard-force is enabled.")
            elif force_mode == 'soft':
                try:
                    with open(output_path, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                    if data.get('extraction_status') == 'SUCCESS':
                        logging.info(f"⏭️  Skipping {image_path} as it was already successful.")
                        return True
                    else:
                        logging.info(f"🔄 Reprocessing {image_path} due to non-SUCCESS status ({data.get('extraction_status')}).")
                except (json.JSONDecodeError, IOError) as e:
                    logging.warning(f"⚠️ Could not read existing file {output_path} to check status. Reprocessing. Error: {e}")
            else: # Default behavior
                logging.info(f"⏭️  Skipping {image_path} as output file already exists. Use --force or --soft-force to overwrite.")
                return True

        logging.info(f"\nProcessing image: {image_path}")

        extraction_result_json = None
        extraction_status = "FAILED"
        parsing_error = None
        final_raw_output = ""

        for attempt in range(1, self.retry + 1):
            logging.info(f"Running LLM Invocation (Attempt {attempt}/{self.retry})...")
            raw_result = self.qwen_model.invoke(image_path, instruction)
            
            if raw_result:
                final_raw_output = self._clean_llm_output(raw_result)
                try:
                    extraction_result_json = json.loads(final_raw_output)
                    if attempt == 1:
                        extraction_status = "SUCCESS"
                        logging.info("✅ JSON parsed successfully on the first attempt.")
                    else:
                        extraction_status = "REINVOKE_SUCCESS"
                        logging.info(f"✅ JSON parsed successfully on attempt {attempt}.")
                    parsing_error = None
                    break  # Exit loop on success
                except json.JSONDecodeError as e:
                    logging.warning(f"⚠️ JSON parsing failed on attempt {attempt}. Error: {e}")
                    parsing_error = str(e)
                    if attempt < self.retry:
                        continue # Go to the next attempt
                    
                    # This is the last attempt, so now we try to repair
                    logging.warning(f"⚠️ All {self.retry} LLM invocations failed to produce valid JSON. Attempting to repair final output...")
                    try:
                        repaired_result = repair_json(final_raw_output)
                        extraction_result_json = json.loads(repaired_result)
                        extraction_status = "REPAIRED"
                        parsing_error = None
                        logging.info("✅ JSON successfully repaired.")
                    except Exception as e2:
                        logging.error(f"❌ JSON repair also failed. Final parsing error: {e2}")
                        extraction_result_json = final_raw_output
                        parsing_error = str(e2)
                        extraction_status = "FAILED"
            else:
                logging.error(f"❌ LLM invocation failed on attempt {attempt}.")
                parsing_error = f"LLM invocation returned no result on attempt {attempt}."
                final_raw_output = ""
                if attempt < self.retry:
                    continue
                else:
                    extraction_status = "FAILED"
                    extraction_result_json = ""
                    break

        output_data = {
            "image_path": Path(image_path).as_posix(),
            "model": self.model_name,
            "extraction_status": extraction_status,
            "prompt": self.prompt,
            "extraction_result": extraction_result_json
        }
        
        if parsing_error:
            output_data["parsing_error"] = parsing_error
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        
        logging.info(f"✅ Result saved to: {output_path}")
        return True

    def process_images_in_folder(self, input_path, output_path, instruction, use_tqdm=False, force_mode='none'):
        """
        Process all image files in a folder, maintaining folder structure
        """
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
        image_files = []
        for extension in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_path, '**', extension), recursive=True))
        
        logging.info(f"Found {len(image_files)} image files to process")
        
        success_count = 0
        set_console_logging(not use_tqdm)

        iterable = tqdm(image_files, desc="Processing images") if use_tqdm else image_files
        
        for image_file in iterable:
            relative_path = os.path.relpath(image_file, input_path)
            json_filename = os.path.splitext(relative_path)[0] + '.json'
            output_file = os.path.join(output_path, json_filename)
            
            if self.process_single_image(image_file, output_file, instruction, force_mode=force_mode):
                success_count += 1
        
        logging.info(f"\n✅ Processing completed! {success_count}/{len(image_files)} images processed successfully.")
        if use_tqdm:
            tqdm.write(f"\n✅ Processing completed! {success_count}/{len(image_files)} images processed successfully.")

def setup_logging(log_file="extract_json_from_pdf.log"):
    """
    Set up logging configuration with timestamp format
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
    file_handler = logging.FileHandler(log_file, encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)
    return logger

def set_console_logging(enabled=True):
    """
    Enable or disable console logging
    """
    logger = logging.getLogger()
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.INFO if enabled else logging.CRITICAL)

def main():
    parser = argparse.ArgumentParser(description="Process Images with Qwen3-VL Model")
    parser.add_argument('-m', '--model_name', type=str, default="unsloth/Qwen3-VL-32B-Instruct-unsloth-bnb-4bit",
                        help='The name of the model to use.')
    parser.add_argument('-i', '--input_path', type=str, required=True,
                        help='The path to the input image file or folder.')
    parser.add_argument('-o', '--output_path', type=str, required=True,
                        help='The path to save the output JSON files.')
    parser.add_argument('--prompt', type=str, required=True,
                        help='Path to a text file (utf-8) containing the prompt.')
    parser.add_argument('--tqdm', action='store_true',
                        help='Use tqdm progress bar instead of print statements')
    parser.add_argument('--force', type=str, default='soft', choices=['none', 'hard', 'soft'],
                        help="""Reprocessing strategy. 
                        'none': skip if output exists (default). 
                        'soft': reprocess if status is not SUCCESS. 
                        'hard': reprocess all.""")
    parser.add_argument('--retry', type=int, default=1,
                        help='Number of retry attempts if V-LLM invocation fails (default: 1)')
    parser.add_argument('--log-file', type=str, default="extract_json_from_pdf.log",
                        help='Path to log file (default: extract_json_from_pdf.log)')
    parser.add_argument('--ollama', action='store_true',
                        help='Use a model from Ollama instead of a local Hugging Face model.')
    args = parser.parse_args()
    
    setup_logging(args.log_file)
    
    logging.info(f"Qwen3-VL Image Processing")
    logging.info("=" * 35)
    
    try:
        with open(args.prompt, 'r', encoding='utf-8') as f:
            instruction = f.read()
    except FileNotFoundError:
        logging.error(f"❌ Prompt file not found at: {args.prompt}")
        return
    except Exception as e:
        logging.error(f"❌ Error reading prompt file: {e}")
        return
    
    logging.info(f"1. Loading {args.model_name} model...")
    try:
        if args.ollama:
            qwen_model = Ollama(model_name=args.model_name)
        else:
            qwen_model = Qwen3VLModel(model_name=args.model_name)
    except Exception as e:
        logging.error("❌ Failed to load model. Exiting.")
        return
    
    processor = ImageProcessor(qwen_model, args.model_name, args.prompt, retry=args.retry)

    if os.path.isfile(args.input_path):
        logging.info(f"\n2. Processing single image file...")
        
        if os.path.isdir(args.output_path):
            output_filename = os.path.splitext(os.path.basename(args.input_path))[0] + '.json'
            output_file = os.path.join(args.output_path, output_filename)
        elif os.path.isfile(args.output_path) or not os.path.exists(args.output_path):
            output_file = args.output_path
        else:
            logging.error(f"❌ Invalid output path: {args.output_path}")
            return
            
        processor.process_single_image(
            args.input_path, 
            output_file, 
            instruction, 
            force_mode=args.force
        )
    elif os.path.isdir(args.input_path):
        if os.path.isfile(args.output_path):
            logging.error(f"❌ Error: Input is a folder but output is a file. Please specify an output directory.")
            return
        logging.info(f"\n2. Processing all images in folder...")
        processor.process_images_in_folder(
            args.input_path, 
            args.output_path, 
            instruction, 
            use_tqdm=args.tqdm, 
            force_mode=args.force
        )
    else:
        logging.error(f"❌ Input path {args.input_path} is neither a file nor a directory.")
        return

if __name__ == "__main__":
    main()
