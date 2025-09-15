import os
import sys

current_working_dir = "/qumulo/shared_data/aofei_summer/RegTok/RegLLM"
sys.path.append(os.path.join(current_working_dir, ""))

from llava.train.train import train

if __name__ == "__main__":
    # train(attn_implementation="flash_attention_2")
    train()
