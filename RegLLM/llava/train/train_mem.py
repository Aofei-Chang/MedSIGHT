import os
import sys

current_working_dir = "/home/avc6555/research/MedSight/RegTok/RegLLM"
sys.path.append(os.path.join(current_working_dir, ""))

from llava.train.train_seg import train

if __name__ == "__main__":
    # train(attn_implementation="flash_attention_2")
    train()
