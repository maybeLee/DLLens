"""
This script is used to generate test inputs from the parallel corpus.
It will split the given counterpart list into N parts and run the codes.program_generator module for each part.
The output will be saved in the output folder.
"""
import os
from pathlib import Path
import time

os.chdir(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # the working directory is the root of the project
MODE="llm"
MODEL_NAME="gpt-4o-mini"
N = 3
TIME_LIMIT = 60*60*5 # 5 hours
NUM_PROGRAM = 1000000
API_SOURCE = "all"

COUNTERPART_DIR = f"./data/working_dir/counterpart/tensorflow_pytorch_{MODEL_NAME}_3_seeds/counterparts"
CONSTRAINTS_DIR = f"./data/working_dir/constraints/{MODE}/tensorflow_pytorch_{MODEL_NAME}_3_seeds/"
PROGRAM_DIR = Path(f"./data/working_dir/tests/tensorflow")
LOG_DIR = PROGRAM_DIR / "logs"
os.makedirs(PROGRAM_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
total_num = len(os.listdir(COUNTERPART_DIR))
step = total_num // N
for i in range(N):
    start = i * step
    end = (i + 1) * step if i != N - 1 else total_num
    print(f"start: {start}, end: {end}")
    os.system(
        f"nohup python -u -m codes.program_generator --counterpart_dir={COUNTERPART_DIR} --time_limit={TIME_LIMIT} --num_program={NUM_PROGRAM} "
        f"--constraints_dir={CONSTRAINTS_DIR} --program_dir={PROGRAM_DIR} --api_source={API_SOURCE} --i_start={start} "
        f"--i_end={end} >> {LOG_DIR}/program_{i}.log &")


COUNTERPART_DIR = f"./data/working_dir/counterpart/pytorch_tensorflow_{MODEL_NAME}_3_seeds/counterparts"
CONSTRAINTS_DIR = f"./data/working_dir/constraints/{MODE}/pytorch_tensorflow_{MODEL_NAME}_3_seeds/"
PROGRAM_DIR = Path(f"./data/working_dir/tests/tensorflow")
LOG_DIR = PROGRAM_DIR / "logs"
os.makedirs(PROGRAM_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
total_num = len(os.listdir(COUNTERPART_DIR))
step = total_num // N
for i in range(N):
    start = i * step
    end = (i + 1) * step if i != N - 1 else total_num
    print(f"start: {start}, end: {end}")
    os.system(
        f"nohup python -u -m codes.program_generator --counterpart_dir={COUNTERPART_DIR} --time_limit={TIME_LIMIT} --num_program={NUM_PROGRAM} "
        f"--constraints_dir={CONSTRAINTS_DIR} --program_dir={PROGRAM_DIR} --api_source={API_SOURCE} --i_start={start} "
        f"--i_end={end} >> {LOG_DIR}/program_{i}.log &")
