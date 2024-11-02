"""
This script is used to extract the constraints from the parallel corpus.
It will split the given counterpart list into N parts and run the codes.counterpart.counterpart_agent module for each part.
The output will be saved in the output folder.
"""
import os


# the working directory is the root of the project
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_NAME = "gpt-4o-mini"
MODE = "rules"
COUNTERPART_DIR = f"./data/working_dir/counterpart/tensorflow_pytorch_{MODEL_NAME}_3_seeds/counterparts"
SAVE_DIR = f"./data/working_dir/constraints/{MODE}/tensorflow_pytorch_{MODEL_NAME}_3_seeds/"
LOG_DIR = f"./data/working_dir/constraints/{MODE}/tensorflow_pytorch_{MODEL_NAME}_3_seeds/logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
total_num = len(os.listdir(COUNTERPART_DIR))
N = 20
step = total_num // N
for i in range(N):
    api_key_file = f"./data/api_keys/api{i%10+1}.key"
    start = i * step
    end = (i+1) * step if i != N-1 else total_num
    print(f"start: {start}, end: {end}")
    os.system(f"nohup python -u -m codes.constraints.constraint_extractor --mode={MODE} "
              f"--counterpart_dir={COUNTERPART_DIR} --save_dir={SAVE_DIR} --i_start={start} "
              f"--i_end={end} --api_key_path={api_key_file} --model_name={MODEL_NAME} >> {LOG_DIR}/constraint_{i}.log 2>&1 &")

COUNTERPART_DIR = f"./data/working_dir/counterpart/pytorch_tensorflow_{MODEL_NAME}_3_seeds/counterparts"
SAVE_DIR = f"./data/working_dir/constraints/{MODE}/pytorch_tensorflow_{MODEL_NAME}_3_seeds/"
LOG_DIR = f"./data/working_dir/constraints/{MODE}/pytorch_tensorflow_{MODEL_NAME}_3_seeds/logs"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
total_num = len(os.listdir(COUNTERPART_DIR))
N = 20
step = total_num // N
for i in range(N):
    api_key_file = f"./data/api_keys/api{i%10+1}.key"
    start = i * step
    end = (i+1) * step if i != N-1 else total_num
    print(f"start: {start}, end: {end}")
    os.system(f"nohup python -u -m codes.constraints.constraint_extractor --mode={MODE} "
              f"--counterpart_dir={COUNTERPART_DIR} --save_dir={SAVE_DIR} --i_start={start} "
              f"--i_end={end} --api_key_path={api_key_file} --model_name={MODEL_NAME} >> {LOG_DIR}/constraint_{i}.log 2>&1 &")
