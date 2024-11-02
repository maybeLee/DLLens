"""
This script is used to collect counterparts for the given source and destination library.
"""
import os
from codes.counterpart.counterpart_agent import load_apis


def run(src, dst):
    WORKING_DIR = os.path.join(SAVE_DIR, f"{src}_{dst}_{MODEL_NAME}_{NUM_SEED}_seeds")
    os.makedirs(WORKING_DIR, exist_ok=True)
    LOG_DIR = os.path.join(WORKING_DIR, "logs")
    os.makedirs(LOG_DIR, exist_ok=True)
    api_sig_list = load_apis(package_name="", library_name=src)
    total_num = len(api_sig_list)
    N = 20
    step = total_num // N
    api_dir = "data/api_keys"
    for i in range(N):
        LOG_PATH = os.path.join(LOG_DIR, f"progress_{i}.log")
        api_key_file = f"{api_dir}/api{i % 10 + 1}.key"
        start = i * step
        end = (i + 1) * step if i != N - 1 else total_num
        print(f"start: {start}, end: {end}")
        os.system(f"nohup python -u -m codes.counterpart.counterpart_agent --source_lib={src} --dest_lib={dst} "
                  f"--api_key_path={api_key_file} --save_dir={SAVE_DIR} --i_start={start} "
                  f"--i_end={end} --num_seed={NUM_SEED} --model_name={MODEL_NAME} >> {LOG_PATH} 2>&1 &")


if __name__ == "__main__":
    SAVE_DIR = "data/working_dir/counterpart/"
    NUM_SEED = 3
    MODEL_NAME = "gpt-4o-mini"
    lib1, lib2 = "tensorflow", "pytorch"
    run(lib1, lib2)  # tf -> torch
    run(lib2, lib1)  # torch -> tf
