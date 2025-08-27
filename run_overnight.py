import time
import subprocess

subprocess.run(["python", "5_eng2hindi_bpe_trainer.py"])
time.sleep(30)
subprocess.run(["python", "6_eng2bengali_bpe_trainer.py"])
