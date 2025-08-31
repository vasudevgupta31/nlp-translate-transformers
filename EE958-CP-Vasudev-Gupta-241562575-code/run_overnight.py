import time
import subprocess

subprocess.run(["python", "2_eng2hindi_trainer.py"])
time.sleep(30)
subprocess.run(["python", "3_eng2bengali_trainer.py"])
