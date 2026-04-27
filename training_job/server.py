from fastapi import FastAPI
import subprocess, threading

app = FastAPI()

def run_training(max_steps):
    subprocess.run(["python3", "train.py", "--max-steps", str(max_steps)])

@app.post("/rerun")
def rerun(max_steps: int = 50):
    t = threading.Thread(target=run_training, args=(max_steps,))
    t.start()
    return {"status": "started"}

@app.get("/health")
def health():
    return {"status": "ok"}
