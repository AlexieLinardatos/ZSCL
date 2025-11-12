import signal, torch

def handle_signal(signum, frame):
    print("Signaled end, saving...")
    torch.save()