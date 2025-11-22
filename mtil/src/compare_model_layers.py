import argparse
import torch

def main(args):

    ckpt1 = torch.load(args.model_path[0], map_location="cpu")
    ckpt2 = torch.load(args.model_path[1], map_location="cpu")

    ckpt1 = ckpt1["state_dict"]
    ckpt2 = ckpt2["state_dict"]

    diffs = []

    for name, param1 in ckpt1.items():
        param2 = ckpt2[name]

        diff = (param1-param2).abs()
        
        mean_diff = diff.mean().item()

        diffs.append((mean_diff,name))
        
        max_diff = diff.max().item()
        l2_norm = torch.norm(param1 - param2).item()
        print(f"{name}: mean={mean_diff:.6f}, max={max_diff:.6f}, L2={l2_norm:.6f}")

    diffs.sort(reverse=True)

    print("Sorted", "-"*60)
    for diff, name in diffs:
        print(f"{name}: mean={diff:.6f}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", nargs=2, type=str, required=True)

    args = parser.parse_args()
    main(args)