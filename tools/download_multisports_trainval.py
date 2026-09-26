from huggingface_hub import hf_hub_download


FILES = [
    "MultiSports.py",
    "README.md",
    "data/trainval/generate_rgb.py",
    "data/trainval/multisports_GT.pkl",
    "data/trainval/aerobic_gymnastics.tar",
    "data/trainval/basketball.tar",
    "data/trainval/football.tar",
    "data/trainval/volleyball.tar",
]


def main():
    for filename in FILES:
        print(f"START {filename}", flush=True)
        path = hf_hub_download(
            repo_id="MCG-NJU/SportsAction",
            repo_type="dataset",
            filename=filename,
            local_dir="data/multisports/hf",
            resume_download=True,
        )
        print(f"DONE {filename} {path}", flush=True)
    print("ALL_DONE", flush=True)


if __name__ == "__main__":
    main()
