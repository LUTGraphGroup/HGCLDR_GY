import argparse
import os
import re
import pandas as pd


METRIC_PATTERN = re.compile(
    r"Test:\s*(?P<Epoch>\d+).*?AUC=(?P<AUC>[\d\.]+).*?AUPR=(?P<AUPR>[\d\.]+)"
    r".*?ACC=(?P<ACC>[\d\.]+).*?Precision=(?P<Precision>[\d\.]+)"
    r".*?Recall=(?P<Recall>[\d\.]+).*?F1=(?P<F1>[\d\.]+).*?MCC=(?P<MCC>[\d\.]+)"
)


def parse_log_file(file_path: str):
    data = {"FileName": os.path.basename(file_path)}
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    for line in lines[:25]:
        line = line.strip()
        if "=" not in line or line.startswith(("Test", "Train", "[")):
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        try:
            data[key] = float(value) if "." in value else int(value)
        except ValueError:
            data[key] = value

    for line in lines[-3:]:
        m = METRIC_PATTERN.search(line)
        if not m:
            continue
        for k, v in m.groupdict().items():
            data[k] = float(v)
    return data


def main():
    parser = argparse.ArgumentParser(description="Parse training log txt files to xlsx summary.")
    parser.add_argument("--input-dir", default="log-files", help="Directory containing .txt logs")
    parser.add_argument("--output", default="Result_Summary.xlsx", help="Output xlsx file name")
    args = parser.parse_args()

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Directory not found: {args.input_dir}")

    txt_files = []
    for root, _, files in os.walk(args.input_dir):
        for name in files:
            if name.endswith(".txt"):
                txt_files.append(os.path.join(root, name))

    if not txt_files:
        raise RuntimeError(f"No .txt files found in: {args.input_dir}")

    rows = [parse_log_file(path) for path in txt_files]
    df = pd.DataFrame(rows)

    priority_order = [
        "FileName", "dataset", "AUC", "AUPR", "F1", "ACC", "MCC",
        "Recall", "Precision", "lr", "network"
    ]
    cols = list(df.columns)
    ordered = [c for c in priority_order if c in cols]
    ordered.extend([c for c in cols if c not in ordered])
    df = df[ordered]

    output_path = os.path.join(args.input_dir, args.output)
    df.to_excel(output_path, index=False)
    print(f"Saved summary: {output_path}")
    print(f"Parsed files: {len(df)}")


if __name__ == "__main__":
    main()
