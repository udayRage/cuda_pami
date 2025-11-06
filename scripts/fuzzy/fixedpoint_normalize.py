#!/usr/bin/env python3
import argparse
from pathlib import Path
from decimal import Decimal
import cudf
import rmm
import re

# Initialize Unified Memory for handling larger-than-GPU-memory files if needed
rmm.reinitialize(managed_memory=True)

def get_output_paths(input_path: Path) -> tuple[Path, Path]:
    """Generates standard _scaled.txt and _quant_mult.txt paths."""
    stem = input_path.stem
    # Clean up standard naming conventions if present
    if stem.endswith('_floating'):
        stem = stem[:-9]
    
    scaled_path = input_path.with_name(f"{stem}_scaled.txt")
    qm_path = input_path.with_name(f"{stem}_quant_mult.txt")
    return scaled_path, qm_path

def process_file(input_str: str, force: bool = False):
    input_path = Path(input_str)
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    scaled_path, qm_path = get_output_paths(input_path)

    if scaled_path.exists() and qm_path.exists() and not force:
        print(f"Output already exists: {scaled_path} (Use --force to overwrite)")
        return

    print(f"Processing {input_path.name}...")

    # Read file into GPU dataframe
    # Assumes format: item1\titem2:prob1\tprob2
    df = cudf.read_csv(
        input_path, sep=":", names=["items", "values"], 
        header=None, dtype=["str", "str"]
    ).fillna("")

    # sep_esc = re.escape(\t"\t")
    sep_esc = re.escape("\t")
    pattern = rf"[{sep_esc}\r\n ]+$"
    # gdf["items"] = gdf["items"].str.replace(pattern, "", regex=True)
    # gdf["values"] = gdf["values"].str.replace(pattern, "", regex=True)
    df["items"] = df["items"].str.replace(pattern, "", regex=True)
    df["values"] = df["values"].str.replace(pattern, "", regex=True)

    # Explode to long format to handle individual probabilities
    df_long = cudf.DataFrame({
        'item': df['items'].str.split('\t').explode(),
        'prob_str': df['values'].str.split('\t').explode(),
    })
    # print(df_long)


    # Determine scaling factor (10^max_decimals)
    decimals = df_long["prob_str"].str.partition(".")[2].str.len().fillna(0)
    max_sf = int(decimals.max())
    # print(f"Max decimal places found: {max_sf}")

    quant_mult = 10 ** max_sf

    # Perform scaling using vectorized string operations for precision
    parts = df_long["prob_str"].str.partition(".")
    int_part = parts[0].fillna("0").astype("int64")
    
    if max_sf > 0:
        # Right-pad fractional parts with zeros to match the scaling factor
        frac_part = parts[2].fillna("").str.slice(stop=max_sf).str.pad(width=max_sf, side='right', fillchar='0')
        scaled_probs = int_part * quant_mult + frac_part.astype("int64")
    else:
        scaled_probs = int_part

    df_long["prob_scaled"] = scaled_probs.astype("uint64")


    # Write outputs
    print(f"Scaling factor determined: {quant_mult} (10^{max_sf})")
    # qm_path.write_text(f"{quant_mult}\n", encoding='utf-8')

    file_output = input_path.with_name(f"{input_path.stem}_fixed_{quant_mult}.{input_path.suffix.lstrip('.')}")
    print(f"Writing fixed point file to {file_output}...")

    if file_output.exists():
        return str(file_output)

    lines = {}

    for row in df_long.to_pandas().itertuples():
        item = row.item
        prob_scaled = str(row.prob_scaled)
        index = row.Index
        # print(f"Index: {index}, Item: {item}, Scaled Prob: {prob_scaled}")
        # lines[index]
        if index not in lines:
            lines[index] = [[],[]]
        lines[index][0].append(item)
        lines[index][1].append(prob_scaled)

    with file_output.open('w', encoding='utf-8') as f:
        for index in sorted(lines.keys()):
            items = '\t'.join(lines[index][0])
            probs = '\t'.join(lines[index][1])
            f.write(f"{items}:{probs}\n")



    print(f"Completed: {file_output}")

    return file_output
    

def main():
    parser = argparse.ArgumentParser(description="GPU transaction scaling (No Parquet)")
    parser.add_argument('input_file', help='Path to floating point transaction file (items:probs)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing output files')
    args = parser.parse_args()

    process_file(args.input_file, force=args.force)

if __name__ == '__main__':
    main()