import json
import re
from pathlib import Path

from Bio import SeqIO


def read_fasta(file_path, fused_lambda=None):
    sequences = []
    for record in SeqIO.parse(file_path, "fasta"):
        description = record.description
        seq_data = {
            "id": record.id,
            "sequence": str(record.seq),
            "name": str(record.name),
            "description": description,
            "fused_at": None,
        }

        if fused_lambda:
            try:
                seq_data["fused_at"] = fused_lambda(description)
            except ValueError:
                pass

        sequences.append(seq_data)
    return sequences


def real_extract_fp(input_string):
    return int(re.search(r"BP=(\d+)", input_string).group(1))


def sim_extract_fp(input_string):
    return int(re.search(r"FusedAt:(\d+)", input_string).group(1))


def save_fusions_to_json(data, file_path):
    with open(file_path, "w") as f:
        json.dump(data, f)


def load_fusions_from_json(file_path):
    with open(file_path, "r") as f:
        return json.load(f)


def load_fusions_from_fusionaitxt(file_path, fused_lambda=None):
    data = []
    with open(file_path, "r") as f:
        for line in f:
            columns = line.strip().split("\t")
            if len(columns) == 11:
                entry = {
                    "gene1": columns[0],
                    "chr1": columns[1],
                    "pos1": int(columns[2]),
                    "strand1": columns[3],
                    "gene2": columns[4],
                    "chr2": columns[5],
                    "pos2": int(columns[6]),
                    "strand2": columns[7],
                    "sequence1": columns[8],
                    "sequence2": columns[9],
                    "target": columns[10],
                }
                if "N" in entry["sequence1"] or "N" in entry["sequence2"]:
                    continue
                data.append(entry)
    return data


def read_fasta_files(files, fused_lambda=None):
    sequences = []
    for f in files:
        sequences.extend(read_fasta(f, fused_lambda))
    return sequences
