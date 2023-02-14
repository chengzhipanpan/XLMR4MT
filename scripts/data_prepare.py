import os, random, tqdm
from pathlib import Path
# merge two txt file one line by one line
from multiprocessing import Pool


def map_function(text):
    return text.strip()


def load_text_file(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    pool = Pool(4)
    lines = pool.map(map_function, lines)
    return lines


def merge_text_files():
    HOME = Path("/mnt/petrelfs/xujingjing/xujingjing/ted/de_en")
    en_path = HOME / "train.en"
    de_path = HOME / "train.de"
    en_lines = load_text_file(en_path)
    de_lines = load_text_file(de_path)
    assert len(en_lines) == len(de_lines)
    merged_path = HOME / "train.merged"
    merged_lines = []
    for index in tqdm.tqdm(range(len(en_lines))):
        merged_lines.append(de_lines[index] + "<extra_id_0>" + en_lines[index] + "\n")
    random.shuffle(merged_lines)
    with open(merged_path, 'w') as f:
        f.writelines(merged_lines)
    print("merged_file saved in: ", merged_path)


if __name__ == "__main__":
    merge_text_files()