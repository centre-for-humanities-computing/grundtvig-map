import subprocess
from pathlib import Path
from typing import Iterable, Union

from gensim.utils import tokenize
from glovpy import GloVe


def stream_sentences(files: Iterable[Union[str, Path]]) -> Iterable[list[str]]:
    for file in files:
        with open(file) as in_file:
            for line in in_file:
                yield list(tokenize(line, lower=True, deacc=True))


OUT_PATH = "models/grundtvig.kv"


def main():
    print("Fetching data...")
    subprocess.run(
        [
            "git",
            "clone",
            "https://github.com/centre-for-humanities-computing/grundtvig-data.git",
        ]
    )
    print("Collecting data.")
    data_path = Path("grundtvig-data/cleaned_data")
    file_paths = data_path.glob("*.txt")
    sentences = list(stream_sentences(file_paths))
    print("Training Word embeddings.")
    model = GloVe(vector_size=50)
    model.train(sentences)
    print("Saving embeddings.")
    out_file = Path(OUT_PATH)
    out_dir = out_file.parent
    out_dir.mkdir(exist_ok=True)
    model.wv.save(str(out_file))
    print("DONE")


if __name__ == "__main__":
    main()
