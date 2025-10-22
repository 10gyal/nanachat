import time
import pytest
import os
from dotenv import load_dotenv

load_dotenv()

from nana_tokenizers import RegexTokenizer


def get_base_dir():
    # co-locate nanochat intermediates with other cached data in ~/.cache (by default)
    if os.environ.get("NANOCHAT_BASE_DIR"):
        nanochat_dir = os.environ.get("NANOCHAT_BASE_DIR")
    else:
        home_dir = os.path.expanduser("~")
        cache_dir = os.path.join(home_dir, ".cache")
        nanochat_dir = os.path.join(cache_dir, "nanochat")
    os.makedirs(nanochat_dir, exist_ok=True)
    return nanochat_dir


# Test the nana_tokenizers
@pytest.fixture(scope="module")
def enwik8_path():
    """Fixture to download and cache enwik8 dataset"""
    import os
    import zipfile

    base_directory = "."

    enwik8_url = "https://mattmahoney.net/dc/enwik8.zip"
    enwik8_local_path = os.path.join(base_directory, "enwik8")
    enwik8_local_path_zip = os.path.join(base_directory, "enwik8.zip")
    if not os.path.exists(enwik8_local_path):
        print(f"Downloading enwik8 to {enwik8_local_path_zip}")
        import requests

        response = requests.get(enwik8_url)
        with open(enwik8_local_path_zip, "wb") as f:
            f.write(response.content)
        with zipfile.ZipFile(enwik8_local_path_zip, "r") as zip_ref:
            zip_ref.extractall(base_directory)
        print(f"Unzipped enwik8 to {enwik8_local_path}")
        os.remove(enwik8_local_path_zip)
        print(f"Removed {enwik8_local_path_zip}")

    else:
        print(f"Using existing enwik8 at {enwik8_local_path}")
    return enwik8_local_path


@pytest.fixture(scope="module")
def enwik8_small(enwik8_path):
    """Fixture providing 100KB of enwik8 for quick tests."""
    with open(enwik8_path, "r") as f:
        return f.read(100_000)


@pytest.fixture(scope="module")
def enwik8_large(enwik8_path):
    """Fixture providing 10MB of enwik8 for performance tests."""
    with open(enwik8_path, "r") as f:
        return f.read(10**7)


def time_function(func, *args, **kwargs):
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    elapsed = end_time - start_time
    return result, elapsed


def test_bpe(enwik8_small):
    text = enwik8_small
    encode_text = text
    vocab_size = 256 + 20

    print("\nTraining bpe using Regex Tokenizer")
    regex_tokenizer = RegexTokenizer()
    ambiguous_flag, regex_tokenizer_train_time = time_function(
        regex_tokenizer.train, text, vocab_size
    )

    regex_tokenizer_ids, regex_tokenizer_encode_time = time_function(
        regex_tokenizer.encode_ordinary, encode_text
    )

    print(f"Regex Tokenizer train time: {regex_tokenizer_train_time:.4f}s")
    print(f"Regex Tokenizer encode time: {regex_tokenizer_encode_time:.4f}s")
    print("Encodings:", regex_tokenizer_ids[:40], "...")

    if ambiguous_flag:
        print(
            "‼️ WARNING: merge order was detected to be ambiguous given current text and vocab size"
        )
        print(
            "The implementation could be correct but we might see different results below"
        )
    else:
        print("✅ Merge order is NOT ambiguous")


if __name__ == "__main__":
    test_bpe()
