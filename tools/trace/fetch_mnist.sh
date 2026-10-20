#!/usr/bin/env bash
# Downloads MNIST (the four IDX files), checks their SHA-256, and writes the CSV files that data::MNISTDataset reads:
#   <dir>/mnist_train.csv, <dir>/mnist_test.csv   one sample per line: the label, then 784 pixels as integers 0..255
# MNISTDataset divides every pixel by 255 itself, so the CSV holds the raw values. (TorC's own datasets/mnist/convert_to_csv.py
# writes p/255, which MNISTDataset then divides again; see docs/library-audit/v0.1.0-findings.md.)
#   tools/trace/fetch_mnist.sh [dir]     default dir: build/mnist. Does nothing for files that are already there and correct.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
DIR="${1:-$ROOT/build/mnist}"
BASE="https://storage.googleapis.com/cvdf-datasets/mnist"
mkdir -p "$DIR"

declare -A SHA=(
  [train-images-idx3-ubyte.gz]=440fcabf73cc546fa21475e81ea370265605f56be210a4024d2ca8f203523609
  [train-labels-idx1-ubyte.gz]=3552534a0a558bbed6aed32b30c495cca23d567ec52cac8be1a0730e8010255c
  [t10k-images-idx3-ubyte.gz]=8d422c7b0a1c1c79245a5bcf07fe86e33eeafee792b84584aec276f5a2dbc4e6
  [t10k-labels-idx1-ubyte.gz]=f7ae60f92e00ec6debd23a6088c31dbd2371eca3ffa0defaefb259924204aec6
)
for f in "${!SHA[@]}"; do
  if [ ! -f "$DIR/$f" ] || ! echo "${SHA[$f]}  $DIR/$f" | sha256sum -c --quiet - 2>/dev/null; then
    curl -sSfL --retry 4 --max-time 300 -o "$DIR/$f" "$BASE/$f"
    echo "${SHA[$f]}  $DIR/$f" | sha256sum -c --quiet - || { echo "fetch_mnist: checksum mismatch for $f" >&2; exit 1; }
  fi
done

convert() {  # $1 images, $2 labels, $3 csv
  [ -f "$3" ] && [ "$3" -nt "$DIR/$1" ] && return 0
  python3 -I - "$DIR/$1" "$DIR/$2" "$3" <<'PY'
import gzip, struct, sys
img_path, lab_path, out_path = sys.argv[1:4]
with gzip.open(img_path, 'rb') as f:
    magic, n, rows, cols = struct.unpack('>IIII', f.read(16))
    assert magic == 2051, magic
    pixels = f.read(n * rows * cols)
with gzip.open(lab_path, 'rb') as f:
    magic, m = struct.unpack('>II', f.read(8))
    assert magic == 2049 and m == n, (magic, m, n)
    labels = f.read(m)
size = rows * cols
with open(out_path + '.tmp', 'w') as out:
    for i in range(n):
        out.write(str(labels[i]) + ',' + ','.join(map(str, pixels[i * size:(i + 1) * size])) + '\n')
import os
os.replace(out_path + '.tmp', out_path)
PY
}
convert train-images-idx3-ubyte.gz train-labels-idx1-ubyte.gz "$DIR/mnist_train.csv"
convert t10k-images-idx3-ubyte.gz t10k-labels-idx1-ubyte.gz "$DIR/mnist_test.csv"
