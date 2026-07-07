set -euo pipefail
python3 preprocess.py --input_dir data/raw --output_dir data/processed
python3 anomaly.py --input_dir data/processed --target your_target --modules your_modules