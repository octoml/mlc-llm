python3 tests/benchmark.py --dtype=float16 --num-warm-up=5  --num-measurements=20 --benchmark-mode=torch-eager --num-input-tokens=32 --num-output-tokens=32
python3 tests/benchmark.py --dtype=float16 --num-warm-up=5  --num-measurements=20 --benchmark-mode=torch-inductor --num-input-tokens=32 --num-output-tokens=32
python3 tests/benchmark.py --dtype=float16 --num-warm-up=5  --num-measurements=20 --benchmark-mode=tvm --num-input-tokens=32 --num-output-tokens=32

python3 tests/benchmark.py --dtype=float16 --num-warm-up=5  --num-measurements=20 --benchmark-mode=torch-eager --num-input-tokens=128 --num-output-tokens=128
python3 tests/benchmark.py --dtype=float16 --num-warm-up=5  --num-measurements=20 --benchmark-mode=torch-inductor --num-input-tokens=128 --num-output-tokens=128
python3 tests/benchmark.py --dtype=float16 --num-warm-up=5  --num-measurements=20 --benchmark-mode=tvm --num-input-tokens=128 --num-output-tokens=128

python3 tests/benchmark.py --dtype=float16 --num-warm-up=5  --num-measurements=20 --benchmark-mode=torch-eager --num-input-tokens=512 --num-output-tokens=512
python3 tests/benchmark.py --dtype=float16 --num-warm-up=5  --num-measurements=20 --benchmark-mode=torch-inductor --num-input-tokens=512 --num-output-tokens=512
python3 tests/benchmark.py --dtype=float16 --num-warm-up=5  --num-measurements=20 --benchmark-mode=tvm --num-input-tokens=512 --num-output-tokens=512
