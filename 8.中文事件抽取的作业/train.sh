CUDA_VISIBLE_DEVICES=1 python -u main.py --mode trigger --epochs 5
python -u transform.py
CUDA_VISIBLE_DEVICES=0 python -u main.py --mode argument --epochs 5
