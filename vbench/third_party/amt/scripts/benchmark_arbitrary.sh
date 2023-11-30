CFG=$1
CKPT=$2

python benchmarks/gopro.py -c $CFG -p $CKPT
python benchmarks/adobe240.py -c $CFG -p $CKPT