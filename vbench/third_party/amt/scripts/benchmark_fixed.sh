CFG=$1
CKPT=$2

python benchmarks/vimeo90k.py -c $CFG -p $CKPT
python benchmarks/ucf101.py -c $CFG -p $CKPT
python benchmarks/snu_film.py -c $CFG -p $CKPT
python benchmarks/xiph.py -c $CFG -p $CKPT