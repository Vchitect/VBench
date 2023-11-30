# Development for evaluation and training

- [Datasets](#Datasets)
- [Pretrained Models](#pretrained-models)
- [Evaluation](#evaluation)
- [Training](#training)

## Datasets<p id="Datasets"></p>
First, please prepare standard datasets for evaluation and training.

We present most of prevailing datasets in video frame interpolation, though some are not used in our project. Hope this collection could help your research. 

<table>
<thead>
  <tr>
    <th> Dataset </th>
    <th> :link: Source </th>
    <th> Train/Eval </th>
    <th> Arbitrary/Fixed </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>Vimeo90k</td>
    <th><a href="http://toflow.csail.mit.edu/">ToFlow (IJCV 2019)</a></th>
    <th>Both</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>ATD-12K</td>
    <th><a href="https://github.com/lisiyao21/AnimeInterp">AnimeInterp (CVPR 2021)</a></th>
    <th>Both</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>SNU-FILM</td>
    <th><a href="https://myungsub.github.io/CAIN/">CAIN (AAAI 2021)</a></th>
    <th>Eval</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>UCF101</td>
    <th><a href="https://drive.google.com/file/d/0B7EVK8r0v71pdHBNdXB6TE1wSTQ/view?resourcekey=0-r6ihCy20h3kbgZ3ZdimPiA">Google Driver</a></th>
    <th>Eval</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>HD</td>
    <th><a href="https://github.com/baowenbo/MEMC-Net">MEMC-Net (TPAMI 2018)</a>/<a href="https://github.com/baowenbo/MEMC-Net">Google Driver</a></th>
    <th>Eval</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>Xiph-2k/-4k</td>
    <th><a href="https://github.com/sniklaus/softmax-splatting/blob/master/benchmark_xiph.py">SoftSplat (CVPR 2020)</a></th>
    <th>Eval</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>MiddleBury</td>
    <th><a href="https://vision.middlebury.edu/flow/data/">MiddleBury</a></th>
    <th>Eval</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>GoPro</td>
    <th><a href="https://seungjunnah.github.io/Datasets/gopro">GoPro</a></th>
    <th>Both</th>
    <th>Arbitrary</th>
  </tr>
  <tr>
    <td>Adobe240fps</td>
    <th><a href="http://www.cs.ubc.ca/labs/imager/tr/2017/DeepVideoDeblurring">DBN (CVPR 2017)</a></th>
    <th>Both</th>
    <th>Arbitrary</th>
  </tr>
   <tr>
    <td>X4K1000FPS</td>
    <th><a href="https://github.com/JihyongOh/XVFI">XVFI (ICCV 2021)</a></th>
    <th>Both</th>
    <th>Arbitrary</th>
  </tr>
</tbody>
</table>


## Pretrained Models

<p id="Pretrained"></p>

<table>
<thead>
  <tr>
    <th> Dataset </th>
    <th> :link: Download Links </th>
    <th> Config file </th>
    <th> Trained on </th>
    <th> Arbitrary/Fixed </th>
  </tr>
</thead>
<tbody>
  <tr>
    <td>AMT-S</td>
    <th> [<a href="https://drive.google.com/file/d/1WmOKmQmd6pnLpID8EpUe-TddFpJuavrL/view?usp=share_link">Google Driver</a>][<a href="https://pan.baidu.com/s/1yGaNLeb9TG5-81t0skrOUA?pwd=f66n">Baidu Cloud</a>]</th>
    <th> [<a href="../cfgs/AMT-S.yaml">cfgs/AMT-S</a>] </th>
    <th>Vimeo90k</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>AMT-L</td>
    <th>[<a href="https://drive.google.com/file/d/1UyhYpAQLXMjFA55rlFZ0kdiSVTL7oU-z/view?usp=share_link">Google Driver</a>][<a href="https://pan.baidu.com/s/1qI4fBgS405Bd4Wn1R3Gbeg?pwd=nbne">Baidu Cloud</a>]</th>
    <th> [<a href="../cfgs/AMT-L.yaml">cfgs/AMT-L</a>] </th>
    <th>Vimeo90k</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>AMT-G</td>
    <th>[<a href="https://drive.google.com/file/d/1yieLtKh4ei3gOrLN1LhKSP_9157Q-mtP/view?usp=share_link">Google Driver</a>][<a href="https://pan.baidu.com/s/1AjmQVziQut1bXgQnDcDKvA?pwd=caf6">Baidu Cloud</a>]</th>
    <th> [<a href="../cfgs/AMT-G.yaml">cfgs/AMT-G</a>] </th>
    <th>Vimeo90k</th>
    <th>Fixed</th>
  </tr>
  <tr>
    <td>AMT-S</td>
    <th>[<a href="https://drive.google.com/file/d/1f1xAF0EDm-rjDdny8_aLyeedfM0QL4-C/view?usp=share_link">Google Driver</a>][<a href="https://pan.baidu.com/s/1eZtoULyduQM8AkXeYEBOEw?pwd=8hy3">Baidu Cloud</a>]</th>
    <th> [<a href="../cfgs/AMT-S_gopro.yaml">cfgs/AMT-S_gopro</a>] </th>
    <th>GoPro</th>
    <th>Arbitrary</th>
  </tr>
</tbody>
</table>

## Evaluation
Before evaluation, you should:

1. Check the dataroot is organized as follows:

```shell
./data
├── Adobe240
│   ├── original_high_fps_videos
│   └── test_frames # using ffmpeg to extract 240 fps frames from `original_high_fps_videos`
├── GOPRO
│   ├── test
│   └── train
├── SNU_FILM
│   ├── GOPRO_test
│   ├── test-easy.txt
│   ├── test-extreme.txt
│   ├── test-hard.txt
│   ├── test-medium.txt
│   └── YouTube_test
├── ucf101_interp_ours
│   ├── 1
│   ├── 1001
│   └── ...
└── vimeo_triplet
    ├── readme.txt
    ├── sequences
    ├── tri_testlist.txt
    └── tri_trainlist.txt
```

2. Download the provided [pretrained models](#pretrained-models).

Then, you can perform evaluation as follows:

+ Run all benchmarks for fixed-time models.

    ```shell
    sh ./scripts/benchmark_fixed.sh [CFG] [CKPT_PATH]
    ## e.g.
    sh ./scripts/benchmark_fixed.sh cfgs/AMT-S.yaml pretrained/amt-s.pth
    ```

+ Run all benchmarks for arbitrary-time models.

    ```shell
    sh ./scripts/benchmark_arbitrary.sh [CFG] [CKPT_PATH]
    ## e.g.
    sh ./scripts/benchmark_arbitrary.sh cfgs/AMT-S.yaml pretrained/gopro_amt-s.pth
    ```

+ Run a single benchmark for fixed-time models. *You can custom data paths in this case*.

    ```shell
    python [BENCHMARK] -c [CFG] -p [CKPT_PATH] -r [DATAROOT]
    ## e.g.
    python benchmarks/vimeo90k.py -c cfgs/AMT-S.yaml -p pretrained/amt-s.pth -r data/vimeo_triplet
    ```

+ Run the inference speed & model size comparisons using:

    ```shell
    python speed_parameters.py -c [CFG]
    ## e.g.
    python speed_parameters.py -c cfgs/AMT-S.yaml
    ```


## Training

Before training, please first prepare the optical flows (which are used for supervision).

We need to install `cupy` first before flow generation:

```shell
conda activate amt # satisfying `requirement.txt`
conda install -c conda-forge cupy
```


After installing `cupy`, we can generate optical flows by the following command:

```shell
python flow_generation/gen_flow.py -r [DATA_ROOT]
## e.g.
python flow_generation/gen_flow.py -r data/vimeo_triplet
```

After obtaining the optical flow of the training data,
run the following commands for training (DDP mode):

```shell
 sh ./scripts/train.sh [NUM_GPU] [CFG] [MASTER_PORT]
 ## e.g.
 sh ./scripts/train.sh 2 cfgs/AMT-S.yaml 14514
```

Our training configuration files are provided in [`cfgs`](../cfgs). Please carefully check the `dataset_dir` is suitable for you.


Note:

- If you intend to turn off DDP training, you can switch the key `distributed` from `true` 
to `false` in the config file.

- If you do not use wandb, you can switch the key `logger.use_wandb` from `true` 
to `false` in the config file.