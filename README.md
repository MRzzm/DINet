# DINet: Deformation Inpainting Network for Realistic Face Visually Dubbing on High Resolution Video (AAAI2023)
![在这里插入图片描述](https://img-blog.csdnimg.cn/178c6b3ec0074af7a2dcc9ef26450e75.png)
[Paper](https://fuxivirtualhuman.github.io/pdf/AAAI2023_FaceDubbing.pdf) &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;     [demo video](https://www.youtube.com/watch?v=UU344T-9h7M&t=6s)  &nbsp;&nbsp;&nbsp;&nbsp; Supplementary materials

## Inference
##### Download resources (asserts.zip) in [Google drive](https://drive.google.com/drive/folders/1rPtOo9Uuhc59YfFVv4gBmkh0_oG0nCQb?usp=share_link). unzip and put dir in ./.
+  Inference with example videos.  Run 
  ```python 
python inference.py --mouth_region_size=256 --source_video_path=./asserts/examples/testxxx.mp4 --source_openface_landmark_path=./asserts/examples/testxxx.csv --driving_audio_path=./asserts/examples/driving_audio_xxx.wav --pretrained_clip_DINet_path=./asserts/clip_training_DINet_256mouth.pth  
```
The results are saved in ./asserts/inference_result

+  Inference with custom videos.  
**Note:** The released pretrained model is trained on HDTF dataset with 363 training videos (video names are in ./asserts/training_video_name.txt), so the generalization is limited. It would be better to test custom videos with normal lighting, frontal view etc.(see the limitation section in the paper).  **We also release the training code**, so if a larger high resolution audio-visual dataset is proposed in the further, you can use the training code to train a model with greater generalization. Besides, we release coarse-to-fine training strategy, **so you can use the training code to train a model in arbitrary resolution** (larger than 416x320 if gpu memory and training dataset are available).

Using [openface](https://github.com/TadasBaltrusaitis/OpenFace) to detect smooth facial landmarks of your custom video. We run the **OpenFaceOffline.exe** on windows 10 system with this setting:
  
| Record | Recording settings |  OpenFace setting | View | Face Detector | Landmark Detector |
|--|--|--|--|--|--|
| 2D landmark & tracked videos | Mask aligned image | Use dynamic AU models | Show video  | Openface (MTCNN)| CE-CLM |

The detected facial landmarks are saved in "xxxx.csv". Run 
  ```python 
python inference.py --mouth_region_size=256 --source_video_path= custom video path --source_openface_landmark_path=  detected landmark path --driving_audio_path= driving audio path --pretrained_clip_DINet_path=./asserts/clip_training_DINet_256mouth.pth  
```
to realize face visually dubbing on your custom videos.
## Training
### Data Processing
We release the code of video processing on [HDTF dataset](https://github.com/MRzzm/HDTF). You can also use this code to process custom videos.

 1. Downloading videos from [HDTF dataset](https://github.com/MRzzm/HDTF). Splitting videos according to xx_annotion_time.txt and **do not** crop&resize videos.
 2. Resampling all split videos into **25fps** and put videos into "./asserts/split_video_25fps". You can see the two example videos in "./asserts/split_video_25fps". We use [software](http://www.pcfreetime.com/formatfactory/cn/index.html) to resample videos. We provide the name list of training videos in  our experiment. (pls see "./asserts/training_video_name.txt")
 3. Using [openface](https://github.com/TadasBaltrusaitis/OpenFace) to detect smooth facial landmarks of all videos. Putting all ".csv" results into "./asserts/split_video_25fps_landmark_openface". You can see the two example csv files in "./asserts/split_video_25fps_landmark_openface".

 4. Extracting frames from all videos and saving frames in "./asserts/split_video_25fps_frame". Run 
```python 
python data_processing.py --extract_video_frame
```
 5. Extracting audios from all videos and saving audios in "./asserts/split_video_25fps_audio". Run 
 ```python 
python data_processing.py --extract_audio
```
 6. Extracting deepspeech features from all audios and saving features in "./asserts/split_video_25fps_deepspeech". Run 
  ```python 
python data_processing.py --extract_deep_speech
```
 7.  Cropping faces from all videos and saving images in "./asserts/split_video_25fps_crop_face". Run
   ```python 
python data_processing.py --crop_face
```
 8. Generating training json file "./asserts/training_json.json". Run
   ```python 
python data_processing.py --generate_training_json
```

### Training models
We split the training process into **frame training stage** and **clip training stage**. In frame training stage, we use coarse-to-fine strategy, **so you can train the model in arbitrary resolution**.

#### Frame training stage.
In frame training stage, we only use perception loss and GAN loss.

 1. Firstly, train the DINet in 104x80 (mouth region is 64x64) resolution. Run 
   ```python 
python train_DINet_frame.py --augment_num=32 --mouth_region_size=64 --batch_size=24 --result_path=./asserts/training_model_weight/frame_training_64
```
You can stop the training when the loss converges (we stop in about 270 epoch).

 2. Loading the pretrained model (face:104x80 & mouth:64x64) and train the DINet in higher resolution (face:208x160 & mouth:128x128). Run
   ```python 
python train_DINet_frame.py --augment_num=100 --mouth_region_size=128 --batch_size=80 --coarse2fine --coarse_model_path=./asserts/training_model_weight/frame_training_64/xxxxxx.pth --result_path=./asserts/training_model_weight/frame_training_128
```
You can stop the training when the loss converges (we stop in about 200 epoch).

 3. Loading the pretrained model (face:208x160 & mouth:128x128) and train the DINet in higher resolution (face:416x320 & mouth:256x256). Run
   ```python 
python train_DINet_frame.py --augment_num=20 --mouth_region_size=256 --batch_size=12 --coarse2fine --coarse_model_path=./asserts/training_model_weight/frame_training_128/xxxxxx.pth --result_path=./asserts/training_model_weight/frame_training_256
```
You can stop the training when the loss converges (we stop in about 200 epoch).

#### Clip training stage.
In clip training stage, we use perception loss, frame/clip GAN loss and sync loss. Loading the pretrained frame model (face:416x320 & mouth:256x256), pretrained syncnet model (mouth:256x256) and train the DINet in clip setting. Run
   ```python 
python train_DINet_clip.py --augment_num=3 --mouth_region_size=256 --batch_size=3 --pretrained_syncnet_path=./asserts/syncnet_256mouth.pth --pretrained_frame_DINet_path=./asserts/training_model_weight/frame_training_256/xxxxx.pth --result_path=./asserts/training_model_weight/clip_training_256
```
You can stop the training when the loss converges and select the best model (our best model is at 160 epoch).

## Acknowledge
The AdaAT is borrowed from [AdaAT](https://github.com/MRzzm/AdaAT). The deepspeech feature is borrowed from [AD-NeRF](https://github.com/YudongGuo/AD-NeRF). The basic module is borrowed from [first-order](https://github.com/AliaksandrSiarohin/first-order-model). Thanks for their released code.