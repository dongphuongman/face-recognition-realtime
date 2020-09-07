# READ ME

Face Recoginition is my project of the deep learning class, which is organized by AI For Everyone (AI4E). 

![img](https://i.imgur.com/fuLRu4B.png)

In this project, I have referred many face detect, face recognition model as [David Sandberg's facenet](https://github.com/davidsandberg/facenet), [InsightFace](https://github.com/deepinsight/insightface) and many other repos on github. Lastly, I end up with [Ultra-lightweight face detection model (ULFD)](https://github.com/Linzaer/Ultra-Light-Fast-Generic-Face-Detector-1MB) face detector and [face.evoLVe](https://github.com/ZhaoJ9014/face.evoLVe.PyTorch) face recognizer. 

Project has not 100% done yet.
TODO:
- [ ] Torch model serving
- [ ] Complete pipeline for production

## Installation

Use the package manager [pip](https://pip.pypa.io/en/stable/) to install package.

```bash
pip install -r requirement.txt
```

NOTE: You need GPU with cuda to run model in realtime

## Performance
Performance was evaluated on my laptop with AMD Ryzen 7 4800HS and NVIDIA GTX 1660TI Max-Q.

|Model|FPS|
| ------------- | ------------- |
|MTCNN + FaceNet + SVM|8 |
|ULFD + face.evoLVe + SVM| 45|


## Usage

Pre-trained model can be found [here](https://drive.google.com/drive/folders/18AJXgk4KyAf9sIDi--n_IMGjxZnKYuGv?usp=sharing).

To train a new SVM classifier: Use train_classifier.py. Your dataset should be like:
```
Dataset
  -> id1
    -> id1_1.jpg
    -> id1_2.jpg
    ...
  -> id2
    -> id1_1.jpg
    -> id1_2.jpg
    ...
  ...
```

To recognize:


```
python quickstart.py --path test.mp4
```

## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License
[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
