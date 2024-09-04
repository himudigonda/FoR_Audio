# Synthetic Speech Detection using the Fake-or-Real (FoR) Dataset

## Project Overview

This project aims to develop a deep learning model to detect synthetic speech using the Fake-or-Real (FoR) dataset. The model distinguishes between real human speech and computer-generated speech, addressing the growing concern of deepfake audio. We focus on the for-norm version of the dataset.

## Dataset

The Fake-or-Real (FoR) dataset is a comprehensive collection of over 195,000 utterances, including both real human speech and computer-generated speech. It incorporates data from:

- Latest TTS solutions (e.g., Deep Voice 3, Google Wavenet TTS)
- Arctic Dataset
- LJSpeech Dataset
- VoxForge Dataset
- Custom speech recordings

### Dataset Version

This project uses the **for-norm** version of the dataset, which is balanced for gender and class, and normalized for sample rate, volume, and number of channels.

### Dataset Source

The dataset is downloaded from Kaggle:
[The Fake-or-Real Dataset (for-norm version)](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset?select=for-norm)

## Project Structure

```
audio_classification_project/
│
├── data/
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
│
├── models.py
├── dataloader.py
├── train.py
├── eval.py
├── utils.py
├── metrics.py
├── main.py
├── run.sh
├── requirements.txt
└── README.md
```

## Setup and Installation

1. Clone the repository:
   ```
   git clone https://github.com/himudigonda/FoR_Audio
   cd FoR_Audio
   ```

2. Create a virtual environment:
   ```
   mamba create -n FoR_Audio python==3.10
   mamba activate FoR_Audio
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Download the for-norm dataset from the Kaggle link provided above and place the audio files in the appropriate directories under `data/`.

## Usage

1. To train the model:
   ```
   python main.py
   ```

2. To evaluate the model:
   ```
   python eval.py
   ```

Alternatively, you can use the provided shell script to run both training and evaluation:
```
bash run.sh
```

## Model Architecture

The project uses a convolutional neural network (CNN) to classify mel spectrograms of audio samples. The architecture is defined in `models.py`.

## Results

(Add information about model performance, accuracy, and any notable findings once you have results)

## Contributing

Contributions to improve the model or extend the project are welcome. Please feel free to submit a Pull Request.

## License

(Add appropriate license information)

## Acknowledgements

- The Fake-or-Real (FoR) Dataset creators
- APTLY and LaSSoftE, Lassonde School of Engineering
- Kaggle for hosting the dataset


## Citations:
[1] https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset?select=for-norm

[2] https://www.kaggle.com/docs/datasets

[3] https://www.kaggle.com/discussions/general/156610

[4] https://www.youtube.com/watch?v=W86uvkzaqLg

[5] https://setronica.com/how-to-use-kaggle-datasets-for-research-a-step-by-step-guide/

[6] https://www.datacamp.com/tutorial/tutorial-kaggle-datasets-tutorials-kaggle-notebooks

[8] https://mostly.ai/blog/ml-classifier-for-fake-vs-real-data

[9] https://mostly.ai/video/fake-or-real-how-to-use-a-data-discriminator-for-evaluating-synthetic-data-quality
