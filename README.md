# FromVoiceToAds
Code repository for the paper "From Voice to Ads: Auditing Commercial Smart Speakers for Targeted Advertising based on Voice Characteristics" published at ACM IMC 2025.

### Paper: [https://dl.acm.org/doi/10.1145/3730567.3764444](https://dl.acm.org/doi/10.1145/3730567.3764444)

**IMC 2025 Slides (No Narrations).mp4** in this repo shows animated presentation slides (No narrations available).

### Datasets:
These data files can be found in their corresponding analysis folders in this repo.

**ads_db.json:** the main ads dataset.
Note: `category` is the label assigned by the LLM. `category_tu` and `category_luca` are labels from two human labelers (Tu and Luca). `category_oracle` is the final ground truth label.

**AmazonAudiences.json:** processed from the AmazonAudiences.csv of each puppet provided by Amazon Data Request.

**AdvertiserAudiences.json:** processed from the AdvertiserAudiences.csv of each puppet provided by Amazon Data Request.

### Preparation:
1. Please use environmental variable or create `api_keys.py` locally to store your API keys with the following constants:
```python
OPENAI_API_KEY = "<your_openai_key>"
```

2. Apart from the other dependencies, please properly install torch and cuda on your PC (otherwise whisper will take forever to transcribe):
```console
pip uninstall -y torch torchvision torchaudio
# the following command was generated using https://pytorch.org/get-started/locally/#with-cuda-1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Contact: tle6@ua.edu (Tu Le)