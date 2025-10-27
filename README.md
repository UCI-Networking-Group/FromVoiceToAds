# FromVoiceToAds
Code repository for the paper "From Voice to Ads: Auditing Commercial Smart Speakers for Targeted Advertising based on Voice Characteristics" published at ACM IMC 2025.

### Paper: TBA

### Datasets:
**ads_db.json:** the main ads dataset

**AmazonAudiences.json:** processed from the AmazonAudiences.csv provided by Amazon Data Request

**AdvertiserAudiences.json:** processed from the AdvertiserAudiences.csv provided by Amazon Data Request

These files can be found in their corresponding analysis folders.

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