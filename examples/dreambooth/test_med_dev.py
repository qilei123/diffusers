from train_dreambooth import DreamBoothDataset4Med

def test_DreamBoothDataset4Med():
    dataset4med = eval("DreamBoothDataset4Med")("./","a photo of gastroscopy disease",None)

if __name__ == "__main__":
    test_DreamBoothDataset4Med()
    pass