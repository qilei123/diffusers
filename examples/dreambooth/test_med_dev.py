from train_dreambooth import DreamBoothDataset4Med
from meddatasets import load_test_data

def test_DreamBoothDataset4Med():
    dataset4med = eval("DreamBoothDataset4Med")("datasets","a photo of gastroscopy disease",None)
    dataset4med[0]


def test_load_test_data():
    image,mask = load_test_data()
    image.save("temp_img.png")
    mask.convert("L").save("temp_msk.png")

if __name__ == "__main__":
    #test_DreamBoothDataset4Med()
    test_load_test_data()
    pass