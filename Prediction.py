import sys
sys.path.append('../cocoapi/PythonAPI')

from pycocotools.coco import COCO
from data_loader import get_loader
from torchvision import transforms
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
import os
import torch
from model import EncoderCNN, DecoderRNN


class Prediction():

    def __init__(self):
        self.transform_test = transforms.Compose([
            transforms.Resize(256),  # smaller edge of image resized to 256
            transforms.RandomCrop(224),  # get 224x224 crop from random location
            # transforms.ColorJitter(brightness=0.2, contrast=0, saturation=0, hue=0),
            transforms.RandomHorizontalFlip(),  # horizontally flip image with probability=0.5
            transforms.ToTensor(),  # convert the PIL Image to a tensor
            transforms.Normalize((0.485, 0.456, 0.406),  # normalize image for pre-trained model
                                 (0.229, 0.224, 0.225))])

        self.data_loader = get_loader(transform=self.transform_test,
                                 mode='test',
                                 cocoapi_loc='../')

    def get_img_tensor(self,path):
        im_pil = Image.open(path)
        im_tensor = self.transform_test(im_pil).unsqueeze(0)

        return im_pil, im_tensor

    def clean_sentence(self, output):
        sentense = ''
        for i in output:
            word = self.data_loader.dataset.vocab.idx2word[i]
            if i == 0:
                continue
            if i == 1:
                break
            if i == 18:
                sentense = sentense + word
            else:
                sentense = sentense + ' ' + word

        return sentense.strip()

    def get_caption(self,img_tensor):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(device)
        print("running")

        # Models
        encoder_file = 'legit_model/encoder_1.pkl'
        decoder_file = 'legit_model/decoder_1.pkl'

        # Embed and hidden
        embed_size = 512
        hidden_size = 512

        # The size of the vocabulary.
        vocab_size = 8856

        # Initialize the encoder and decoder, and set each to inference mode.
        encoder = EncoderCNN(embed_size)
        encoder.eval()

        decoder = DecoderRNN(embed_size, hidden_size, vocab_size)
        decoder.eval()

        # Load the trained weights.
        encoder.load_state_dict(torch.load(os.path.join('./models', encoder_file)))
        decoder.load_state_dict(torch.load(os.path.join('./models', decoder_file)))

        # Move models to GPU if CUDA is available.
        encoder.to(device)
        decoder.to(device)

        img_d = img_tensor.to(device)

        # Obtain the embedded image features.
        features = encoder(img_d).unsqueeze(1)

        # Pass the embedded image features through the model to get a predicted caption.
        img_output = decoder.sample(features)

        sentence = self.clean_sentence(img_output)

        return sentence

    def visualize(self, img_pil, sentence):
        plt.imshow(np.squeeze(img_pil))
        plt.title(sentence)
        plt.show()
