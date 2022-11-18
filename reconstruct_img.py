import matplotlib.pyplot as plt
from PIL import Image as im
import numpy as np

# input type: list[numpy]
def reconstruct_spectrogram(reconstructions, name=None):
    for i in range(len(reconstructions)):
        rec = reconstructions[i]
        if (np.min(rec) < 0):
            rec += abs(np.min(rec))
            times = 255 / np.max(rec)
            rec *= times
        img = im.fromarray(rec)
        img = img.convert('RGB')
        if name != None:
            img.save("./rec_imgs/"+str(name[i])+".png")
        else:
            img.save("rec"+str(i)+".png")
