import os
import numpy as numpy
from skimage import io, transform
import numpy as np

weight = 256
hight = 256
data = None
labels = None

domains = ["amazon", "dslr", "webcam"]


def process_image(image, weight, hight):
    img = io.imread(image)
    img = transform.resize(img, (weight, hight), mode="reflect")
    return img


for d in domains:
    #path = "domain_adaptation_images/" + d + "/" "images/"
    path = "data/Office31/" + d + "/images/"
    print("processing " + path)

    for _, dirnames, _ in os.walk(path):
        dirnames.sort()
        for dirname in dirnames:
            index = dirnames.index(dirname)
            workdir = os.path.join(path, dirname)
            print(workdir)
            processed_images = io.ImageCollection(
                workdir + "/*.jpg", load_func=process_image, weight=weight, hight=hight)
            label = np.full(len(processed_images),
                            fill_value=index, dtype=np.int64)
            # print(processed_images)
            images = io.concatenate_images(processed_images)

            if index == 0:
                data = images
                labels = label

            else:
                data = np.vstack((data, images))
                labels = np.append(labels, label)

    print(np.shape(data))
    print(np.shape(labels))

    partial = [0, 1, 5, 10, 11, 12, 15, 16, 17, 22]
    idx = np.where(np.isin(labels, partial))
    data_p = data[idx]
    label_p = labels[idx]

    print(np.shape(data_p))
    print(np.shape(label_p))

    np.savez("office31/"+d+"10.npz",
             data=data_p, label=label_p)
    print("Saved {}10.npz. It's length is {}".format(d, len(labels[idx])))
    np.savez("office31/"+d+"31.npz", data=data, label=labels)
    print("Saved {}31.npz. It's length is {}".format(d, len(labels)))
