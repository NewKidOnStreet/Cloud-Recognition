
import cv2
import pickle
import numpy as np
from sklearn.cluster import MiniBatchKMeans

def sfilter(sigma, tau):

    sigma_x = sigma
    sigma_y = float(sigma)

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x), abs(nstds * sigma_y))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x), abs(nstds * sigma_y))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    gb = np.cos(np.sqrt(x**2 + y**2)*np.pi*tau/sigma)*np.exp(-(x**2 + y**2)/(2*sigma**2))
    return gb

def gabor_fn(sigma, theta, Lambda, psi, gamma):

    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Bounding box
    nstds = 3 # Number of standard deviation sigma
    xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    xmax = np.ceil(max(1, xmax))
    ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    ymax = np.ceil(max(1, ymax))
    xmin = -xmax
    ymin = -ymax
    (y, x) = np.meshgrid(np.arange(ymin, ymax + 1), np.arange(xmin, xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb


def scale_minmax(img):
    return (img - np.min(img))/(np.max(img) - np.min(img))


def get_textons(sparams, train_set, cloud_types, sample_size, name, path):
    gabor_keys = list(range(0, len(sparams)))
    response_dict = {}

    # Iterating over: cloud types, images of type, filter type
    # Temporarily stores responses in dict
    for ctype in cloud_types:
        for image in train_set[ctype]:
            for key, param in zip(gabor_keys, sparams):
                #             print(key, param)
                filter = sfilter(param[0], param[1])

                imgpath = path + ctype + "\\images\\" + image

                img = scale_minmax(cv2.imread(imgpath, 1)[::, ::, 0]/cv2.imread(imgpath, 1)[::, ::, 2])
                output = np.reshape(scale_minmax(cv2.filter2D(img, -1, filter)), (125 * 125, 1))

                if key not in response_dict.keys():
                    response_dict[key] = output
                else:
                    response_dict[key] = np.append(response_dict[key], output)

    responses = np.zeros((img.shape[0] * img.shape[1] * sample_size * 5, len(sparams)))

    keys = list(response_dict.keys())
    for col, key in enumerate(keys):
        responses[::, col] = response_dict[key]
    try:
        kmeans_min = MiniBatchKMeans(n_clusters=30, random_state=0).fit(responses)
    except ValueError:
        return filter, cv2.imread(imgpath, 1)

    with open(name + '.pkl', 'wb') as f:  # Python 3: open(..., 'wb')
        pickle.dump([kmeans_min], f)

    return kmeans_min


def nearest_hist(clouddict, filt_img):

    def chidiff(truth, img):
        return np.sum(np.power((truth - img), 2)/(2*(truth + img)))

    closest = None
    dist = None
    for key in list(clouddict.keys()):
        if closest is None:
            closest = key
            dist = chidiff(clouddict[key], filt_img)
            print(dist)
        elif chidiff(clouddict[key], filt_img) < dist:
            closest = key
            dist = chidiff(clouddict[key], filt_img)

    return key
