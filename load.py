from os import listdir
from os.path import isdir, isfile, join
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from imageio import imread


def get_files(directory):
    return [f for f in listdir(directory) if isfile(join(directory, f)) and f.lower().endswith(('png', 'jpg', 'bmp'))]


def get_dirs(directory):
    return [f for f in listdir(directory) if isdir(join(directory, f))]


def load(directory,
         batchsize=128,
         max_workers=12,
         dtype=np.uint8,
         fn=lambda x: x):
    executor = ThreadPoolExecutor(max_workers=max_workers)
    files = get_files(directory)
    dirs = get_dirs(directory)
    for _directory in dirs:
        files.extend([join(_directory, filename) for filename in listdir(join(directory, _directory))])
    total = len(files)
    filename_batches = np.array_split(files, total // batchsize)
    print("%s images found." % len(files))

    def read(directory, filename):
        #implement lambda function below
        img = imread(join(directory, filename))
        return img, filename

    j = 0
    for filename_batch in tqdm(filename_batches):
        futures = [executor.submit(read, directory, filename) for filename in filename_batch]
        batch = []
        names = []
        for future in futures:
            try:
                image, filename = future.result()
                batch.append(image)
                names.append(filename)
            except Exception as e:
                print('load() generated an exception: %s' % e)
        j += 1
        # print("Rendering batch %s/%s" % (j, len(filename_batches)), end="\r")
        yield batch, names, total
