import numpy as np

class Dataset:
    def __init__(self, label_file, label_data, image_file, image_data):
        print("Processing dataset: {}".format(label_file))
        self.label_filename = label_file
        self.image_filename = image_file
        self.label_header = convert_be_int32(label_data[:4])
        self.image_header = convert_be_int32(image_data[:4])
        self.total_items = convert_be_int32(label_data[4:8])
        if convert_be_int32(image_data[4:8]) != self.total_items:
            raise RuntimeError("Mismatched number of entries in " + label_file + " and " + image_file)
        self.labels = list(map(int, label_data[8:]))
        self.label_vectors = list(map(vectorize, label_data[8:]))
        self.image_height = convert_be_int32(image_data[8:12])
        self.image_width = convert_be_int32(image_data[12:16])
        self.image_size = self.image_height*self.image_width
        self.images = [np.reshape(np.fromiter(map(lambda pixel: int(pixel)/256.0, image_data[i:i + self.image_size]), float), (self.image_size, 1)) for i in range(16, len(image_data), self.image_size)]
        self.data = list(zip(self.images, self.label_vectors))
        self.test_zip = list(zip(self.images, self.labels))

    def __str__(self):
        data = {k: v for k, v in self.__dict__.items() if k not in {'labels', 'images', 'data', 'label_vectors', 'test_zip'}}
        return str(self.__class__.__name__) + ": " + str(data)

    def image_to_str(self, ix):
        image_str = "".join(list(map(lambda x: "#" if x > 0.5 else "+" if x > 0.25 else "." if x > 0 else " ", list(self.images[ix]))))
        return [image_str[i:i + self.image_width] for i in range(0, len(image_str), self.image_width)]

    def print_image(self, ix):
        print("Value: " + str(self.labels[ix]))
        for line in self.image_to_str(ix):
            print(line)

def read_dataset(label_file, image_file):
    with open(label_file, "rb") as labels:
        with open(image_file, "rb") as images:
            return Dataset(label_file, labels.read(), image_file, images.read())

def convert_be_int32(bytes):
    return bytes[0] << 24 | bytes[1] << 16 | bytes[2] << 8 | bytes[3]

def vectorize(byte_val):
    arr = np.zeros((10,1))
    arr[int(byte_val)] = 1.0
    return arr