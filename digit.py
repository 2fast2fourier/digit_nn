import dataset
import network
import random
import time

start_time = time.time()

training_dataset = dataset.read_dataset("./training/train-labels.idx1-ubyte", "./training/train-images.idx3-ubyte")
test_dataset = dataset.read_dataset("./test_data/t10k-labels.idx1-ubyte", "./test_data/t10k-images.idx3-ubyte")
print(training_dataset)
print(test_dataset)
# training_dataset.print_image(random.randrange(0, training_dataset.total_items))
# test_dataset.print_image(random.randrange(0, test_dataset.total_items))


# import mnist_loader
# training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
# training_data = list(training_data)

digit_nn = network.Network([784,30,10])
digit_nn.sgd(training_dataset.data, 30, 10, 3.0, test_dataset=list(test_dataset.test_zip))

# import networkoooo
# net = networkoooo.Network([784, 30, 10])
# net.SGD(training_dataset.data, 30, 10, 3.0, test_data=test_dataset.test_zip)

print("Overall Time: {}".format(time.time() - start_time))