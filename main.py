from textgenrnn import textgenrnn

from os import listdir
from os.path import isfile, join

textgen = textgenrnn()

textgen.load("./filename_trained_network.hdf5")

onlyfiles = [f for f in listdir('./') if isfile(join('./', f))]

for i in onlyfiles:
    textgen.train_from_file('./'+i, num_epochs=1)
    textgen.generate(5, temperature=1.0)
    textgen.save('./filename_trained_network.hdf5')

textgen.generate(1000, temperature=1.0)
            
        