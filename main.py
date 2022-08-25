from textgenrnn import textgenrnn

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir('./') if isfile(join('./', f))]
print(onlyfiles)

bow = {}

for i in onlyfiles:
    textgen.train_from_file('./'+i, num_epochs=1)
    textgen.generate()
        

print(bow)
            
        