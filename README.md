# imSearch: A generic framework to build your own reverse image search engine

imsearch helps to create your own custom, robust & scalable reverse image search engine. This project uses state of the art object detection algorithm ([retinanet](https://arxiv.org/pdf/1708.02002.pdf)) at its core to extract the features from an image. It uses an efficient cross-platform similarity search library [NMSLIB](https://github.com/nmslib/nmslib) for similarity search. [Redis](https://redis.io/) is used as a messaging queue between feature extractor and core engine. [MongoDB](https://www.mongodb.com/) is used to store the meta-data of all the indexed images. HD5 file system is used to store the feature vectors extracted from indexed images. 

## Installation
For the setup, a simple `install.sh` script can be used.
Follow these simple steps to install imsearch library. 
- Feature extraction is GPU intensive process. So, to make the search real-time, running this engine on GPU enabled machine is recommended. 
- Install CUDA & NVIDIA graphics drivers ([here](https://medium.com/@taylordenouden/installing-tensorflow-gpu-on-ubuntu-18-04-89a142325138))
- Install `tensorflow` ([here](https://www.tensorflow.org/install/))
- Install `MongoDB` ([here](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/))
- Install `Redis` ([here](https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-redis-on-ubuntu-18-04))
- Run following commands 
```
git clone https://github.com/rikenmehta03/imsearch.git
chmod +x install.sh
./install.sh
```

## Example usage
```
import glob
import imsearch

all_images = glob.glob('path/to/image/folder')

# Initialize the index
index = imsearch.init('test')

# Add single image to the index
index.addImage(all_images[0]) 

# Add multiple image to the index
index.addImageBatch(all_images[1:])

# Create index and make it ready for the search query
index.createIndex() 

# find k nearest similar images
similar = index.knnQuery('path/to/query/image', k=10)
```
