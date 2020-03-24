# imSearch: A generic framework to build your own reverse image search engine

imsearch helps to create your own custom, robust & scalable reverse image search engine. This project uses state of the art object detection algorithm ([yolov3](https://pjreddie.com/darknet/yolo/)) at its core to extract the features from an image. It uses an efficient cross-platform similarity search library [NMSLIB](https://github.com/nmslib/nmslib) for similarity search. [Redis](https://redis.io/) is used as a messaging queue between feature extractor and core engine. [MongoDB](https://www.mongodb.com/) is used to store the meta-data of all the indexed images. HD5 file system is used to store the feature vectors extracted from indexed images. 

## Installation
For the setup, a simple `install.sh` script can be used or can be installed using `pip`.
Follow these simple steps to install imsearch library. 
- Feature extraction is GPU intensive process. So, to make the search real-time, running this engine on GPU enabled machine is recommended. 
- Install CUDA & NVIDIA graphics drivers ([here](https://medium.com/@taylordenouden/installing-tensorflow-gpu-on-ubuntu-18-04-89a142325138))
- Install `PyTorch` ([here](https://pytorch.org/get-started/locally/))
- Install `MongoDB` ([here](https://docs.mongodb.com/manual/tutorial/install-mongodb-on-ubuntu/))
- Install `Redis` ([here](https://www.digitalocean.com/community/tutorials/how-to-install-and-secure-redis-on-ubuntu-18-04))
- Run following commands 
```
pip install --no-binary :all: nmslib
pip install imsearch
```

### Build from source using `install.sh`
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
# choose policy from 'object' or 'global'. Search results will change accordingly.
# object: Object level matching. The engine will look for similarity at object level for every object detected in the image.
# global: Overall similarity using single feature space on the whole image. 
similar = index.knnQuery('path/to/query/image', k=10, policy='object')
```
For detailed usage see [`examples/index.py`](examples/index.py)
## Credit

### YOLOv3: An Incremental Improvement
_Joseph Redmon, Ali Farhadi_ <br>

**Abstract** <br>
We present some updates to YOLO! We made a bunch
of little design changes to make it better. We also trained
this new network that’s pretty swell. It’s a little bigger than
last time but more accurate. It’s still fast though, don’t
worry. At 320 × 320 YOLOv3 runs in 22 ms at 28.2 mAP,
as accurate as SSD but three times faster. When we look
at the old .5 IOU mAP detection metric YOLOv3 is quite
good. It achieves 57.9 AP50 in 51 ms on a Titan X, compared
to 57.5 AP50 in 198 ms by RetinaNet, similar performance
but 3.8× faster. As always, all the code is online at
https://pjreddie.com/yolo/.

[[Paper]](https://pjreddie.com/media/files/papers/YOLOv3.pdf) [[Project Webpage]](https://pjreddie.com/darknet/yolo/) [[Authors' Implementation]](https://github.com/pjreddie/darknet)

```
@article{yolov3,
  title={YOLOv3: An Incremental Improvement},
  author={Redmon, Joseph and Farhadi, Ali},
  journal = {arXiv},
  year={2018}
}
```

### PyTorch-YOLOv3
Minimal PyTorch implementation of YOLOv3 [[GitHub]](https://github.com/eriklindernoren/PyTorch-YOLOv3)