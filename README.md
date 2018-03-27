Someone interested in Object detecion API **yolo_v2** implementation. 
Just to go [object_detection_api](https://github.com/rky0930/yolo_v2/tree/master/research/object_detection)  

The most of source codes originally come from [Tensorflow-models](https://github.com/tensorflow/models).  
I just personally added Yolo v2 using Object detecion API. 

In addition to base Tensorflow detection model definitions, this 
version includes:  

* Classification model:
  * [Darknet19](https://github.com/rky0930/models/blob/object_detection_yolo/research/slim/nets/darknet.py)
* Detection model:
  * You Look Only Once v2(YOLOv2) with Darknet19
* Frozen weight (trained on ImageNet 2015) for Darknet 19 model to be used for 
  fine-tunning detection model. ([download](https://drive.google.com/open?id=1bYeZHNyQgTqzYTh_9cW3SoEb75SiH7yQ))  
  Final evaluation result
    * Accuracy: 0.7
    * Recal: 0.9
* Frozen weight (trained on MS-COCO) for YOLOv2 model. ([download](https://drive.google.com/open?id=1g3RdX6xpKdkT9ovjB1G4E2coSPK_Gl1K))  
  Final evaluation result(codalab.org, test-dev2017): 
    * Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.198
    * Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.339
    * Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.204
    * Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.044
    * Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.202
    * Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.337
    * Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.185
    * Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.249
    * Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.251
    * Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.056
    * Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.244
    * Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.442
 
 
Added or Updated file list
* Config:
  * research/object_detection/samples/configs/yolo_v2_darknet_19_voc.config
  * research/object_detection/samples/configs/yolo_v2_darknet_19_mscoco.config

* Anchor box: 
  * research/object_detection/anchor_generators/yolo_grid_anchor_generator.py
  * research/object_detection/anchor_generators/yolo_grid_anchor_generator_test.py
  * research/object_detection/builders/anchor_generator_builder.py

* Loss:
  * research/object_detection/core/losses.py
  * research/object_detection/builders/losses_builder.py

* Feature map: 
  * research/slim/nets/darknet.py
  * research/slim/nets/darknet_test.py
  * research/object_detection/models/yolo_v2_darknet_19_feature_extractor.py
  * research/object_detection/models/yolo_v2_darknet_19_feature_extractor_test.py

* Proto: 
  * research/object_detection/protos/anchor_generator.proto
  * research/object_detection/protos/yolo_anchor_generator.proto
  * research/object_detection/protos/losses.proto

* ETC: 
  * research/object_detection/builders/model_builder.py

-------------------------------------

# TensorFlow Models

This repository contains a number of different models implemented in [TensorFlow](https://www.tensorflow.org):

The [official models](official) are a collection of example models that use TensorFlow's high-level APIs. They are intended to be well-maintained, tested, and kept up to date with the latest stable TensorFlow API. They should also be reasonably optimized for fast performance while still being easy to read. We especially recommend newer TensorFlow users to start here.

The [research models](research) are a large collection of models implemented in TensorFlow by researchers. It is up to the individual researchers to maintain the models and/or provide support on issues and pull requests.

The [samples folder](samples) contains code snippets and smaller models that demonstrate features of TensorFlow, including code presented in various blog posts.

The [tutorials folder](tutorials) is a collection of models described in the [TensorFlow tutorials](https://www.tensorflow.org/tutorials/).
