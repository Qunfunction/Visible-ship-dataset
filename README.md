# Marine ship instance segmentation by deep neural networks using a global and local attention (GALA) mechanism  
GALA : It is the global and local attention mechanism.
LabelMe2COCO:It is a method to convert labelme format to MS COCO format for convolutional neural network training.
Plot a scatter plot of size distribution:It is a method to plot the distribution of image size and object size in a given dataset
coco_annotation extraction:It is a method for extracting annotations containing specific object labels from MS COCO format dataset
coco_image extraction:It is a method for extracting images containing specific object labels from MS COCO format dataset.
crawling_datasets:It is a method for automatically grabbing images of specific object labels from massive online databases
merge_subclasses:It is a method for merging a single label dataset into a dataset containing multiple  object labels.
rename_label:It is a method for modifying the target name to a specific label name in the dataset.

Visible-ship-dataset is a data set for ship instance segmentation tasks in visible light images. This dataset includes two types of marine ship instance segmentation 
datasets, named as MariBoats and MariBoatsSubclass respectively, which can be used for different research purpose.
# The MariBoats dataset
The MariBoats dataset used all the 6.2k images and all the ships labelled were assigned to only one category, namely ‘ship’, resulting in 15.7k ship segmentation annotations. This dataset with one category can satisfy the basic instance segmentation requirements (For example, avoiding obstacles (ships) during unmanned driving under the complex sea scene). 
# The MariBoatsSubclass dataset
The MariBoatsSubclass dataset contains 3.1k images and 4.5k ship annotations. This dataset has six categories of marine ships: Engineering Ship (Eng.), Cargo Ship (Carg.), Speedboat (Sp.), Passenger Ship (Pass.), Official Ship (Off.), and Unknown Ship (Unk.). This dataset can be used for both segmentation of ships and precise identification of marine ship categories in marine scenes. the Visible-ship-dataset draws on the construction process of the Microsoft Common Objects in Context (COCO) datasets, including visible light images with different resolutions. This dataset is a benchmark for researchers to evaluate their approaches. 

# Dataset download
link：https://pan.baidu.com/s/1Lg2doyPyh2uazRma_pZpJw Extraction code:：2w0m ；weiyun link：https://share.weiyun.com/1jFkqLpK 密码：4yi5x3
