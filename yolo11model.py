import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
from ultralytics import YOLO
from ultralytics.data.dataset import YOLODataset
import ultralytics.data.build as build
import numpy as np

class YOLOWeightedDataset(YOLODataset):
    def __init__(self, *args, mode="train", **kwargs):
        """
        Initialize the WeightedDataset.

        Args:
            class_weights (list or numpy array): A list or array of weights corresponding to each class.
        """

        super(YOLOWeightedDataset, self).__init__(*args, **kwargs)

        self.train_mode = "train" in self.prefix

        # You can also specify weights manually instead
        self.count_instances()
        class_weights = np.sum(self.counts) / self.counts
        self.agg_func = np.mean

        self.class_weights = np.array(class_weights)
        self.weights = self.calculate_weights()
        self.probabilities = self.calculate_probabilities()

    def count_instances(self):
        """
        Count the number of instances per class

        Returns:
            dict: A dict containing the counts for each class.
        """
        self.counts = [0 for i in range(len(self.data["names"]))]
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)
            for id in cls:
                self.counts[id] += 1

        self.counts = np.array(self.counts)
        self.counts = np.where(self.counts == 0, 1, self.counts)

    def calculate_weights(self):
        """
        Calculate the aggregated weight for each label based on class weights.

        Returns:
            list: A list of aggregated weights corresponding to each label.
        """
        weights = []
        for label in self.labels:
            cls = label['cls'].reshape(-1).astype(int)

            # Give a default weight to background class
            if cls.size == 0:
              weights.append(1)
              continue

            # Take mean of weights
            # You can change this weight aggregation function to aggregate weights differently
            # weight = np.mean(self.class_weights[cls])
            # weight = np.max(self.class_weights[cls])
            weight = self.agg_func(self.class_weights[cls])
            weights.append(weight)
        return weights

    def calculate_probabilities(self):
        """
        Calculate and store the sampling probabilities based on the weights.

        Returns:
            list: A list of sampling probabilities corresponding to each label.
        """
        total_weight = sum(self.weights)
        probabilities = [w / total_weight for w in self.weights]
        return probabilities

    def __getitem__(self, index):
        """
        Return transformed label information based on the sampled index.
        """
        # Don't use for validation
        if not self.train_mode:
            return self.transforms(self.get_image_and_label(index))
        else:
            index = np.random.choice(len(self.labels), p=self.probabilities)
            return self.transforms(self.get_image_and_label(index))

build.YOLODataset = YOLOWeightedDataset

hyperparameters = {
    "data":"data.yaml",
    "device":0,
    "optimizer":'SGD',
    "lr0":1e-2,
    "cos_lr":True,
    "pretrained":False,
    "warmup_epochs":0,
    "epochs":100,
    "batch":16,
    "imgsz":800,
    # loss
    "cls":0.5,
    "box":7.5,
    # data augmentation
    "hsv_h":0.015,
    "hsv_s":0.7,
    "hsv_v":0.4,
    "scale":0.5,
    "translate":0.1,
    "degrees":0,
    "fliplr":0.5,
    "copy_paste":0.3,
    "mosaic":1,
    "mixup":0.15,
    "close_mosaic":0,
}

if __name__ == "__main__":
    # Load a model
    model = YOLO("yolo11s.pt")

    # Train the model
    train_results = model.train(**hyperparameters)
    
    # Export the model to ONNX format
    path = model.export(format="onnx")  # return path to exported model
    # validation
    #results = model.val(data=r'C:\Users\USER\Desktop\school\data.yaml',save_json=True)
    # predict
    #results = model.predict(source=r'C:\Users\USER\Desktop\school\Fisheye8K_all\test\images', save=True)

    
