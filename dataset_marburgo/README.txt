Identifying and Counting Avian Blood Cells in Whole Slide Images via Deep Learning

Simple Summary: Avian blood analysis is crucial for understanding the health of birds. Currently, avian blood cells are often counted manually in microscopic images, which is time-consuming, expensive, and, prone to errors. In this article, we present a novel deep learning approach to automate the quantification of different types of avian red and white blood cells in whole slide images of avian blood smears. Our approach supports ornithologists in terms of hematological data acquisition, accelerates avian blood analysis, and achieves high accuracy in counting different types of avian blood cells.

This dataset contains annotated image datasets and models trained on this data as described in our paper. Each image is a crop from a whole slide image of an avian blood smear. We provide annotated scans and models for both steps in our proposed approach, i.e., tile selection and cell instance segmentation. The tile selection model is provided in the generic ONNX format (.onnx) whereas the instance segmentation model is saved as a Detectron2 model in the PyTorch model format (.pth). Please see our publicly available code repository for instructions on how to apply these models on custom data.

[1] dataset_countability.tar.gz: This compressed archive contains images for the tile selection task. The data is split up into training set (train) and validation set (val) which are further divided into positive (pos) and negative (neg) samples. In summary, the folder structure is as follows:

	./
	├─ train/
	│  ├─ pos/
	│  │  ├─ *.png
	│  ├─ neg/
	│     ├─ *.png
	├─ val/
	   ├─ pos/
	   │  ├─ *.png
	   ├─ neg/
	      ├─ *.png



[2] dataset_segmentation.tar.gz: This compressed archive contains images and annotations for the instance segmentation task. The image data is split up into training set (train) and validation set (val). Furthermore, we provide JSON files following the COCO format (https://cocodataset.org/#format-data) for training (train.json) and validation (val.json), respectively. The annotations contain a bounding box and a segmentation mask for each single cell instance.
	In our particular case, we use the following structure and data fields:

		{
			"info": {
				"year": int,
				"version": str,
				"description": str,
				"contributor": str,
				"url": str
			},
			"images": [
				{
					"id": int,
					"width": int,
					"height": int,
					"file_name": str,
				}
			],
			"annotations": [
				{
					"id": int,
					"image_id": int,
					"category_id": int,
					"segmentation": RLE,
					"area": float,
					"bbox": [x,y,width,height],
					"iscrowd": 0 or 1,
				}
			],
			"licenses": [],
			"categories": [
				{
					"id": int,
					"name": str,
					"supercategory": str,
				}
			]
		}


[3] efficientNet_B0.onnx: The tile selection model saved in the generic ONNX format (.onnx). 

[4] condInst_R101.pth: The cell instance segmentation model saved as a Detectron2 model in the PyTorch format (.pth).


If you have used our datasets or models in your research, please consider citing our paper as described at https://github.com/umr-ds/avibloodcount.
