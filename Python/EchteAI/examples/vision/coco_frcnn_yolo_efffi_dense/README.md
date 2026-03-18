## Abstract

Convolutional neural networks operate on high-dimensional visual representations, where final activations can vary significantly even when final predictions remain stable. This makes validation of post-training quantization challenging, as standard accuracy metrics often fail to capture underlying deviations. 

In this work, we analyze the effect of INT8 quantization on a Faster R-CNN model trained on the COCO dataset using ONNX. By collecting and comparing intermediate convolutional activations, we observe that although detection accuracy remains largely unchanged, internal feature maps exhibit measurable differences. The median relative error in the final convolutional layer was found to be 4–6%.

We further show that the error propagation trend across layers is consistent across different input images, suggesting that full dataset evaluation is not always necessary. Additionally, quantization error is significantly lower within detected object regions, while increasing error correlates with bounding box distortion and eventual detection failure.