# CS6353
## Dataset from https://aistudio.baidu.com/datasetdetail/102884/0
## OCR model from https://aistudio.baidu.com/projectdetail/4330587
## intel model from https://github.com/openvinotoolkit/open_model_zoo/blob/master/demos/handwritten_text_recognition_demo/python/README.md
## BERT model from https://huggingface.co/google-bert/bert-base-chinese

## To get result for OCRv3 model go to folder PaddleOCR use command
### python tools/infer/predict_rec.py --image_dir="../data/train_data/0.png" --rec_model_dir="./inference/rec_ppocrv3/Student"

## To get result for intel model go to folder openVINO/open_model_zoo/tools/infer/ use command
### python predict_rec.py