#!/usr/bin/env python3

"""
 Copyright (c) 2020-2024 Intel Corporation
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
      http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

import sys
from time import perf_counter
import logging as log
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import cv2
import numpy as np

from openvino import Core, get_version
from utils.codec import CTCCodec

from transformers import BertTokenizer, BertForMaskedLM
import torch

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument("-m", "--model", type=str, required=True,
                      help="Required. Path to an .xml file with a trained model.")
    args.add_argument("-i", "--input", type=str, required=True,
                      help="Required. Path to an image to infer")
    args.add_argument("-d", "--device", type=str, default="CPU",
                      help="Optional. Specify the target device to infer on; CPU, GPU or HETERO is "
                           "acceptable. The demo will look for a suitable plugin for device specified. Default "
                           "value is CPU")
    args.add_argument("-ni", "--number_iter", type=int, default=1,
                      help="Optional. Number of inference iterations")
    args.add_argument("-cl", "--charlist", type=str, default=str(Path(__file__).resolve().parents[3] / "data/dataset_classes/kondate_nakayosi.txt"),
                      help="Path to the decoding char list file. Default is for Japanese")
    args.add_argument("-dc", "--designated_characters", type=str, default=None, help="Optional. Path to the designated character file")
    args.add_argument("-tk", "--top_k", type=int, default=20, help="Optional. Top k steps in looking up the decoded character, until a designated one is found")
    args.add_argument("-ob", "--output_blob", type=str, default=None, help="Optional. Name of the output layer of the model. Default is None, in which case the demo will read the output name from the model, assuming there is only 1 output layer")
    return parser


def get_characters(cl):
    '''Get characters'''
    with open(cl, 'r', encoding='utf-8') as f:
        return ''.join(line.strip('\n') for line in f)


def preprocess_input(image_name, height, width):
    src = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise RuntimeError(f"Failed to imread {image_name}")
    ratio = float(src.shape[1]) / float(src.shape[0])
    tw = int(height * ratio)
    rsz = cv2.resize(src, (tw, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    # [h,w] -> [c,h,w]
    img = rsz[None, :, :]
    _, h, w = img.shape
    # right edge padding
    pad_img = np.pad(img, ((0, 0), (0, height - h), (0, width - w)), mode='edge')
    return pad_img


def main():
    results = []
    expected_results = []
    with open(r"C:\Users\wudi1\Desktop\Project\data\train.txt", "r", encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            name, expected_result = line.split("\t")
            # python handwritten_text_recognition_demo.py 
            input = r"C:\Users\wudi1\Desktop\data\{}".format(name)
            model_path =  r"C:\Users\wudi1\Desktop\Project\OpenVINO\open_model_zoo\tools\model_tools\intel\handwritten-simplified-chinese-recognition-0001\FP32\handwritten-simplified-chinese-recognition-0001.xml" 
            character_list = r"C:\Users\wudi1\Desktop\Project\OpenVINO\open_model_zoo\data\dataset_classes/scut_ept.txt"

            # Plugin initialization
            core = Core()

            # Read IR
            model = core.read_model(model_path)

            input_tensor_name = model.inputs[0].get_any_name()

            output_tensor_name = model.outputs[0].get_any_name()

            characters = get_characters(character_list)
            codec = CTCCodec(characters, r'data/digit_hyphen.txt', 20)

            input_batch_size, input_channel, input_height, input_width = model.inputs[0].shape

            # Read and pre-process input image (NOTE: one image only)
            input_image = preprocess_input(input, height=input_height, width=input_width)[None, :, :, :]

            # Loading model to the plugin
            compiled_model = core.compile_model(model, 'CPU')
            infer_request = compiled_model.create_infer_request()

            # Start sync inference
            infer_request.infer(inputs={input_tensor_name: input_image})
            preds = infer_request.get_tensor(output_tensor_name).data[:]
            result = codec.decode(preds)

            results.append(result[0])
            expected_results.append(expected_result)
            print(result[0])
            print(expected_result)

    count = 0
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    model = BertForMaskedLM.from_pretrained("bert-base-chinese")

    for i in range(len(results)):
        if results[i] == expected_results[i]:
            count += 1
        else:
            new_result = ""
            for i in range(len(results[i])):
                masked = results[i][:i] + "[MASK]"+ results[i][i+1:]
                input_ids = tokenizer.encode(masked, return_tensors='pt')
                mask_token_index = torch.where(input_ids == tokenizer.mask_token_id)[1]

                with torch.no_grad():
                    outputs = model(input_ids)
                
                logits = outputs.logits
                mask_token_logits = logits[0, mask_token_index, :]
                probabilities = torch.nn.functional.softmax(mask_token_logits, dim=-1)
                top_k_values, top_k_indices = torch.topk(probabilities, 3)
                predicted_token = tokenizer.decode(top_k_indices[0, 0])
                new_result += predicted_token
            print(new_result)
            if new_result == expected_results[i]:
                count +=1
                print(i)
    
    print(count/len(results))
    sys.exit()


if __name__ == '__main__':
    main()
