import json
import os
from PIL import Image
import torch
import webdataset as wds
from minigpt4.datasets.datasets.base_dataset import BaseDataset
from collections import OrderedDict


class __DisplMixin:
    def displ_item(self, index):
        sample, ann = self.__getitem__(index), self.annotation[index]

        return OrderedDict(
            {
                "file": ann["image"],
                "question": ann["question"],
                "answer": ann["answer"],
                "image": sample["image"],
            }
        )


class ClevrDataset(BaseDataset, __DisplMixin):

    def __init__(
        self, vis_processor=None, text_processor=None, vis_root=None, ann_paths=[]
    ):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        """
        self.vis_root = vis_root

        self.annotation = []
        for ann_path in ann_paths:
            self.annotation.extend(json.load(open(ann_path, "r"))['questions'])

        self.vis_processor = vis_processor
        self.text_processor = text_processor

        self._add_instance_ids()

    def __getitem__(self, index):

        # TODO this assumes image input, not general enough
        ann = self.annotation[index]

        img_file = ann["image_filename"]
        image_path = os.path.join(self.vis_root, img_file)
        image = Image.open(image_path).convert("RGB")

        image = self.vis_processor(image)
        question = ann["question"]
        image_id = ann["image_index"]

        answer = ann["answer"]

        return {
            "image": image,
            "text_input": answer,
            "question": question,
            "image_id": image_id,
        }

    def collater(self, samples):
        image_list, question_list, answer_list = [], [], []

        num_answers = []

        for sample in samples:
            image_list.append(sample["image"])
            question_list.append(sample["text_input"])
            answer_list.append(sample["answer"])

        return {
            "image": torch.stack(image_list, dim=0),
            "text_input": answer_list,
            "question_split": question_list,
        }