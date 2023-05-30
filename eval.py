import argparse
import os
import random

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import gradio as gr

from minigpt4.common.config import Config
from minigpt4.common.dist_utils import get_rank
from minigpt4.common.registry import registry
from minigpt4.conversation.conversation import Chat, Conversation, SeparatorStyle

# imports modules for registration
from minigpt4.datasets.builders import *
from minigpt4.models import *
from minigpt4.processors import *
from minigpt4.runners import *
from minigpt4.tasks import *

from clevr.load_clevr import get_clevr_random_question, generate_output, eval_output

path_info = {
        "clevr_path": "/nobackup/users/zfchen/zt/clevr/CLEVR_v1.0",
        "result_file_path": "/nobackup/users/zfchen/zt/MiniGPT-4/output/result_file_clevr.json",
        "ann_file_path": "/nobackup/users/zfchen/zt/MiniGPT-4/output/ann_file_clevr.json",
        "ques_file_path": "/nobackup/users/zfchen/zt/MiniGPT-4/output/ques_file_clevr.json",
        "output_path": "/nobackup/users/zfchen/zt/MiniGPT-4/output/output_clevr.json",
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Demo")
    parser.add_argument("--nums", type=int, default=100)
    parser.add_argument("--cfg-path", required=True, help="path to configuration file.")
    parser.add_argument("--gpu-id", type=int, default=0, help="specify the gpu to load the model.")
    parser.add_argument(
        "--options",
        nargs="+",
        help="override some settings in the used config, the key-value pair "
        "in xxx=yyy format will be merged into config file (deprecate), "
        "change to --cfg-options instead.",
    )
    args = parser.parse_args()
    return args


def setup_seeds(config):
    seed = config.run_cfg.seed + get_rank()

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    cudnn.benchmark = False
    cudnn.deterministic = True


# ========================================
#             Model Initialization
# ========================================

print('Initializing Chat')
args = parse_args()
cfg = Config(args)

model_config = cfg.model_cfg
model_config.device_8bit = args.gpu_id
model_cls = registry.get_model_class(model_config.arch)
model = model_cls.from_config(model_config).to('cuda:{}'.format(args.gpu_id))

vis_processor_cfg = cfg.datasets_cfg.clevr.vis_processor.train
vis_processor = registry.get_processor_class(vis_processor_cfg.name).from_config(vis_processor_cfg)
chat = Chat(model, vis_processor, device='cuda:{}'.format(args.gpu_id))
print('Initialization Finished')


class ClevrConversation(Conversation):
    prompt = "###Human: <Img><ImageHere></Img> {} ###Assistant: "
    def add_question(self, question):
        self.question = question
    def get_prompt(self):
        return self.prompt.format(self.question)
    def copy(self):
        return ClevrConversation(
            prompt=self.prompt,
            system=self.system,
            # system_img=self.system_img,
            roles=self.roles,
            messages=[[x, y] for x, y in self.messages],
            offset=self.offset,
            sep_style=self.sep_style,
            sep=self.sep,
            sep2=self.sep2,
            conv_id=self.conv_id)


CLEVR_VISION = ClevrConversation(
    system=" ",
    roles=("Human", "Assistant"),
    messages=[],
    offset=2,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)

def gradio_reset(chat_state, img_list):
    if chat_state is not None:
        chat_state.messages = []
    if img_list is not None:
        img_list = []
    return chat_state, img_list

def upload_img(gr_img):
    chat_state = CLEVR_VISION.copy()
    img_list = []
    llm_message = chat.upload_img(gr_img, chat_state, img_list)
    return chat_state, img_list

def gradio_ask(user_message, chat_state):
    chat_state.add_question(user_message)
    chat.ask(user_message, chat_state)
    return chat_state


def gradio_answer(chat_state, img_list, num_beams, temperature):
    llm_message = chat.answer(conv=chat_state,
                              img_list=img_list,
                              num_beams=num_beams,
                              temperature=temperature,
                              max_new_tokens=300,
                              max_length=2000)[0]
    return chat_state, img_list, llm_message


nums = args.nums
question_list = get_clevr_random_question(path_info, split='val', nums=nums)
response_list = []

for question in question_list:
    chat_state, img_list = upload_img(question["image_path"])
    chat_state = gradio_ask(question["question"], chat_state)
    chat_state, img_list, llm_message = gradio_answer(chat_state, img_list, 1, 1)
    response_list.append(llm_message)
    # print(llm_message)
    # chat_state, img_list = gradio_reset(chat_state, img_list)

generate_output(path_info, response_list)
eval_output(path_info)


# ========================================
#             Gradio Setting
# ========================================

