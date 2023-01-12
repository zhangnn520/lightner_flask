import os
import sys
import logging
import torch
import warnings
from tools.tools import read_json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))
from torch.utils.data import DataLoader
from lightner_flask.few_shot.models.model import PromptBartModel, PromptGeneratorModel
from lightner_flask.few_shot.module.datasets import ConllNERProcessor, ConllNERDataset
from lightner_flask.few_shot.module.train import Trainer
from lightner_flask.few_shot.utils.util import set_seed

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
warnings.filterwarnings("ignore", category=UserWarning)
os.environ["HYDRA_FULL_ERROR"] = "1"

base_dir = os.path.dirname(__file__)

enoch_mapping = {'一般动作': '<<一般动作>>', '操作部件2': '<<操作部件2>>', '嵌套&方位': '<<嵌套&方位>>',
                 '目标部件1位置': '<<目标部件1位置>>','目标部件位置1': '<<目标部件位置1>>', '物理量': '<<物理量>>',
                 '方位': '<<方位>>', '操作部件2位置': '<<操作部件2位置>>','一般工具': '<<一般工具>>', '拆卸动作': '<<拆卸动作>>',
                 '嵌套&目标部件2': '<<嵌套&目标部件2>>', '操作程序': '<<操作程序>>','嵌套&目标部件1': '<<嵌套&目标部件1>>',
                 '安装动作': '<<安装动作>>', '嵌套&操作部件1': '<<嵌套&操作部件1>>', '操作规范': '<<操作规范>>',
                 '目标部件2': '<<目标部件2>>', '链接': '<<链接>>', '量词': '<<量词>>', '嵌套&操作部件2': '<<嵌套&操作部件2>>',
                 '操作部件1位置': '<<操作部件1位置>>', '目标部件1': '<<目标部件1>>', '工作区域': '<<工作区域>>',
                 '操作程序选项': '<<操作程序选项>>',
                 '操作部件1': '<<操作部件1>>'}

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                    datefmt='%m/%d/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

cfg = read_json(os.path.join(base_dir, "config.json"))


def get_data(content_list):
    process = ConllNERProcessor(lines=content_list, mapping=enoch_mapping,
                                bart_name=cfg["bart_name"], learn_weights=cfg["learn_weights"])
    label_ids = list(process.mapping2id.values())
    test_dataset = ConllNERDataset(data_processor=process, mode='test')
    test_dataloader = DataLoader(test_dataset,
                                 collate_fn=test_dataset.collate_fn,
                                 batch_size=cfg["batch_size"],
                                 num_workers=cfg["num_workers"])
    return process, label_ids, test_dataloader


def ner_server(process, label_ids):
    set_seed(cfg["seed"])  # set seed, default is 1
    # 将配置文件的相对位置转化为绝对路径
    cfg["bart_name"] = os.path.join(base_dir, cfg["bart_name"])
    cfg["model_predict_path"] = os.path.join(base_dir, cfg["model_predict_path"])
    cfg["write_path"] = os.path.join(base_dir, cfg["write_path"])

    prompt_model = PromptBartModel(tokenizer=process.tokenizer, label_ids=label_ids, args=cfg)
    model = PromptGeneratorModel(prompt_model=prompt_model,
                                 bos_token_id=0,
                                 eos_token_id=1,
                                 max_length=cfg["tgt_max_len"],
                                 max_len_a=cfg["src_seq_ratio"],
                                 num_beams=cfg["num_beams"],
                                 do_sample=False,
                                 repetition_penalty=1,
                                 length_penalty=cfg["length_penalty"],
                                 pad_token_id=1,
                                 restricter=None)
    return model


def my_trainer(model):
    trainer = Trainer(train_data=None,
                      dev_data=None,
                      model=model,
                      args=cfg,
                      logger=logger,
                      loss=None,
                      load_path=cfg["model_predict_path"],
                      metrics=None)
    return trainer


def get_model_first_upload():
    process1 = ConllNERProcessor(mapping=enoch_mapping,
                                 bart_name=cfg["bart_name"],
                                 learn_weights=cfg["learn_weights"])
    label_ids = list(process1.mapping2id.values())
    model = ner_server(process1, label_ids)
    trainer = my_trainer(model)
    model2 = trainer.model_return()
    model2.load_state_dict(torch.load(cfg["model_predict_path"]), strict=False)
    return trainer, model2


def ner_server_main(content_list, trainer, model2):
    process, label_ids, data_loader = get_data(content_list)
    texts, labels = trainer.predict(model2, data_loader, process)
    return texts, labels


def get_words_bio_label(label_list):
    """返回服务所需的数据格式"""
    outputs_list = []
    raw_words, raw_targets = [], []
    raw_word, raw_target = [], []
    for line in label_list:
        if line != "\n":
            raw_word.append(line.split("\t")[0])
            raw_target.append(line.split("\t")[1])
        else:
            raw_words.append(raw_word)
            raw_targets.append(raw_target)
            raw_word, raw_target = [], []

    for words, targets in zip(raw_words, raw_targets):
        output, entities, entity_tags, entity_spans = {}, [], [], []
        start, end, start_flag = 0, 0, False
        for idx, tag in enumerate(targets):
            if tag.startswith('B-'):  # 一个实体开头 另一个实体（I-）结束
                end = idx
                if start_flag:  # 另一个实体以I-结束，紧接着当前实体B-出现
                    entities.append("".join(words[start:end]))
                    entity_tags.append(targets[start][2:].lower())
                    entity_spans.append({"start_index": start, "end_index": end})
                    start_flag = False
                start = idx
                start_flag = True
            elif tag.startswith('I-'):  # 实体中间，不是开头也不是结束，end+1即可
                end = idx
            elif tag.startswith('O'):  # 无实体，可能是上一个实体的结束
                end = idx
                if start_flag:  # 上一个实体结束
                    entities.append("".join(words[start:end]))
                    entity_tags.append(targets[start][2:].lower())
                    entity_spans.append({"start_index": start, "end_index": end})
                    start_flag = False
        if start_flag:  # 句子以实体I-结束，未被添加
            entities.append("".join(words[start:end + 1]))
            entity_tags.append(targets[start][2:].lower())
            entity_spans.append({"start_index": start, "end_index": end + 1})
            start_flag = False

        output['entities'] = {i: entities[i_index] for i_index, i in enumerate(entity_tags)}
        output['raw_text'] = "".join(words)
        output['entity_spans'] = {i: entity_spans[i_index] for i_index, i in enumerate(entity_tags)}
        outputs_list.append(output)
    return outputs_list


def get_predict_label(test_dict):
    """将模型预测的数据转化为模型训练的格式，用于数据处理"""
    bio_label_list = list()
    assert len(test_dict["content_list"]) == len(test_dict["labels"]) == len(test_dict["text"])
    for num, string in enumerate(test_dict["content_list"]):
        labels = test_dict["labels"][num]
        texts = test_dict["text"][num]
        bio_label_list += [text + "\t" + labels[index] for index, text in enumerate(texts)]
        bio_label_list.append("\n")
    return bio_label_list
