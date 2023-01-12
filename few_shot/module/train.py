import torch
from torch import optim
from tqdm import tqdm
from ..utils import convert_preds_to_outputs


class Trainer(object):
    def __init__(self, train_data=None, dev_data=None, load_path=None,
                 model=None, args=None, logger=None, loss=None, metrics=None) -> None:
        self.train_data = train_data
        self.model = model
        self.dev_data = dev_data
        self.logger = logger
        self.metrics = metrics
        self.loss = loss
        self.num_epochs = args["num_epochs"]
        self.batch_size = args["batch_size"]
        self.lr = args["learning_rate"]
        self.eval_begin_epoch = args["eval_begin_epoch"]
        self.device = args["device"]
        self.load_path = load_path
        self.save_path = args["save_path"]
        self.refresh_step = 1
        self.best_metric = args["best_metric"]
        self.best_dev_epoch = None
        self.optimizer = None
        if self.train_data is not None:
            self.train_num_steps = len(self.train_data) * self.num_epochs
        self.step = 0
        self.args = args

    def model_return(self):
        self.model.eval()
        return self.model

    def predict(self, predict_model,test_data,process):
        # self.logger.info("Load model successful!")
        predict_model.to(self.device)

        with torch.no_grad():
            with tqdm(total=len(test_data), leave=False, dynamic_ncols=True) as pbar:
                pbar.set_description_str(desc="Test")
                texts = []
                labels = []
                for batch in test_data:
                    batch = (tup.to(self.device) if isinstance(tup, torch.Tensor) else tup for tup in batch)
                    # to cpu/cuda device
                    src_tokens, src_seq_len, first, raw_words = batch
                    preds = self._step((src_tokens, src_seq_len, first), mode="test")
                    outputs = convert_preds_to_outputs(preds, raw_words, process.mapping, process.tokenizer)
                    texts.extend(raw_words)
                    labels.extend(outputs)
                    pbar.update()

        return texts, labels

    def _step(self, batch, mode="train"):
        if mode == "dev":  # dev: compute metric
            src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first, target_span = batch
            pred = self.model.predict(src_tokens, src_seq_len, first)
            self.metrics.evaluate(target_span, pred, tgt_tokens)
            return
        elif mode == "test":  # test: just get pred
            src_tokens, src_seq_len, first = batch
            pred = self.model.predict(src_tokens, src_seq_len, first)
            return pred
        else:  # train: get loss
            src_tokens, tgt_tokens, src_seq_len, tgt_seq_len, first, target_span = batch
            pred = self.model(src_tokens, tgt_tokens, src_seq_len, first)
            loss = self.loss(tgt_tokens, tgt_seq_len, pred)
            return loss

    def before_train(self):
        parameters = []
        params = {'lr': self.lr, 'weight_decay': 1e-2}
        params['params'] = [param for name, param in self.model.named_parameters() if
                            not ('bart_encoder' in name or 'bart_decoder' in name)]
        parameters.append(params)

        params = {'lr': self.lr, 'weight_decay': 1e-2}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if ('bart_encoder' in name or 'bart_decoder' in name) and not ('layernorm' in name or 'layer_norm' in name):
                params['params'].append(param)
        parameters.append(params)

        params = {'lr': self.lr, 'weight_decay': 0}
        params['params'] = []
        for name, param in self.model.named_parameters():
            if ('bart_encoder' in name or 'bart_decoder' in name) and ('layernorm' in name or 'layer_norm' in name):
                params['params'].append(param)
        parameters.append(params)

        self.optimizer = optim.AdamW(parameters)

        if self.args["freeze_plm"]:  # freeze pretrained language model(bart)
            for name, par in self.model.named_parameters():
                if 'prompt_encoder' in name or 'prompt_decoder' in name and "bart_mlp" not in name:
                    par.requires_grad = False

        self.model.to(self.device)
