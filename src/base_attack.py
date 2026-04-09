import torch
import torch.nn as nn
import jieba
import string
import numpy as np
from copy import deepcopy
from tqdm import tqdm
import time
import os
from .GenerateAPI import generate as mygenerate
from undecorated import undecorated
from types import MethodType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    StoppingCriteria,
    StoppingCriteriaList,
    StopStringCriteria,
)
torch.autograd.set_detect_anomaly(True)

class BaseAttack:
    def __init__(self, model, tokenizer, device, config, space_token):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.model = self.model.to(self.device)
        self.embedding = self.model.get_input_embeddings().weight  # Keep the live reference!
        self.model.requires_grad_(True)  # Force PyTorch to track gradients for Llama 3.1
        self.specical_token = self.tokenizer.all_special_tokens
        self.specical_id = self.tokenizer.all_special_ids
        self.eos_token_id = self.model.config.eos_token_id
        self.pad_token_id = self.model.config.pad_token_id
        self.space_token = space_token

        self.num_beams = config['num_beams']
        self.num_beam_groups = config['num_beam_groups']
        self.max_per = config['max_per']
        self.max_len = config['max_len']
        self.source_language = config['src']
        self.target_language = config['tgt']

        self.softmax = nn.Softmax(dim=1)
        self.bce_loss = nn.BCELoss()

    def compute_seq_len(self, seq):
        if seq[0].eq(self.pad_token_id):
            return int(len(seq) - sum(seq.eq(self.pad_token_id)))
        else:
            if self.pad_token_id == self.eos_token_id:
                return int(len(seq)) - 1
            else:
                return int(len(seq) - sum(seq.eq(self.pad_token_id))) - 1

    def get_prediction(self, text):
        def remove_pad(s):
            for i, tk in enumerate(s):
                if tk == self.eos_token_id and i != 0:
                    return s[:i + 1]
            return s

        input_token = self.tokenizer(text, return_tensors="pt", padding=True)
        input_ids = input_token["input_ids"].to(self.device)
        attention_mask = input_token["attention_mask"].to(self.device)
        prompt_len = input_ids.shape[1]

        # 1. Generate the sequence safely and quickly (without gradients)
        with torch.no_grad():
            scores_criteria = StoppingCriteriaList([
                StopStringCriteria(self.tokenizer, stop_strings=[self.tokenizer.eos_token])
            ])

            # Using the standard HF generate method
            out_token = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pad_token_id=self.pad_token_id,
                num_beams=self.num_beams,
                num_beam_groups=self.num_beam_groups,
                max_length=self.max_len,
                stopping_criteria=scores_criteria,
                output_scores=False,  # We don't need the detached scores!
                return_dict_in_generate=True
            )

        seqs_full = out_token['sequences']

        # 2. THE SILVER BULLET: Re-run the full sequence in ONE standard forward pass
        self.model.requires_grad_(True)  # Force PyTorch to track
        seq_attention_mask = (seqs_full != self.pad_token_id).long()

        with torch.set_grad_enabled(True):  # Guarantee the graph is built
            outputs = self.model(
                input_ids=seqs_full,
                attention_mask=seq_attention_mask
            )

        # 3. Slice out the continuous logits for ONLY the generated tokens
        # (Starts at prompt_len - 1, ends at the second to last token)
        gen_logits = outputs.logits[:, prompt_len - 1: -1, :]

        # Convert back to the exact Tuple format the old code expects
        out_scores = tuple(gen_logits[:, i, :] for i in range(gen_logits.shape[1]))

        # Clean up the padding formatting
        seqs = [remove_pad(seq) for seq in seqs_full]
        seqs[0] = seqs[0][-len(out_scores):]
        pred_len = [self.compute_seq_len(seq) for seq in seqs]

        return pred_len, seqs, out_scores

    def get_trans_string_len(self, text):
        pred_len, seqs, _ = self.get_prediction(text)
        return seqs[0], pred_len[0]

    def get_trans_len(self, text):
        pred_len, _, _ = self.get_prediction(text)
        return pred_len

    def get_trans_strings(self, text):
        pred_len, seqs, _ = self.get_prediction(text)
        out_res = [self.tokenizer.decode(seq, skip_special_tokens=True) for seq in seqs]
        return out_res, pred_len

    def prepare_attack(self, text):
        ori_len = self.get_trans_len(text)[0]
        best_adv_text, best_len = deepcopy(text[0]), ori_len
        current_adv_text, current_len = deepcopy(text[0]), ori_len
        return ori_len, (best_adv_text, best_len), (current_adv_text, current_len)

    def run_attack(self, text, used_tokens=None):
        assert len(text) == 1
        ori_len, (best_adv_text, best_len), (current_adv_text, current_len) = self.prepare_attack(text)
        adv_his = [(deepcopy(current_adv_text), deepcopy(current_len), 0.0)]
        modify_pos = []

        pbar = tqdm(range(self.max_per))
        t1 = time.time()
        for it in pbar:
            try:
                loss = self.compute_loss([current_adv_text])
                loss = sum(loss)
                self.model.zero_grad()
                with torch.cuda.amp.autocast(dtype=torch.float32):  # ✅ safe float32 grad
                    loss.backward()
                grad = self.embedding.grad
                new_strings = self.mutation(current_adv_text, grad, modify_pos, used_tokens)

                if not new_strings:  # ✅ safety check
                    print("error: mutation returned empty strings")
                    return False, adv_his

                current_adv_text, current_len = self.select_best(new_strings)
                log_str = "%d, %d, %.2f" % (it, len(new_strings), best_len / ori_len)
                pbar.set_description(log_str)
                if current_len > best_len:
                    best_adv_text = deepcopy(current_adv_text)
                    best_len = current_len
                t2 = time.time()
                adv_his.append((best_adv_text, int(best_len), t2 - t1))
            except Exception as e:
                print("Exception in attack loop:", e)
                return False, adv_his
        return True, adv_his


class SEAttack(BaseAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(SEAttack, self).__init__(model, tokenizer, device, config, space_token)

        self.port_dict = {
            'de': 9000,
            'en': 9000,
            'zh': 9001,
            'pt': 9000
        }

    def split_token(self, origin_target_sent):
        if self.target_language == 'zh':
            target_sent_seg = ' '.join(jieba.cut(origin_target_sent))
        else:
            target_sent_seg = ' '.join(origin_target_sent.split(' '))
        return target_sent_seg


class MyAttack(BaseAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(MyAttack, self).__init__(model, tokenizer, device, config, space_token)
        self.insert_character = string.punctuation
        self.insert_character += string.digits
        self.insert_character += string.ascii_letters

    def leave_eos_loss(self, scores, pred_len):
        loss = []
        for i, s in enumerate(scores):
            if self.pad_token_id != self.eos_token_id:
                s[:, self.pad_token_id] = 1e-12
            eos_p = self.softmax(s)[:pred_len[i], self.eos_token_id]
            loss.append(self.bce_loss(eos_p, torch.zeros_like(eos_p)))
        return loss

    def leave_eos_target_loss(self, scores, seqs, pred_len):
        loss = []
        for i, s in enumerate(scores):
            s = s.clone()  # From our previous fix
            if self.pad_token_id != self.eos_token_id:
                s[:, self.pad_token_id] = 1e-12
            softmax_v = self.softmax(s)
            eos_p = softmax_v[:pred_len[i], self.eos_token_id]
            target_p = torch.stack([softmax_v[iii, tk] for iii, tk in enumerate(seqs[i][1:])])
            target_p = target_p[:pred_len[i]]

            pred = eos_p + target_p
            pred[-1] = pred[-1] / 2

            # THE FIX: Clamp the values to strictly stay inside (0, 1) to prevent log(0)
            pred = torch.clamp(pred, min=1e-7, max=1.0 - 1e-7)

            loss.append(self.bce_loss(pred, torch.zeros_like(pred)))
        return loss

    def compute_best_len(self, seq):
        # return int(len(seq) - sum(seq.eq(self.pad_token_id)))
        if seq[0].eq(self.pad_token_id):
            return int(len(seq) - sum(seq.eq(self.pad_token_id)))
        else:
            if (self.pad_token_id == self.eos_token_id):
                return int(len(seq) - sum(seq.eq(self.pad_token_id)))
            else:
                return int(len(seq) - sum(seq.eq(self.pad_token_id))) - 1

    @torch.no_grad()
    def select_best(self, new_strings, batch_size=30):
        seqs = []
        batch_num = len(new_strings) // batch_size
        if batch_size * batch_num != len(new_strings):
            batch_num += 1

        for i in range(batch_num):
            st, ed = i * batch_size, min(i * batch_size + batch_size, len(new_strings))
            input_token = self.tokenizer(
                new_strings[st:ed],
                return_tensors="pt",
                padding=True,
                truncation=True
            )
            input_ids = input_token['input_ids'].to(self.device)
            attention_mask = input_token['attention_mask'].to(self.device)

            stopping = StoppingCriteriaList([
                StopStringCriteria(self.tokenizer, stop_strings=[self.tokenizer.eos_token])
            ])

            trans_res = self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                num_beams=self.num_beams,
                num_beam_groups=self.num_beam_groups,
                max_length=self.max_len,
                stopping_criteria=stopping,
                output_scores=True,
                return_dict_in_generate=True,
            )

            seqs.extend(trans_res['sequences'].tolist())

            length_output = np.array([self.compute_best_len(torch.tensor(seq)) for seq in seqs])
            if any(element >= self.max_len - 1 for element in length_output):
                break

        pred_len = np.array([self.compute_best_len(torch.tensor(seq)) for seq in seqs])
        return new_strings[pred_len.argmax()], max(pred_len)

    # @torch.no_grad()
    # def select_best(self, new_strings, batch_size=30):
    #     seqs = []
    #     # batch_num = len(new_strings) // batch_size
    #     # if batch_size * batch_num != len(new_strings):
    #     #     batch_num += 1
    #     for i in range(len(new_strings)):
    #         # st, ed = i * batch_size, min(i * batch_size + batch_size, len(new_strings))
    #         input_token = self.tokenizer(new_strings[i], return_tensors="pt", padding=True).input_ids
    #         input_token = input_token.to(self.device)
    #         # trans_res = translate(
    #         #     self.model, input_token,
    #         #     early_stopping=False, num_beams=self.num_beams,
    #         #     num_beam_groups=self.num_beam_groups, use_cache=True,
    #         #     max_length=self.max_len
    #         # )
    #
    #         scores = StoppingCriteriaList(
    #             [
    #             EndOfFunctionCriteria(
    #             start_length=len(input_token[0]),
    #             eos=EOS,
    #             tokenizer=self.tokenizer,
    #                 )
    #             ]
    #         )
    #         trans_res = self.model.generate(
    #             input_token,
    #             # early_stopping=True,
    #             # use_cache=True,
    #             # # num_beams=2,
    #             # temperature=0.0,
    #             # no_repeat_ngram_size=2,
    #             # repetition_penalty=2.0,
    #             stopping_criteria=scores,
    #             max_length=self.max_len,
    #             # num_return_sequences=30,
    #             # do_sample=True,
    #             # length_penalty=-1,
    #             output_scores=True, return_dict_in_generate=True,
    #         )
    #         seqs.extend(trans_res['sequences'].tolist())
    #     pred_len = np.array([self.compute_best_len(torch.tensor(seq)) for seq in seqs])
    #     assert len(new_strings) == len(pred_len)
    #     return new_strings[pred_len.argmax()], max(pred_len)

    def prepare_attack(self, text):
        ori_len = self.get_trans_len(text)[0]  # int
        best_adv_text, best_len = deepcopy(text[0]), ori_len
        current_adv_text, current_len = deepcopy(text[0]), ori_len  # current_adv_text: List[str]
        return ori_len, (best_adv_text, best_len), (current_adv_text, current_len)

    def compute_loss(self, xxx):
        raise NotImplementedError

    def mutation(self, current_adv_text, grad, modify_pos, used_tokens = None):
        raise NotImplementedError

    def find_inserted_token(self, original, modified):
        original_words = set(original.split())
        for word in modified.split():
            if word not in original_words:
                return word
        return None

    def run_black_attack(self, text, used_tokens=None):
        if used_tokens is None:
            used_tokens = set()

        assert len(text) == 1
        current_adv_text = text[0]
        prev_adv_text = text[0]  # ← track previous to find inserted token
        current_len = 0
        adv_his = [(current_adv_text, deepcopy(current_len), 0.0)]
        modify_pos = []

        pbar = tqdm(range(self.max_per))
        t1 = time.time()
        for it in pbar:
            try:
                grad = 0
                new_strings = self.mutation(current_adv_text, grad, modify_pos, used_tokens)

                if new_strings:
                    prev_adv_text = current_adv_text  # ← save before updating
                    current_adv_text, current_len = self.select_best(new_strings)

                    # find and penalize only the winning token
                    winning_token = self.find_inserted_token(prev_adv_text, current_adv_text)
                    if winning_token:
                        used_tokens.add(winning_token)

                    log_str = "%d, %d, %.2f" % (it, len(new_strings), current_len)
                    pbar.set_description(log_str)
                    t2 = time.time()
                    adv_his.append((current_adv_text, int(current_len), t2 - t1))
                else:
                    return False, adv_his
            except Exception as e:
                import traceback
                print(f"BLACK BOX CRASH: {e}")
                traceback.print_exc()
                return False, adv_his
        return True, adv_his

    def run_attack(self, text, used_tokens = None):
        assert len(text) == 1
        ori_len, (best_adv_text, best_len), (current_adv_text, current_len) = self.prepare_attack(text)
        adv_his = [(deepcopy(current_adv_text), deepcopy(current_len), 0.0)]
        modify_pos = []

        pbar = tqdm(range(self.max_per))
        t1 = time.time()
        for it in pbar:
            loss = self.compute_loss([current_adv_text])
            loss = sum(loss)
            self.model.zero_grad()
            try:
                loss.backward()
                grad = self.embedding.grad
                new_strings = self.mutation(current_adv_text, grad, modify_pos, used_tokens)
                if new_strings:
                    current_adv_text, current_len = self.select_best(new_strings)
                    log_str = "%d, %d, %.2f" % (it, len(new_strings), best_len / ori_len)
                    pbar.set_description(log_str)
                    if current_len > best_len:
                        best_adv_text = deepcopy(current_adv_text)
                        best_len = current_len
                    t2 = time.time()
                    adv_his.append((best_adv_text, int(best_len), t2 - t1))
                else:
                    return False, adv_his
            except Exception as e:
                import traceback
                print(f"GRADIENT LOOP CRASH: {e}")
                traceback.print_exc()
                return False, adv_his
        return True, adv_his


class BaselineAttack(BaseAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(BaselineAttack, self).__init__(model, tokenizer, device, config, space_token)

        self.insert_character = string.punctuation
        self.insert_character += string.digits
        self.insert_character += string.ascii_letters

    def compute_score(self, text):
        batch_size = len(text)
        # assert batch_size == 1
        index_list = [i * self.num_beams for i in range(batch_size + 1)]

        pred_len, seqs, out_scores = self.get_prediction(text)

        # scores = torch.cat([d.max(1)[0].reshape([1, -1]) for d in out_scores])
        # scores = scores[:, 0][:seqs.shape[1] - 1]
        # assert seqs.shape[1] == len(out_scores) + 1

        scores = [[] for _ in range(batch_size)]
        for out_s in out_scores:
            for i in range(batch_size):
                current_index = index_list[i]
                scores[i].append(out_s[current_index: current_index + 1])
        scores = [torch.cat(s) for s in scores]
        scores = [s[:pred_len[i]] for i, s in enumerate(scores)]
        return scores, seqs, pred_len

    def leave_eos_loss(self, scores, pred_len):
        loss = []
        for i, s in enumerate(scores):
            # Create a clone to prevent in-place modification crashes
            s = s.clone()
            s[:, self.pad_token_id] = 1e-12
            eos_p = self.softmax(s)[:pred_len[i], self.eos_token_id]
            loss.append(self.bce_loss(eos_p, torch.zeros_like(eos_p)))
        return loss

    def untarget_loss(self, scores, seqs, pred_len):
        loss = []
        for i, s in enumerate(scores):
            # Create a clone to prevent in-place modification crashes
            s = s.clone()
            s[:, self.pad_token_id] = 1e-12
            softmax_v = self.softmax(s)
            target_p = torch.stack([softmax_v[iii, s] for iii, s in enumerate(seqs[i][1:])])
            target_p = target_p[:pred_len[i]]
            pred = target_p
            # pred[-1] = pred[-1] / 2
            loss.append(self.bce_loss(pred, torch.zeros_like(pred)))
        return loss