import copy
import time
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import nltk
import string
import random
from copy import deepcopy
from transformers import BertTokenizer, BertModel, BertForMaskedLM
from nltk.tokenize.treebank import TreebankWordTokenizer, TreebankWordDetokenizer
from .base_attack import MyAttack


class Black_box_CharacterAttack(MyAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(Black_box_CharacterAttack, self).__init__(model, tokenizer, space_token, device, config)

    @torch.no_grad()
    def mutation(self, current_adv_text, grad, modify_pos):
        words = current_adv_text.split()

        result_array = [current_adv_text] + [current_adv_text.replace(word, '').strip() for word in words]

        # Capture the full dictionary first
        encoded_inputs = self.tokenizer(result_array, return_tensors="pt", padding=True)

        # Extract the ids and mask safely
        input_token = encoded_inputs["input_ids"].to(self.device)
        attention_mask = encoded_inputs["attention_mask"].to(self.device)
        trans_res = self.model.generate(
            input_token,
            attention_mask=attention_mask,  # ✅ important
            pad_token_id=self.tokenizer.pad_token_id,  # ✅ ensures padding works
            max_length=self.max_len,
            num_beams=self.num_beams,
            num_beam_groups=self.num_beam_groups,
            output_scores=True, return_dict_in_generate=True,
        )
        seqs = []
        seqs.extend(trans_res['sequences'].tolist())
        pred_len = np.array([self.compute_best_len(torch.tensor(seq)) for seq in seqs])
        max_diff_index = np.argmax(np.abs(pred_len - pred_len[0]))
        max_diff_value = pred_len[max_diff_index]
        insert_character = string.punctuation
        insert_character += string.digits
        insert_character += string.ascii_letters
        ori_token = words[max_diff_index - 1]
        candidate = [ori_token[:i] + insert + ori_token[i:] for i in range(len(ori_token)) for insert in
                     insert_character]
        candidate += [ori_token[:i - 1] + self.transfer(ori_token[i - 1]) + ori_token[i:] for i in
                      range(1, len(ori_token))]
        new_strings = []
        new_strings += [current_adv_text.replace(ori_token, c, 1) for c in candidate]
        return new_strings

    @staticmethod
    def transfer(c: str):
        if c in string.ascii_lowercase:
            return c.upper()
        elif c in string.ascii_uppercase:
            return c.lower()
        return c

    def compute_best_len(self,seq):
        # return int(len(seq) - sum(seq.eq(self.pad_token_id)))
        if seq[0].eq(self.tokenizer.pad_token_id):
            return int(len(seq) - sum(seq.eq(self.tokenizer.pad_token_id)))
        else:
            if self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
                return int(len(seq) - sum(seq.eq(self.tokenizer.pad_token_id))) - 1
            else:
                return int(len(seq) - sum(seq.eq(self.tokenizer.pad_token_id)))

    #     return new_strings

class Black_box_WordAttack(MyAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(Black_box_WordAttack, self).__init__(model, tokenizer, space_token, device, config)

    def mutation(self, current_adv_text, grad, modify_pos):
        new_strings = self.token_replace_mutation(current_adv_text, grad, modify_pos)
        return new_strings

    def generate_random_numbers(n, count):
        random_numbers = random.sample(range(n + 1), count)
        return random_numbers

    @torch.no_grad()
    def mutation(self, current_adv_text, grad, modify_pos):
        import random
        import numpy as np
        import torch

        # --- 1. THE BLACK BOX TARGETING SYSTEM ---
        # Find out which word is the most sensitive to latency drops
        words = current_adv_text.split()
        if len(words) == 0:
            return []

        # Create test sentences with one word removed at a time
        result_array = [current_adv_text] + [current_adv_text.replace(word, '', 1).strip() for word in words]

        encoded_inputs = self.tokenizer(result_array, return_tensors="pt", padding=True)
        input_token = encoded_inputs["input_ids"].to(self.device)
        attention_mask = encoded_inputs["attention_mask"].to(self.device)

        # Run the model to check latency impacts
        trans_res = self.model.generate(
            input_token,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.max_len,
            num_beams=self.num_beams,
            num_beam_groups=self.num_beam_groups,
            output_scores=True, return_dict_in_generate=True,
        )

        seqs = trans_res['sequences'].tolist()
        pred_len = np.array([self.compute_best_len(torch.tensor(seq)) for seq in seqs])

        # Rank the words from most impactful to least impactful
        diffs = np.abs(pred_len - pred_len[0])
        sorted_indices = np.argsort(-diffs)

        # --- 2. GATHER CLEAN AMMUNITION ---
        # Pre-generate 50 perfectly clean, normal English words from the vocabulary
        vocab_size = self.tokenizer.vocab_size
        all_tokens = list(range(vocab_size))
        random.shuffle(all_tokens)

        clean_replacements = []
        for tgt_t in all_tokens:
            # Safely skip special tokens if the list exists
            if hasattr(self, 'specical_token') and tgt_t in self.specical_token:
                continue

            single_token_str = self.tokenizer.decode([tgt_t], skip_special_tokens=True).strip()

            # FILTER: Must be standard letters/numbers, purely ASCII, and a real word (length > 2)
            if single_token_str.isalnum() and single_token_str.isascii() and len(single_token_str) > 2:
                clean_replacements.append(single_token_str)

            # Stop once we have 50 good words
            if len(clean_replacements) >= 50:
                break

        # --- 3. THE MACHINE GUN SEARCH ---
        # Try to replace the most vulnerable words until we get a success
        for max_diff_index in sorted_indices:
            if max_diff_index == 0:
                continue  # Skip the baseline sentence

            ori_word = words[max_diff_index - 1]

            # Don't try to attack 1-letter words, skip to the next best target
            if len(ori_word) <= 1:
                continue

            new_strings = []
            for clean_word in clean_replacements:
                # Safely swap the word using pure strings!
                # (This completely prevents the <|begin_of_text|> glitch)
                candidate = current_adv_text.replace(ori_word, clean_word, 1)

                if candidate != current_adv_text:
                    new_strings.append(candidate)

            # If we successfully created mutated strings, return them and end the loop
            if len(new_strings) > 0:
                return new_strings

        # Only return empty if every single word in the sentence was immune
        return []


class Black_box_StructureAttack(MyAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(Black_box_StructureAttack, self).__init__(model, tokenizer, space_token, device, config)
        self.berttokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        bertmodel = BertForMaskedLM.from_pretrained('bert-large-uncased')
        self.bertmodel = bertmodel.eval().to(self.model.device)
        self.num_of_perturb = 50

    def compute_best_len(self, seq):
        """Helper to accurately calculate sequence length minus padding."""
        if seq[0].eq(self.tokenizer.pad_token_id):
            return int(len(seq) - sum(seq.eq(self.tokenizer.pad_token_id)))
        else:
            if self.tokenizer.pad_token_id != self.tokenizer.eos_token_id:
                return int(len(seq) - sum(seq.eq(self.tokenizer.pad_token_id))) - 1
            else:
                return int(len(seq) - sum(seq.eq(self.tokenizer.pad_token_id)))

    @torch.no_grad()
    def mutation(self, current_adv_text, grad, modify_pos):
        import numpy as np
        import torch
        import nltk
        import re

        words = current_adv_text.split()
        if len(words) == 0:
            return []

        # --- NUMBER GUARD HELPER (moved up, defined once) ---
        number_words = {
            "zero", "one", "two", "three", "four", "five", "six", "seven", "eight",
            "nine", "ten", "eleven", "twelve", "thirteen", "fourteen", "fifteen",
            "sixteen", "seventeen", "eighteen", "nineteen", "twenty", "thirty",
            "forty", "fifty", "sixty", "seventy", "eighty", "ninety", "hundred",
            "thousand", "million", "billion", "trillion", "half", "quarter"
        }

        def is_number(word):
            cleaned = re.sub(r'^[\$\£\€\¥]', '', word)
            try:
                float(cleaned.replace(',', ''))
                return True
            except ValueError:
                pass
            if '/' in word:
                parts = word.split('/')
                if all(p.strip().lstrip('-').isdigit() for p in parts if p.strip()):
                    return True
            if re.match(r'^\d+(st|nd|rd|th)$', word, re.IGNORECASE):
                return True
            return word.lower() in number_words

        # --- 1. SENSITIVITY ANALYSIS (Leave-One-Out) ---
        # Done once, no duplication
        result_array = [current_adv_text] + [
            current_adv_text.replace(word, '', 1).strip() for word in words
        ]
        encoded_inputs = self.tokenizer(result_array, return_tensors="pt", padding=True)
        input_token = encoded_inputs["input_ids"].to(self.device)
        attention_mask = encoded_inputs["attention_mask"].to(self.device)

        trans_res = self.model.generate(
            input_token,
            attention_mask=attention_mask,
            pad_token_id=self.tokenizer.pad_token_id,
            max_length=self.max_len,
            num_beams=self.num_beams,
            output_scores=True, return_dict_in_generate=True,
        )

        seqs = trans_res['sequences'].tolist()
        pred_len = np.array([self.compute_best_len(torch.tensor(seq)) for seq in seqs])
        diffs = np.abs(pred_len - pred_len[0])
        sorted_indices = np.argsort(-diffs)

        # --- 2. REASONING TOKENS ---
        reasoning_map = {
            'RB': ["logically", "therefore", "consequently", "hence", "furthermore", "evidently", "actually"],
            'IN': ["because", "whereas", "since", "despite", "although", "provided"],
            'VB': ["conclude", "reason", "infer", "deduce", "assume", "validate", "verify"],
            'JJ': ["logical", "consistent", "rational", "evident", "causal", "analytical"]
        }
        universal_reasoning = ["so", "thus", "however", "instead"]

        # Pre-cache reasoning candidates for all POS types to avoid recomputing
        reasoning_cache = {
            pos: list(dict.fromkeys(words_list + universal_reasoning))
            for pos, words_list in reasoning_map.items()
        }
        reasoning_cache['default'] = universal_reasoning

        def get_reasoning_candidates(pos_type):
            return reasoning_cache.get(pos_type, reasoning_cache['default'])

        def find_insertion_points(sentence_words, pos_tags):
            insertion_points = []
            n = len(sentence_words)

            # Pre-compute number flags for all positions once
            # instead of recomputing prev/curr/next inside the loop
            is_num_flags = [
                is_number(w) or pos_tags[i][1] == 'CD'
                for i, w in enumerate(sentence_words)
            ]

            for i, (word, tag) in enumerate(pos_tags):
                # Skip if current, previous, or next word is a number
                if is_num_flags[i]:
                    continue
                if i > 0 and is_num_flags[i - 1]:
                    continue
                if i + 1 < n and is_num_flags[i + 1]:
                    continue

                if tag.startswith('VB') and i > 1:
                    insertion_points.append((i, 'RB'))

                if word == ',' and i + 1 < n - 1:
                    insertion_points.append((i + 1, 'IN'))

                if tag.startswith('JJ') and 1 < i < n - 1 and not pos_tags[i - 1][1].startswith('RB'):
                    insertion_points.append((i, 'RB'))

            # Deduplicate while preserving order
            seen = set()
            return [
                seen.add(pt) or pt
                for pt in insertion_points
                if pt not in seen
            ]

        # Pre-compute POS tags for original sentence once
        original_pos_tags = nltk.tag.pos_tag(words)

        # Pre-compute number flags for original words once
        original_num_flags = [
            is_number(w) or original_pos_tags[i][1] == 'CD'
            for i, w in enumerate(words)
        ]

        new_strings = []
        seen_sentences = set()  # Use set for O(1) duplicate checking instead of list

        for max_diff_index in sorted_indices:
            if max_diff_index == 0:
                continue

            word_idx = max_diff_index - 1
            ori_word = words[word_idx]

            # Use pre-computed flag instead of calling is_number again
            if original_num_flags[word_idx]:
                continue

            if len(ori_word) <= 2:
                continue

            target_pos_base = original_pos_tags[word_idx][1][:2]

            # --- 3. BERT CONTEXTUAL SUGGESTIONS ---
            masked_sentence = current_adv_text.replace(ori_word, '[MASK]', 1)
            bert_inputs = self.berttokenizer(masked_sentence, return_tensors="pt").to(self.device)
            mask_indices = torch.where(bert_inputs["input_ids"] == self.berttokenizer.mask_token_id)[1]

            if len(mask_indices) == 0:
                continue
            mask_index = mask_indices[0]

            logits = self.bertmodel(**bert_inputs).logits
            top_tokens = torch.topk(logits[0, mask_index, :], self.num_of_perturb, dim=0).indices.tolist()

            for token_id in top_tokens:
                bert_word = self.berttokenizer.decode([token_id]).strip()
                if not bert_word.isalpha() or bert_word.startswith("##") or is_number(bert_word):
                    continue

                # --- 4. Replace sensitive word with BERT suggestion ---
                bert_replaced_words = list(words)
                bert_replaced_words[word_idx] = bert_word
                bert_replaced_pos_tags = nltk.tag.pos_tag(bert_replaced_words)

                # --- 5. Find insertion points in BERT-replaced sentence ---
                insertion_points = find_insertion_points(bert_replaced_words, bert_replaced_pos_tags)

                for insert_idx, preferred_pos in insertion_points:
                    reasoning_candidates = get_reasoning_candidates(preferred_pos)

                    for r_word in reasoning_candidates:
                        candidate_words = list(bert_replaced_words)
                        candidate_words.insert(insert_idx, r_word)

                        # --- 6. STRUCTURAL VERIFICATION ---
                        candidate_pos_tags = nltk.tag.pos_tag(candidate_words)

                        inserted_tag = candidate_pos_tags[insert_idx][1][:2] if insert_idx < len(
                            candidate_pos_tags) else ''
                        if inserted_tag not in ('RB', 'IN', 'CC', 'JJ', 'VB'):
                            continue

                        new_bert_idx = word_idx if insert_idx > word_idx else word_idx + 1
                        bert_word_tag = candidate_pos_tags[new_bert_idx][1][:2] if new_bert_idx < len(
                            candidate_pos_tags) else ''
                        if not ((bert_word_tag == target_pos_base) or (target_pos_base == 'RB')):
                            continue

                        test_sentence = " ".join(candidate_words)
                        if test_sentence not in seen_sentences:
                            seen_sentences.add(test_sentence)
                            new_strings.append(test_sentence)

            if new_strings:
                return new_strings

        return []

    # @torch.no_grad()
    # def mutation(self, current_adv_text, grad, modify_pos):
    #     import numpy as np
    #     import torch
    #
    #     words = current_adv_text.split()
    #     if len(words) == 0:
    #         return []
    #
    #     # --- 1. TARGET THE WEAKEST WORD ---
    #     # Find which word removal causes the biggest latency drop
    #     result_array = [current_adv_text] + [current_adv_text.replace(word, '', 1).strip() for word in words]
    #
    #     encoded_inputs = self.tokenizer(result_array, return_tensors="pt", padding=True)
    #     input_token = encoded_inputs["input_ids"].to(self.device)
    #     attention_mask = encoded_inputs["attention_mask"].to(self.device)
    #
    #     trans_res = self.model.generate(
    #         input_token,
    #         attention_mask=attention_mask,
    #         pad_token_id=self.tokenizer.pad_token_id,
    #         max_length=self.max_len,
    #         num_beams=self.num_beams,
    #         num_beam_groups=self.num_beam_groups,
    #         output_scores=True, return_dict_in_generate=True,
    #     )
    #
    #     seqs = trans_res['sequences'].tolist()
    #     pred_len = np.array([self.compute_best_len(torch.tensor(seq)) for seq in seqs])
    #
    #     diffs = np.abs(pred_len - pred_len[0])
    #     sorted_indices = np.argsort(-diffs)
    #
    #     # --- 2. USE BERT TO FIND CONTEXTUAL SYNONYMS ---
    #     new_strings = []
    #     for max_diff_index in sorted_indices:
    #         if max_diff_index == 0:
    #             continue  # Skip the baseline sentence
    #
    #         ori_word = words[max_diff_index - 1]
    #
    #         # Skip tiny grammar words (like "a", "is", "to") that BERT struggles to swap cleanly
    #         if len(ori_word) <= 2:
    #             continue
    #
    #         # Mask the target word in plain text! (This bypasses all tokenizer clashes)
    #         masked_sentence = current_adv_text.replace(ori_word, '[MASK]', 1)
    #
    #         # Tokenize specifically for BERT
    #         bert_inputs = self.berttokenizer(masked_sentence, return_tensors="pt").to(self.device)
    #
    #         # Locate where BERT placed the [MASK] token
    #         mask_indices = torch.where(bert_inputs["input_ids"] == self.berttokenizer.mask_token_id)[1]
    #         if len(mask_indices) == 0:
    #             continue
    #         mask_index = mask_indices[0]
    #
    #         # Get BERT's predictions for the blank space
    #         logits = self.bertmodel(**bert_inputs).logits
    #         mask_token_logits = logits[0, mask_index, :]
    #         top_tokens = torch.topk(mask_token_logits, self.num_of_perturb, dim=0).indices.tolist()
    #
    #         # Generate the new candidate sentences
    #         for token_id in top_tokens:
    #             predicted_word = self.berttokenizer.decode([token_id]).strip()
    #
    #             # Clean up BERT subwords (like '##ing') and weird artifacts
    #             if not predicted_word.isalpha() or len(predicted_word) <= 1 or predicted_word.startswith("##"):
    #                 continue
    #
    #             # Safely swap the word using pure strings
    #             candidate = current_adv_text.replace(ori_word, predicted_word, 1)
    #
    #             # Only accept it if the string actually changed
    #             if candidate != current_adv_text:
    #                 new_strings.append(candidate)
    #
    #         # Stop if we successfully generated valid structural swaps
    #         if len(new_strings) > 0:
    #             return new_strings
    #
    #     # Only return empty if every single word failed to mutate
    #     return []

class CharacterAttack(MyAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(CharacterAttack, self).__init__(model, tokenizer, space_token, device, config)
        # self.eos_token_id = self.model.config.eos_token_id
        # self.pad_token_id = self.model.config.pad_token_id
        #
        # self.num_beams = config['num_beams']
        # self.num_beam_groups = config['num_beam_groups']
        # self.max_per = config['max_per']
        # self.embedding = self.model.get_input_embeddings().weight
        # self.softmax = nn.Softmax(dim=1)
        # self.bce_loss = nn.BCELoss()
        # self.specical_token = self.tokenizer.all_special_tokens
        # self.space_token = space_token
        # self.max_len = config['max_len']
        # self.insert_character = string.punctuation
        # self.insert_character += string.digits
        # self.insert_character += string.ascii_letters

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

    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        # loss_list = self.leave_eos_loss(scores, pred_len)
        loss_list = self. leave_eos_target_loss(scores, seqs, pred_len)
        
        return loss_list

    def mutation(self, current_adv_text, grad, modify_pos):
        current_tensor = self.tokenizer([current_adv_text], return_tensors="pt", padding=True).input_ids[0]
        new_strings = self.character_replace_mutation(current_adv_text, current_tensor, grad)
        return new_strings

    @staticmethod
    def transfer(c: str):
        if c in string.ascii_lowercase:
            return c.upper()
        elif c in string.ascii_uppercase:
            return c.lower()
        return c

    def character_replace_mutation(self, current_text, current_tensor, grad):
        important_tensor = (-grad.sum(1)).argsort()
        # current_string = [self.tokenizer.decoder[int(t)] for t in current_tensor]
        new_strings = [current_text]
        for t in important_tensor:
            if int(t) not in current_tensor:
                continue
            ori_decode_token = self.tokenizer.decode([int(t)])
            if self.space_token in ori_decode_token:
                ori_token = ori_decode_token.replace(self.space_token, '')
            else:
                ori_token = ori_decode_token
            if len(ori_token) == 1 or ori_token in self.specical_token:  #todo
                continue
            candidate = [ori_token[:i] + insert + ori_token[i:] for i in range(len(ori_token)) for insert in self.insert_character]
            candidate += [ori_token[:i - 1] + self.transfer(ori_token[i - 1]) + ori_token[i:] for i in range(1, len(ori_token))]

            new_strings += [current_text.replace(ori_token, c, 1) for c in candidate]

            # ori_tensor_pos = current_tensor.eq(int(t)).nonzero()
            #
            # for p in ori_tensor_pos:
            #     new_strings += [current_string[:p] + c + current_string[p + 1:] for c in candidate]
            if len(new_strings) != 0:
                return new_strings
        return new_strings






class WordAttack(MyAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(WordAttack, self).__init__(model, tokenizer, space_token, device, config)

    def mutation(self, current_adv_text, grad, modify_pos):
        new_strings = self.token_replace_mutation(current_adv_text, grad, modify_pos)
        return new_strings

    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        # loss_list = self.leave_eos_loss(scores, pred_len)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        return loss_list

    def token_replace_mutation(self, current_adv_text, grad, modify_pos):
        new_strings = []
        current_tensor = self.tokenizer(current_adv_text, return_tensors="pt", padding=True).input_ids[0]
        base_tensor = current_tensor.clone()
        for pos in modify_pos:
            t = current_tensor[0][pos]
            grad_t = grad[t]
            score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
            index = score.argsort()
            for tgt_t in index:
                if tgt_t not in self.specical_token:
                    base_tensor[pos] = tgt_t
                    break
        current_text = self.tokenizer.decode(current_tensor)
        for pos, t in enumerate(current_tensor):
            if t not in self.specical_id:
                cnt, grad_t = 0, grad[t]
                score = (self.embedding - self.embedding[t]).mm(grad_t.reshape([-1, 1])).reshape([-1])
                index = score.argsort()
                for tgt_t in index:
                    if tgt_t not in self.specical_token:
                        new_base_tensor = base_tensor.clone()
                        new_base_tensor[pos] = tgt_t
                        candidate_s = self.tokenizer.decode(new_base_tensor)
                        # if new_tag[pos][:2] == ori_tag[pos][:2]:
                        new_strings.append(candidate_s)
                        cnt += 1
                        if cnt >= 50:
                            break
        return new_strings


class StructureAttack(MyAttack):
    def __init__(self, model, tokenizer, space_token, device, config):
        super(StructureAttack, self).__init__(model, tokenizer, space_token, device, config)
        # self.tree_tokenizer = TreebankWordTokenizer()
        # self.detokenizer = TreebankWordDetokenizer()
        # BERT initialization
        self.berttokenizer = BertTokenizer.from_pretrained('bert-large-uncased')
        bertmodel = BertForMaskedLM.from_pretrained('bert-large-uncased')
        self.bertmodel = bertmodel.eval().to(self.model.device)
        self.num_of_perturb = 50

    def compute_loss(self, text):
        scores, seqs, pred_len = self.compute_score(text)
        # loss_list = self.leave_eos_loss(scores, pred_len)
        loss_list = self.leave_eos_target_loss(scores, seqs, pred_len)
        return loss_list

    def perturbBert(self, tokens, ori_tensors, masked_indexL, masked_index):
        new_sentences = list()
        # invalidChars = set(string.punctuation)

        # for each idx, use Bert to generate k (i.e., num) candidate tokens
        original_word = tokens[masked_index]

        low_tokens = [x.lower() for x in tokens]
        low_tokens[masked_index] = '[MASK]'
        # try whether all the tokens are in the vocabulary
        try:
            indexed_tokens = self.berttokenizer.convert_tokens_to_ids(low_tokens)
            tokens_tensor = torch.tensor([indexed_tokens])
            tokens_tensor = tokens_tensor.to(self.model.device)
            prediction = self.bertmodel(tokens_tensor)

            # skip the sentences that contain unknown words
            # another option is to mark the unknow words as [MASK]; we skip sentences to reduce fp caused by BERT
        except KeyError as error:
            print('skip a sentence. unknown token is %s' % error)
            return new_sentences

        # get the similar words
        topk_Idx = torch.topk(prediction[0][0, masked_index], self.num_of_perturb)[1].tolist()
        topk_tokens = self.berttokenizer.convert_ids_to_tokens(topk_Idx)

        # remove the tokens that only contains 0 or 1 char (e.g., i, a, s)
        # this step could be further optimized by filtering more tokens (e.g., non-english tokens)
        topk_tokens = list(filter(lambda x: len(x) > 1, topk_tokens))

        # generate similar sentences
        for t in topk_tokens:
            # if any(char in invalidChars for char in t):
            #     continue
            tokens[masked_index] = t
            new_pos_inf = nltk.tag.pos_tag(tokens)

            # only use the similar sentences whose similar token's tag is still the same
            if new_pos_inf[masked_index][1][:2] == masked_indexL[masked_index][1][:2]:
                new_t = self.tokenizer.encode(tokens[masked_index])[0]
                new_tensor = ori_tensors.clone()
                new_tensor[masked_index] = new_t
                new_sentence = self.tokenizer.decode(new_tensor)
                new_sentences.append(new_sentence)
        tokens[masked_index] = original_word
        return new_sentences

    def mutation(self, current_adv_text, grad, modify_pos):
        new_strings = self.structure_mutation(current_adv_text, grad)
        return new_strings

    def get_token_type(self, input_tensor):
        # tokens = self.tree_tokenizer.tokenize(sent)
        tokens = self.tokenizer.convert_ids_to_tokens(input_tensor)
        tokens = [tk.replace(self.space_token, '') for tk in tokens]
        pos_inf = nltk.tag.pos_tag(tokens)
        bert_masked_indexL = list()
        # collect the token index for substitution
        for idx, (word, tag) in enumerate(pos_inf):
            if word in self.tokenizer.all_special_tokens or len(word.strip()) <= 1:
                continue  # Skip the <|begin_of_text|> junk
            # substitute the nouns and adjectives; you could easily substitue more words by modifying the code here
            # if tag.startswith('NN') or tag.startswith('JJ'):
            #     tagFlag = tag[:2]
                # we do not perturb the first and the last token because BERT's performance drops on for those positions
            # if idx != 0 and idx != len(tokens) - 1:
            bert_masked_indexL.append((idx, tag))

        return tokens, bert_masked_indexL

    def structure_mutation(self, current_adv_text, grad):
        new_strings = []
        important_tensor = (-grad.sum(1)).argsort()

        current_tensor = self.tokenizer(current_adv_text, return_tensors="pt", padding=True).input_ids[0]

        ori_tokens, ori_tag = self.get_token_type(current_tensor)
        assert len(ori_tokens) == len(current_tensor)
        assert len(ori_tokens) == len(ori_tag)
        current_tensor_list = current_tensor.tolist()
        for t in important_tensor:
            if int(t) not in current_tensor_list:
                continue
            pos_list = torch.where(current_tensor.eq(int(t)))[0].tolist()
            for pos in pos_list:
                new_string = self.perturbBert(ori_tokens, current_tensor, ori_tag, pos)
                new_strings.extend(new_string)
            if len(new_strings) > 2000:
                break
        return new_strings
