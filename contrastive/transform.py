class SelectionSequentialTransform(object):
	def __init__(self, tokenizer, max_len):
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __call__(self, texts):
		input_ids_list, segment_ids_list, input_masks_list, contexts_masks_list = [], [], [], []
		for text in texts:
			tokenized_dict = self.tokenizer.encode_plus(text, max_length=self.max_len, pad_to_max_length=True)
			input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
			assert len(input_ids) == self.max_len
			assert len(input_masks) == self.max_len
			input_ids_list.append(input_ids)
			input_masks_list.append(input_masks)

		return input_ids_list, input_masks_list


class SelectionJoinTransform(object):
	def __init__(self, tokenizer, max_len):
		self.tokenizer = tokenizer
		self.max_len = max_len

	def __call__(self, text):
		tokenized_dict = self.tokenizer.encode_plus(text, max_length=self.max_len, pad_to_max_length=True)
		input_ids, input_masks = tokenized_dict['input_ids'], tokenized_dict['attention_mask']
		assert len(input_ids) == self.max_len
		assert len(input_masks) == self.max_len

		return input_ids, input_masks
	

class SelectionConcatTransform(object):
	def __init__(self, tokenizer, max_response_len, max_contexts_len):
		self.tokenizer = tokenizer
		self.max_response_len = max_response_len
		self.max_contexts_len = max_contexts_len
		self.max_len = max_response_len + max_contexts_len

	def __call__(self, context, responses):
		tokenized_dict = self.tokenizer.encode_plus(context, max_length=self.max_contexts_len)
		context_ids, context_masks, context_segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']
		ret_input_ids = []
		ret_input_masks = []
		ret_segment_ids = []
		for response in responses:
			tokenized_dict = self.tokenizer.encode_plus(response, max_length=self.max_response_len+1)
			response_ids, response_masks, response_segment_ids = tokenized_dict['input_ids'], tokenized_dict['attention_mask'], tokenized_dict['token_type_ids']
			input_ids = context_ids + response_ids[1:]
			input_masks = context_masks + response_masks[1:]
			input_segment_ids = context_segment_ids + [1]*(len(response_segment_ids)-1)
			input_ids += [0] * (self.max_len - len(input_ids))
			input_masks += [0] * (self.max_len - len(input_masks))
			input_segment_ids += [0] * (self.max_len - len(input_segment_ids))
			assert len(input_ids) == self.max_len
			assert len(input_masks) == self.max_len
			assert len(input_segment_ids) == self.max_len
			ret_input_ids.append(input_ids)
			ret_input_masks.append(input_masks)
			ret_segment_ids.append(input_segment_ids)
		
		return ret_input_ids, ret_input_masks, ret_segment_ids
