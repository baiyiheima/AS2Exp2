from transformers import BertTokenizer, BertModel
if __name__ == '__main__':
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    model = BertModel.from_pretrained("bert-base-cased")

    text = "replace me by any text you'd like.","replace me by any text you'd like."
    batch_tokenized = tokenizer.batch_encode_plus(text, add_special_tokens=True,
                                                  max_length=100, padding='max_length',
                                                  truncation=True)  # tokenize、add special token、pad
    encoded_input = tokenizer(text)
    print(batch_tokenized)
    print(encoded_input)