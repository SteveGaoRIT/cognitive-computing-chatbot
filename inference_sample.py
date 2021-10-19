import torch, os, pickle
from transformers import T5Tokenizer, T5ForConditionalGeneration


def main():
    # initialize the tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    # initialize the T5 model
    model = T5ForConditionalGeneration.from_pretrained("t5-base").cuda()
    # create the directory to save pre-trained model and the configuration of the model.
    pre_train_model_dir = 'pre_train'
    if not os.path.exists(pre_train_model_dir):
        os.makedirs(pre_train_model_dir)
    if not os.path.exists(os.path.join(pre_train_model_dir, 'pre_train_T5_config')):
        with open(os.path.join(pre_train_model_dir, 'pre_train_T5_config'), 'wb') as f:
            pickle.dump(model.config, f)
    if not os.path.exists(os.path.join(pre_train_model_dir, 'pre_train_T5')):
        torch.save(model.state_dict(), os.path.join(pre_train_model_dir, 'pre_train_T5'))
    # test different approaches to generate an output.
    input_ids = tokenizer('translate English to German: The house is wonderful', return_tensors='pt').input_ids.cuda()
    decoder_ids = tokenizer('Das Haus ist wunderbar', return_tensors='pt').input_ids.cuda()
    outputs = model.generate(input_ids=input_ids)
    print(tokenizer.decode(outputs[0], skip_special_tokens=True))
    outputs = model(input_ids=input_ids, labels=decoder_ids)
    print(tokenizer.decode(outputs.logits[0].argmax(dim=-1)))


if __name__ == '__main__':
    main()
# Das Haus ist wunderbar.