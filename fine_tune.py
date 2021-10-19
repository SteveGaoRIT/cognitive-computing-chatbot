import torch, os, pickle
from dataset import MyDataset
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from transformers import T5Tokenizer, T5ForConditionalGeneration
import tensorboardX


def main():
    # create the directory to save fine-tuned model.
    fine_tune_model_dir = 'fine_tune'
    if not os.path.exists(fine_tune_model_dir):
        os.makedirs(fine_tune_model_dir)
    # initialize the tokenizer
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    with open(os.path.join('pre_train', 'pre_train_T5_config'), 'rb') as f:
        config = pickle.load(f)
    # initialize T5 model and load parameters from fine-tuned model or pre-trained model.
    model = T5ForConditionalGeneration(config=config).cuda()
    if os.path.exists(os.path.join('fine_tune', 'fine_tune_T5')):
        model.load_state_dict(torch.load(os.path.join('fine_tune', 'fine_tune_T5')))
    else:
        model.load_state_dict(torch.load(os.path.join('pre_train', 'pre_train_T5')))
    # initialize the optimizer, I used AdamW here.
    optimizer = AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.98))
    # learning rate scheduler, I did not warm up the model.
    scheduler = CosineAnnealingLR(optimizer, T_max=19, eta_min=1e-5)
    # training parameters here
    epochs = 20
    batch_size = 4
    accumulation_batch = 32
    accumulation_steps = accumulation_batch // batch_size
    # initialize dataset and dataloader here
    dataset = MyDataset('datasets/training/ambignq.txt')
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    # initialize tensorboard here
    steps = 0
    writer = tensorboardX.SummaryWriter()
    # task prefix, which is used when pre-training T5, and I keep this prefix in my fine-tuning stage.
    task_prefix = "question: "
    # train the model here
    for i in range(epochs):
        print(f"epoch: {i}")
        cur_lr = optimizer.param_groups[0]['lr']
        print(f"learning rate: {cur_lr}")
        for _, batch in enumerate(dataloader):
            steps += 1
            # tokenizer the input and target
            encoding = tokenizer([task_prefix + sequence for sequence in batch['question']],
                                  padding='longest',
                                  max_length=512,
                                  truncation=True,
                                  return_tensors='pt')
            input_ids, attention_mask = encoding.input_ids.cuda(), encoding.attention_mask.cuda()

            target_encoding = tokenizer(batch['answer'],
                                        padding='longest',
                                        max_length=128,
                                        truncation=True)
            labels = target_encoding.input_ids
            # make sure pad_tokens are ignored when calculating loss
            labels = [
                [(label if label != tokenizer.pad_token_id else -100) for label in labels_example] for labels_example in
                labels
            ]
            labels = torch.tensor(labels).cuda()
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            # cause I use accumulation gradients here, I need to scale the loss.
            loss = loss / accumulation_steps
            loss.backward()
            if (_ + 1) % accumulation_steps == 0:
                # I'd like to store the loss value each time I update T5's parameters
                optimizer.step()
                optimizer.zero_grad()
                print(loss * accumulation_steps)
                print(tokenizer.decode(input_ids[0], skip_special_tokens=True))
                print('Inference:')
                print(tokenizer.decode(model.generate(input_ids=input_ids[0].unsqueeze(0))[0], skip_special_tokens=True))
                print('Truth:')
                print(batch['answer'][0])
                print()
                writer.add_scalar("pretrain_loss", loss * accumulation_steps, steps)
        optimizer.step()
        optimizer.zero_grad()
        scheduler.step()
        torch.save(model.state_dict(), os.path.join(fine_tune_model_dir, "fine_tune_T5"))
    writer.close()


if __name__ == '__main__':
    main()