import os, torch, json
from torch.utils.data import DataLoader


class MyDataset(torch.utils.data.Dataset):
    def __init__(self, filepath):
        self.questions = []
        self.answers = []
        with open('datasets/ambignq/train_light.json', 'r') as f:
            json_data = json.load(f)
            for data in json_data:
                try:
                    qaPairs = data['annotations'][0]['qaPairs']
                    for qaPair in qaPairs:
                        question = qaPair['question']
                        answer = qaPair['answer'][0]
                        self.questions.append(question)
                        self.answers.append(answer)
                except Exception as e:
                    continue
        self.data_len = len(self.questions)

    def __getitem__(self, index):
        question = self.questions[index]
        answer = self.answers[index]
        return {'question': question, 'answer': answer}

    def __len__(self):
        return self.data_len


def test():
    dataset = MyDataset('datasets/training/ambignq.txt')
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for _, batch in enumerate(dataloader):
        print(batch['answer'])


if __name__ == '__main__':
    test()