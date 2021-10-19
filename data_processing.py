import json


def main():
    training_file = open('datasets/training/ambignq.txt', 'w')
    with open('datasets/ambignq/train_light.json', 'r') as f:
        json_data = json.load(f)
        for data in json_data:
            try:
                qaPairs = data['annotations'][0]['qaPairs']
                for qaPair in qaPairs:
                    question = qaPair['question']
                    answer = qaPair['answer'][0]
                    if answer == '' or question == '':
                        continue
                    training_file.write(question)
                    training_file.write('\n')
                    training_file.write(answer)
                    training_file.write('\n')
            except Exception as e:
                continue
    training_file.close()


if __name__ == '__main__':
    main()