from utils import *
from Dataset_Reader import read_dataset
import json
datasets_dir = "all_datasets/" 
dataset_namelist = ['bluebird', 
                    'flowers', 
                    'web', 
                    'dog', 
                    'rte', 
                    'SP', 
                    'SP_amt', 
                    'ZenCrowd_all',
                    'ZenCrowd_in',
                    'ZenCrowd_us',
                    'face',
                    'product',
                    'sentiment']


def random_generator(num_tsk, proportion_to_train):
    num_tsk_train = int(round(num_tsk * proportion_to_train))
    tsk_id = list(range(num_tsk))
    train_task_id = random.sample(tsk_id, num_tsk_train)
    return train_task_id


def generate_train_data(dataset_id, proportion_to_train):
    _, _, _, _, _, num_tsk, true_labels, name, _ = read_dataset(dataset_id)
    train_id = random_generator(num_tsk, proportion_to_train)
    train_gt = []
    for tsk_id in train_id:
        train_gt.append(true_labels[tsk_id])

    train_id = np.array(train_id)
    train_gt = np.array(train_gt)
    return train_id, train_gt

def generate_train_data_files(training_proportion):
    for dataset_id in range(1, 14):
        dataset_name = dataset_namelist[dataset_id - 1]
        train_dir = datasets_dir + 'train_proportion_' + str(training_proportion) + '/' + dataset_name + '_train_id.npz'
        train_id, train_gt = generate_train_data(dataset_id, training_proportion)

        np.savez(train_dir, train_id=train_id, train_gt=train_gt)


def get_train_data_from_files(dataset_name, proportion=0.1):
    train_dir = datasets_dir + 'train_proportion_' + str(proportion) + '/' + dataset_name + '_train_id.npz'
    data = np.load(train_dir)
    train_id = data['train_id']
    train_gt = data['train_gt']
    return train_id, train_gt



def get_cross_val_train_data_from_files(dataset_id, train_proportion):
    name = dataset_namelist[dataset_id-1]
    dir_ = datasets_dir + "cross-val/" + 'train_proportion_' + str(train_proportion) + '/'
    path = dir_ + name + '_cross_val_train_idx.txt'
    with open(path, "r", encoding='UTF-8') as f:
        train_data = f.read()
        train_data = json.loads(train_data)

    return train_data

if __name__ == '__main__':
    pass






