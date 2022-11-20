# Python vision: 3.8.12   torch vision: 1.10.2
# from torch.utils.data import Dataset
from torch.utils.data import Sampler
from typing import Iterator, Sized
from read_data import *
from functions import *
from net_pack import *

# Hyper parameter
EPOCH = 300
BATCH_SIZE = 6  # a batch contains 6 * 10 mins
step = 10  # slide step of timewindow
pre = 15

class newsampler(Sampler[int]):
    r"""Samples elements sequentially, always in the same order.

    Args:
        data_source (Dataset): dataset to sample from
    """
    data_source: Sized

    def __init__(self, data_source: Sized) -> None:
        self.data_source = data_source

    def __iter__(self) -> Iterator[int]:
        self.t = []
        for i in range(0, len(self.data_source), step):
            self.t.append(i)
        return iter(self.t)

    def __len__(self) -> int:
        return len(self.data_source)


def multi_train_gpu(cuda, room_list, date_list):
    for cross in range(len(date_list)):  # cross validation
        copy = date_list[:]
        copy.pop(cross)
        train_date = copy
        test_date = date_list[cross]
        test_date = [test_date]
        dataset = MyDataset(train_date, '', length=6600, mode=2, label_start=30, label_end=30+pre, read_type='new')
        test_dataset = MyDataset(test_date, '', length=600, mode=2, label_start=30, label_end=30+pre, read_type='new')
        for r in room_list:
            temp_dataset = MyDataset(train_date, r, label_start=30, label_end=30+pre, read_type='new')
            temp_test_dataset = MyDataset(test_date, r, label_start=30, label_end=30+pre, read_type='new')
            dataset.data = torch.cat((dataset.data, temp_dataset.data), 1)
            dataset.labels = torch.cat((dataset.labels, temp_dataset.labels), 1)
            test_dataset.data = torch.cat((test_dataset.data, temp_test_dataset.data), 1)
            test_dataset.labels = torch.cat((test_dataset.labels, temp_test_dataset.labels), 1)
        dataset.data = dataset.data[:, 1:].to(device=cuda)
        dataset.labels = dataset.labels[:, 1:].to(device=cuda)
        test_dataset.data = test_dataset.data[:, 1:].to(device=cuda)
        test_dataset.labels = test_dataset.labels[:, 1:].to(device=cuda)
        data_loader = DataLoader(dataset, batch_size=BATCH_SIZE, sampler=newsampler(dataset))
        test_data_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, sampler=newsampler(test_dataset))

        enc = Encoder(cuda=cuda)
        dec = Decoder()
        model = seq2seq(enc, dec, cuda=cuda)
        model.to(device=cuda)

        # training and testing
        record = []
        LR = 0.00003
        if cross >= 0:  # Control start date
            for epoch in range(EPOCH):
                # if epoch < 50:
                #     LR = 0.000005
                # elif epoch % 50 == 0:
                #     LR = LR * 0.5
                LR *= 0.99
                bd = 0.05

                optimizer = torch.optim.Adam(model.parameters(), lr=LR)  # , weight_decay= 0.01

                for iteration, (train_x, train_y) in enumerate(data_loader):
                    output = model(train_x, train_y, pre, 'Hadamard', t_mode='train')
                    loss = BATCH_CVRMSE(output, train_y)
                    optimizer.zero_grad()  # clear gradients for this training step
                    loss.backward()  # backpropagation, compute gradients
                    optimizer.step()  # apply gradients

                    if iteration == 109:
                        for it, (test_x, test_y) in enumerate(test_data_loader):
                            test_output = model(test_x, train_y, pre, 'Hadamard', t_mode='train')
                            # print(test_output - test_y)
                            eva = evaluation(test_output, test_y, bd)  # bound = ±set_bound/(indoor_upper-indoor_lower)℃
                            l = [test_date[0], LR, epoch, iteration, "{:<6.4f}".format(loss), eva]
                            record.append(l)
                            print(
                                'test_date:{:<2s} | LR:{:<.8f} | epoch:{:<3d} | iteration:{:<3d} | loss:{:<6.4f} | Evaluation:{:<4.2f}'.format(
                                    test_date[0], LR, epoch, iteration, loss, eva))

            array = np.array(record)
            df = pd.DataFrame(array)
            record_name = "Hadamard_pre_15"
            if not os.path.exists('./Records/' + record_name):
                os.makedirs('./Records/' + record_name)
                os.makedirs('./Models/' + record_name)
            df.to_csv('./Records/' + record_name + '/' + test_date[0] + '.csv', index=False,
                      header=['test date', 'learning rate', 'epoch', 'iteration', 'loss', 'evaluation'])
            torch.save(model, './Models/' + record_name + '/' + 'tested_' + test_date[0] + '.pt')


if __name__ == "__main__":
    # Load data
    d_list = ['2021-08-09', '2021-08-10', '2021-08-11', '2021-08-12', '2021-08-13', '2021-08-14', \
              '2021-08-16', '2021-08-17', '2021-08-18', '2021-08-19', '2021-08-20', '2021-08-21']
    r_list = ['room_1.pt', 'room_2.pt', 'room_4.pt', 'room_5.pt', 'room_6.pt', 'room_7.pt']
    if torch.cuda.is_available():
        cuda = torch.device('cuda:0')
        print("Device = cuda:0")
    else:
        cuda = torch.device('cpu')
        print("Device = cpu")
    seed_torch()
    multi_train_gpu(cuda, r_list, d_list)

