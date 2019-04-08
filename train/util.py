import matplotlib.pyplot as plt

class Util(object):

    def data_split(seq_datasetVID, seq_datasetGOT):
        seq_dataset = []
        for i in seq_datasetVID:
            seq_dataset.append(i)

        for i, data in enumerate(seq_datasetGOT):
            seq_dataset.append(data)
            if i >= 8600:
                break
        return seq_dataset

class AverageMeter(object):
    '''Computes and stores the average and current value'''
    def __init__(self):
        self.reset()
        self.plot_c = 0

    steps_array = []
    loss_array  = []
    closs_array = []
    rloss_array = []

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    '''setup plot'''
    def plot(self, exp_name_dir, step = steps_array, loss = loss_array, closs = closs_array, rloss = rloss_array, title = "Siamese RPN", ylabel = 'error',  xlabel = 'epoch', show=False):
        plt.plot(step, loss, 'r', label='loss', color='blue')
        plt.plot(step, closs, 'r', label='closs', color='red')
        plt.plot(step, rloss, 'r', label='rloss', color='black')
        plt.title(title)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        if self.plot_c == 0:
            plt.legend()
            self.plot_c +=1

        '''save plot'''
        plt.savefig("{}/test.png".format(exp_name_dir))
        if show:
            plt.show()

    def experiment_name_dir(experiment_name):
        experiment_name_dir = 'experiments/{}'.format(experiment_name)
        if experiment_name == 'default':
            print('You are using "default" experiment, my advice to you is: Copy "default" change folder name and change settings in file "parameters.json"')
        else:
            print('You are using "{}" experiment'.format(experiment_name))
        return experiment_name_dir
