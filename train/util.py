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

    loss_array_val  = []
    closs_array_val = []
    rloss_array_val = []

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
    def plot(self,  exp_name_dir,
                    step   = steps_array,
                    train_loss   = loss_array,
                    train_closs  = closs_array,
                    train_rloss  = rloss_array,

                    val_loss   = loss_array_val,
                    val_closs  = closs_array_val,
                    val_rloss  = rloss_array_val,

                    title  = "Siamese RPN",
                    ylabel = 'error',
                    xlabel = 'epoch',
                    show   = False):
        plt.plot(step, train_loss, 'r', label='train loss', color='blue')
        plt.plot(step, train_closs, 'r', label='train closs', color='red')
        plt.plot(step, train_rloss, 'r', label='train rloss', color='black')

        plt.plot(step, val_loss, 'k--', label='val loss', color='blue')
        plt.plot(step, val_closs, 'k--', label='val closs', color='red')
        plt.plot(step, val_rloss, 'k--', label='val rloss', color='black')

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
