from got10k.experiments import *

from net import TrackerSiamRPN


if __name__ == '__main__':

    '''setup tracker'''
    net_path = '../train/experiments/default/model/model_e4.pth'
    tracker = TrackerSiamRPN(net_path=net_path)

    '''setup experiments'''
    experiments = ExperimentGOT10k('/Users/arbi/Desktop', subset='val', result_dir='results', report_dir='reports')
    #experiments = ExperimentOTB('data/OTB', version=2015, result_dir='resultsOTB', report_dir='reportsOTB')
    #experiments = ExperimentVOT('../data/vot2018', version=2018, result_dir='../results_two', report_dir='../reports_two')

    '''run tracking experiments and report performance'''
    experiments.run(tracker, visualize=True)
    experiments.report([tracker.name])
