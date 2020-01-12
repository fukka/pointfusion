import glob
from matplotlib import pyplot as plt

if __name__ == '__main__':
    l = glob.glob(r'/home/fengjia/pointfusion/trained_model/2020_01_09__1/*')
    epoch_id = []
    training_loss = []
    for f in l:
        epoch_id.append(int(f.split(':')[1].split('_')[0]))
        training_loss.append(float(f.split(':')[2]))
    epoch_id, training_loss = zip(*sorted(zip(epoch_id, training_loss)))
    plt.plot(training_loss)
    plt.show()
