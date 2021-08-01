import argparse
import random
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from Printer import Printer
from Reader import Reader
from const import EPSILON, PARAMS_FILEPATH, PARAMS_HEADER


def mean_lst(lst):
    return float(sum(lst)) / len(lst)


def std_lst(lst):
    variance = sum([((x - mean_lst(lst)) ** 2) for x in lst]) / len(lst)
    stddev = variance ** 0.5
    return stddev


def norm_data(lst):
    new_lst = []
    for elem in lst:
        new_lst.append((elem - mean_lst(lst)) / std_lst(lst))
    return new_lst


def mse(y, y_pred):
    loss_lst = []
    for i in range(len(y)):
        loss_lst.append((y_pred[i] - y[i]) ** 2)
    return mean_lst(loss_lst)


def predict_lst(x, theta0, theta1):
    y_pred = []
    for i in range(len(x)):
        y_pred.append(theta0 + (theta1 * x[i]))
    return y_pred


def update_thetas(x, y, lr, theta0, theta1):
    a = []
    b = []
    for i in range(len(x)):
        a.append(theta0 + (theta1 * x[i]) - y[i])
        b.append((theta0 + (theta1 * x[i]) - y[i]) * x[i])
    new_theta0 = theta0 - lr * mean_lst(a)
    new_theta1 = theta1 - lr * mean_lst(b)
    return new_theta0, new_theta1


def save_params(x, theta0, theta1):
    try:
        with open(PARAMS_FILEPATH, 'w') as f:
            f.write(f'{PARAMS_HEADER}\n')
            f.write(f'{str(theta0)},{str(theta1)},{str(mean_lst(x))},{str(std_lst(x))}')
    except IOError:
        error("Cant open {} file.".format(param_file))


def r_squared(y, y_pred):
    ss_tot = []
    ss_res = []
    mean_y = mean_lst(y)
    for i in range(len(y)):
        ss_tot.append((y[i] - mean_y)**2)
        ss_res.append((y[i] - y_pred[i])**2)
    return 1 - (sum(ss_res)/sum(ss_tot))


def plot(x, y, y_pred):
    sns.set_style('white')
    sns.scatterplot(x=x, y=y, label='Data')
    plt.plot(x, y_pred, color='red', label='Linear Regression')
    plt.legend(loc='best')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.savefig('plot.png')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to data file')
    parser.add_argument('-plot', help='show plot', action='store_true')
    parser.add_argument('-r2', help='show R2 metric', action='store_true')
    parser.add_argument('-epochs', help='training iterations number', default=1000, type=int)
    parser.add_argument('-lr', help='learning rate (step) value', default=0.1, type=float)
    parser.add_argument('-verbose', help='verbose train steps', action='store_true')
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
    return args


def train(km_list, prices, lr, verbose, epochs):
    theta0 = random.random()
    theta1 = random.random()
    km_list = norm_data(km_list)
    loss = mse(prices, predict_lst(km_list, theta0, theta1))
    epoch, prev_loss  = 0, 0
    while abs(loss - prev_loss) > EPSILON:
        epoch += 1
        theta0, theta1 = update_thetas(km_list, prices, lr, theta0, theta1)
        prev_loss, loss = loss, mse(prices, predict_lst(km_list, theta0, theta1))
        if verbose:
            print(f'epoch: {epoch:3}, loss: {loss:20}, learning rate(step): {lr:3.10f}')
        if epoch == epochs:
            break
    if verbose:
        print(f'Model trained {epoch} epochs, theta0 - {theta0}, theta1 - {theta1}')
    return theta0, theta1


def main():
    args = parse_args()
    km_list, prices = Reader.read_data_file(args.path)
    if (args.lr > 1) | (args.lr < 0):
        Printer.print_error_exit('invalid parameter value: -lr: learning rate must be in range (0;1)')
    if args.epochs <= 0:
        Printer.print_error_exit('invalid parameter value: -epochs: must be positive')
    theta0, theta1 = train(km_list, prices, args.lr, args.verbose, args.epochs)
    save_params(km_list, theta0, theta1)
    if args.plot:
        plot(km_list, prices, predict_lst(norm_data(km_list), theta0, theta1))
        print('Graph save to plot.png')
    if args.r2:
        print('R2 metric -', r_squared(prices, predict_lst(norm_data(km_list), theta0, theta1)))


if __name__ == '__main__':
    main()