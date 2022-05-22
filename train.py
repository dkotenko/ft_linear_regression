import argparse
import random
import sys
import matplotlib.pyplot as plt
from Printer import Printer
from Reader import Reader
from const import EPSILON, PARAMS_FILEPATH, PARAMS_HEADER, GREEN, RESET
from calc import *


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
        Printer.print_error_exit(f"Cant open {PARAMS_FILEPATH} file.")


def draw_plot(x, y, y_pred):
    plt.figure(figsize=(5, 5))
    plt.grid(True)
    plt.scatter(x=x, y=y, label='Data', color='red')
    plt.plot(x, y_pred, label='Linear Regression')
    plt.legend(loc='best')
    plt.xlabel('km')
    plt.ylabel('price')
    plt.gca().set_axisbelow(True)
    plt.show()


def draw_errors(errors):
    plt.plot([i for i in range(len(errors))], errors, label='Model loss')
    plt.legend(loc='best')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('path', help='path to data file')
    parser.add_argument('-errors', help='show plot', action='store_true')
    parser.add_argument('-plot', help='show plot', action='store_true')
    parser.add_argument('-r2', help='show R2 metric', action='store_true')
    parser.add_argument('-epochs', help='training iterations number', default=1000, type=int)
    parser.add_argument('-lr', help='learning rate (step) value', default=0.1, type=float)
    parser.add_argument('-verbose', help='verbose train steps', action='store_true')
    args = parser.parse_args()
    if len(sys.argv) < 2:
        parser.print_help()
    return args


def get_updated_loss(km_list, theta0, theta1, prices, errors):
    predicted = predict_lst(km_list, theta0, theta1)
    mse_val = mse(prices, predicted)
    errors.append(mse_val)
    return mse_val


def train(km_list, prices, lr, verbose, epochs):
    theta0 = random.random()
    theta1 = random.random()
    km_list = norm_data_lst(km_list)
    errors = []
    loss = get_updated_loss(km_list, theta0, theta1, prices, errors)
    epoch, prev_loss = 0, 0
    while abs(loss - prev_loss) > EPSILON:
        epoch += 1
        theta0, theta1 = update_thetas(km_list, prices, lr, theta0, theta1)
        prev_loss, loss = loss, get_updated_loss(km_list, theta0, theta1, prices, errors)
        if verbose:
            print(f'epoch: {epoch:3}, loss: {loss:20}, learning rate(step): {lr:3.10f}')
        if epoch == epochs:
            break
    if verbose:
        print(f'Model trained {epoch} epochs, theta0 - {theta0}, theta1 - {theta1}')
    return theta0, theta1, errors


def main():
    args = parse_args()
    km_list, prices = Reader.read_data_file(args.path)
    if (args.lr > 1) | (args.lr < 0):
        Printer.print_error_exit('invalid parameter value: -lr: learning rate must be in range (0;1)')
    if args.epochs <= 0:
        Printer.print_error_exit('invalid parameter value: -epochs: must be positive')
    theta0, theta1, errors = train(km_list, prices, args.lr, args.verbose, args.epochs)
    save_params(km_list, theta0, theta1)
    if args.plot or args.r2:
        predicted_values = predict_lst(norm_data_lst(km_list), theta0, theta1)
        if args.plot:
            draw_plot(km_list, prices, predicted_values)
        if args.r2:
            print('R2 metric -', r_squared(prices, predicted_values))
    if args.errors:
        draw_errors(errors)
    print(f"{GREEN}Model has been trained{RESET}")


if __name__ == '__main__':
    main()