import argparse
from Printer import Printer
from Reader import Reader
from calc import norm_data
from const import GREEN, RESET


def predict(theta0, theta1, x, model_mean, model_std):
    return theta0 + (theta1 * norm_data(x, model_mean, model_std))


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='path', help='path to params.csv')
    parser.add_argument(dest='mileage', help='mileage for price prediction', type=int)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    theta0, theta1, model_mean, model_std = Reader.read_params_file(args.path)
    if not theta0 and not theta1:
        Printer.print_error_exit('model is not trained')
    if args.mileage < 0:
        Printer.print_error_exit('mileage must not be negative')
    predicted_price = predict(theta0, theta1, args.mileage, model_mean, model_std)
    print(f"{GREEN}Predicted price for car with mileage = {args.mileage}km is {predicted_price}{RESET}")


if __name__ == '__main__':
    main()
