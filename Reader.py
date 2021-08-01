import os
import csv
from Printer import Printer


class Reader:
    @staticmethod
    def read_data_file(path):
        if os.path.exists(path):
            km_list = []
            prices = []
            rows_counter = 1
            try:
                with open(path, 'r') as f:
                    rows = [row for row in csv.DictReader(f)]
                    for column_name in ['km', 'price']:
                        if column_name not in rows[0]:
                            Printer.print_error_exit(f"no column {column_name} in file {path}")
                    for row in rows:
                        rows_counter += 1
                        km_list.append(int(row['km']))
                        prices.append(int(row['price']))
                return km_list, prices
            except IOError:
                Printer.print_error_exit(f"can't open file {path}")
            except ValueError:
                Printer.print_error_exit(f"invalid values in file {path}")
            except TypeError:
                Printer.print_error_exit(f"Not enough data in file {path} at row {rows_counter}")

        else:
            Printer.print_error_exit('invalid path to data file')


    @staticmethod
    def read_params_file(path):
        if os.path.exists(path):
            theta0_name = 'theta0'
            theta1_name = 'theta1'
            data_mean_name = 'data_mean'
            data_sd_name = 'data_sd'
            try:
                with open(path, 'r') as f:
                    rows = [row for row in csv.DictReader(f)]
                    if len(rows) == 0:
                        Printer.print_error_exit(f"must be at least 2 rows in file {path}")
                    row = rows[0]
                    for column_name in [theta0_name, theta1_name, data_mean_name, data_sd_name]:
                        if column_name not in row:
                            Printer.print_error_exit(f"no parameter {column_name} in file {path}")
                    theta0 = float(row[theta0_name])
                    theta1 = float(row[theta1_name])
                    model_mean = float(row[data_mean_name])
                    model_std = float(row[data_sd_name])
                return theta0, theta1, model_mean, model_std
            except IOError:
                Printer.print_error_exit(f"can't open file {path}")
            except ValueError:
                Printer.print_error_exit(f"invalid values in file {path}")
            except TypeError:
                Printer.print_error_exit(f"Not enough data in file {path}")
        else:
            Printer.print_error_exit('invalid path to params file')