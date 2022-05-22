def norm_data(x, model_mean, model_std):
    if not model_std:
        return x
    return (x - model_mean) / model_std


def mean_lst(lst):
    return float(sum(lst)) / len(lst)


def r_squared(y, y_pred):
    ss_tot = []
    ss_res = []
    mean_y = mean_lst(y)
    for i in range(len(y)):
        ss_tot.append((y[i] - mean_y)**2)
        ss_res.append((y[i] - y_pred[i])**2)
    return 1 - (sum(ss_res)/sum(ss_tot))


def r_squared(y, y_pred):
    ss_tot = []
    ss_res = []
    mean_y = mean_lst(y)
    for i in range(len(y)):
        ss_tot.append((y[i] - mean_y)**2)
        ss_res.append((y[i] - y_pred[i])**2)
    return 1 - (sum(ss_res)/sum(ss_tot))


def std_lst(lst):
    variance = sum([((x - mean_lst(lst)) ** 2) for x in lst]) / len(lst)
    stddev = variance ** 0.5
    return stddev


def norm_data_lst(lst):
    new_lst = []
    std = std_lst(lst)
    if not std:
        std = 1
    for elem in lst:
        new_lst.append((elem - mean_lst(lst)) / std)
    return new_lst


def mse(y, y_pred):
    loss_lst = []
    for i in range(len(y)):
        loss_lst.append((y_pred[i] - y[i]) ** 2)
    return mean_lst(loss_lst)
