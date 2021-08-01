def norm_data(x, model_mean, model_std):
    if not model_std:
        return x
    return (x - model_mean) / model_std