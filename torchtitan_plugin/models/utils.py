BASE_DIM = 256


def get_glu_dim(x):
    return convert_to_multiple_of_base(int(8 * x / 3))


def convert_to_multiple_of_base(x):
    return BASE_DIM * ((x + BASE_DIM - 1) // BASE_DIM)
