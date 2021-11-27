from codes import see_data


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 11

TARGET_FOLDER = "sampling-test"  # et-clb, boi, sampling or sampling-test
# see_data.features(NUMBER, TARGET_FOLDER)
# see_data.pixels(NUMBER, "y-hat-et", 2, False)
see_data.pixels_test(NUMBER, "y-hat-et", 2, False)
