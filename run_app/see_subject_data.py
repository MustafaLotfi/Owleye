from codes import see_data


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 12

TARGET_FOLDER = "et-clb"  # et-clb, boi, sampling or sampling-test
# see_data.features(NUMBER, TARGET_FOLDER)
# see_data.pixels(NUMBER, "y-hat-et")
see_data.pixels_test(NUMBER, "y-hat-et-flt")
