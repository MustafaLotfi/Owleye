from codes.see_data import See


# ----------- PARAMETERS ------------
# Subject Information
NUMBER = 14

TARGET_FOLDER = "sampling-test"  # et-clb, boi, sampling or sampling-test
# see_data.features(NUMBER, TARGET_FOLDER)
# see_data.pixels(NUMBER, "y-hat-et", 2, False)
# see_data.pixels_test(NUMBER, "y-hat-et", 2, False)
see = See(NUMBER)
see.features(TARGET_FOLDER)
# see.pixels(testing=True)
see.pixels_test()