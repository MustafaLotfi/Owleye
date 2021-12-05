from codes.get_model import Modeling


mdl = Modeling()
mdl.create_boi()
mdl.create_et()
mdl.train_boi(subjects=[1], n_epochs=5, patience=2)
mdl.train_et(subjects=[1], n_epochs=5, patience=2)
