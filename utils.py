import pandas as pd
import os

def make_submission(y_pred_lunch, y_pred_dinner, datadir):
    submission = pd.read_csv(os.path.join(datadir,f'sample_submission.csv'))

    submission["중식계"] = y_pred_lunch
    submission["석식계"] = y_pred_dinner

    return submission

def version_update(logdir):
    # make logs folder
    savedir = logdir
    if not os.path.isdir(savedir):
        os.mkdir(savedir)

    # check save version
    version = len(os.listdir(savedir))

    # make save folder
    savedir = os.path.join(savedir, f'version{version}')
    os.mkdir(savedir)

    print(f'Version {version}')

    return savedir



