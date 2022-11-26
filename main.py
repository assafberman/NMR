import numpy
import pandas as pd
from model import initialize_model, cosine_similarity, import_pre_trained
# from database import import_database_as_df
from database_carbon import import_database_as_df

pd.set_option('display.max_columns', None)
import os.path
from datetime import datetime


def prompt_message(message_str: str):
    print('{}:  {}'.format(datetime.now().strftime('%d-%m-%Y %H:%M:%S'), message_str))


def train_new_model(x_train, y_train, input_size=256, output_size=512, epochs=10, batch_size=32):
    prompt_message('Model initialization started.')
    new_model = initialize_model(input_size, output_size)
    prompt_message('Model initialized successfuly.')
    prompt_message('Model fitting started.')
    new_model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size)
    prompt_message('Model fitted successfuly.')
    return new_model


"""
if (os.path.exists('./carbon_nmr.csv')):
    nmr_df = pd.read_csv('./carbon_nmr.csv', sep=';')
else:
    nmr_df = import_database_as_df()
    nmr_df.to_csv('./carbon_nmr.csv', sep=';')
"""

prompt_message('Importing database.')
nmr_df = import_database_as_df()
input_list = [x for x in nmr_df['Input']]
output_list = [x for x in nmr_df['Morgan']]
# train_test_cutoff = int(0.8*len(nmr_df['Input'].values))
# input_train = input_list[:train_test_cutoff]
# output_train = output_list[:train_test_cutoff]
# input_test = input_list[train_test_cutoff:]
# output_test = output_list[train_test_cutoff:]
input_train = input_list[0:1000]
output_train = output_list[0:1000]
input_test = input_list[1000:1500]
output_test = output_list[1000:1500]
prompt_message('Database imported successfuly.')

if os.path.exists('./pre_trained'):
    user_load_model = ''
    while user_load_model != 'y' or user_load_model != 'n':
        user_load_model = input('Model exists. Load model [y/n]?')
        if user_load_model == 'y':
            model = import_pre_trained('./pre_trained')
            prompt_message('Model imported successfuly.')
        else:
            model = train_new_model(x_train=input_train, y_train=output_train, input_size=256,
                                    output_size=512, epochs=10, batch_size=32)
else:
    prompt_message('Model wasn\'t found.')
    model = train_new_model(x_train=input_train, y_train=output_train, input_size=256,
                            output_size=512, epochs=10, batch_size=32)
    model.save('./pre_trained/')
    prompt_message('Model saved.')

prompt_message('Model prediction started.')
predicted_output = model.predict(input_test)
prompt_message('Model prediction ended.')

print('Cosine similarity between predictions and label:', cosine_similarity(predicted_output, output_test))
