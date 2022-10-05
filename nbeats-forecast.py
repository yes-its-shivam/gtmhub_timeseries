import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from model.nbeats import NBeatsBlock
from dependencies import dataprep,make_preds,evaluate_preds

#PARAMS
HORIZON = 1
WINDOW_SIZE = 7

# Values from N-BEATS paper Figure 1 and Table 18/Appendix D
N_EPOCHS = 5000 # "Iterations" in Table 18
N_NEURONS = 512 # called "Width" in Table 18
N_LAYERS = 4 # called "block layers" in Table 18
N_STACKS = 30 # called "stacks" in Table 18

INPUT_SIZE = WINDOW_SIZE * HORIZON # called "Lookback" in Table 18
THETA_SIZE = INPUT_SIZE + HORIZON

# import the dataset
filename = 'PATH_TO_DATAFILE_HERE'
df = pd.read_csv(filename, parse_dates = ['timestamp'], index_col = 'timestamp')
# sort by dates
df.sort_index(inplace = True)
df.drop('Unnamed: 0',axis=1,inplace=True)

score_dict=dict()
tf.random.set_seed(42)

# 0. prepare the data
train_dataset,test_dataset,y_train,y_test = dataprep(df,WINDOW_SIZE)

# 1. Setup N-BEATS Block layer
useRevIN=False
nbeats_block_layer = NBeatsBlock(input_size=INPUT_SIZE,
                                 theta_size=THETA_SIZE,
                                 horizon=HORIZON,
                                 n_neurons=N_NEURONS,
                                 n_layers=N_LAYERS,
                                 name="InitialBlock")

# 2. Create input to stacks
stack_input = layers.Input(shape=(INPUT_SIZE), name="stack_input")
# 3. Create initial backcast and forecast input (backwards predictions are referred to as residuals in the paper)
backcast, forecast = nbeats_block_layer(stack_input)

# Add in subtraction residual layer
residuals = layers.subtract([stack_input, backcast], name=f"subtract_00") 

if useRevIN==True:
  residuals = revinlayer(residuals, mode='norm') #residualts will be normalised using revin layer
# 4. Create stacks of blocks
for i, _ in enumerate(range(N_STACKS-1)): # first stack is already creted in step 3

  # 5. Use the NBeatsBlock to calculate the backcast as well as block forecast
  backcast, block_forecast = NBeatsBlock(
      input_size=INPUT_SIZE,
      theta_size=THETA_SIZE,
      horizon=HORIZON,
      n_neurons=N_NEURONS,
      n_layers=N_LAYERS,
      name=f"NBeatsBlock_{i}"
  )(residuals) 
  if useRevIN==True:
    residuals = revinlayer(residuals, mode='denorm') #denormalising the residuals with revin

  # 6. Create the double residual stacking
  residuals = layers.subtract([residuals, backcast], name=f"subtract_{i}") 
  forecast = layers.add([forecast, block_forecast], name=f"add_{i}")

# 7. Put model stack together
nbeats_model = tf.keras.Model(inputs=stack_input, 
                         outputs=forecast, 
                         name="model_N-BEATS")
# 8. Compile:- loss:MAE, Optimizer:ADAM, LR:0.001
nbeats_model.compile(loss="mae",
                optimizer=tf.keras.optimizers.Adam(0.001),
                metrics=["mae", "mse"])

# 9. Fit the model with EarlyStopping and ReduceLROnPlateau callbacks
nbeats_model.fit(train_dataset,
            epochs=N_EPOCHS,
            validation_data=test_dataset,
            verbose=0,
            callbacks=[tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=200, restore_best_weights=True),
                      tf.keras.callbacks.ReduceLROnPlateau(monitor="val_loss", patience=100, verbose=1)])

#evaluation
nbeats_model.evaluate(test_dataset)

#predictions
nbeats_model_preds = make_preds(nbeats_model, test_dataset)
nbeats_model_preds[:10]

#metrices results
nbeats_model_results = evaluate_preds(y_true=y_test,
                                 y_pred=nbeats_model_preds)

# Plot the created N-BEATS model 
# from tensorflow.keras.utils import plot_model
# plot_model(nbeats_model)

preds = pd.DataFrame(index = np.arange(len(y_train) + len(y_test)))
preds['Real Value'] = np.hstack([y_train,y_test])
preds['Prediction'] = np.hstack([y_train,nbeats_model_preds])
# and plot the predictions
plt.figure(figsize=(14,5))
plt.ylabel(' ')
plt.title('Trend of Actual and Predictions')
plt.plot(preds['Real Value'][:int(len(preds['Prediction'])*0.75)],color ='blue',linewidth=0.5)
plt.plot(preds['Real Value'][int(len(preds['Prediction'])*0.75):],color ='blue',linestyle='dashed',linewidth=0.5)
plt.plot(preds['Prediction'][int(len(preds['Prediction'])*0.75):],color='red',linewidth=0.5)
plt.legend(['Real value train','Real value test','Prediction'])
plt.grid(True)
plt.savefig('FILENAME_PATH'+'.png')

score=[]
for i,j in nbeats_model_results.items():
  score.append(i+':'+str(j))
score_dict[str(filename)]= score

with open('PATH_TO_SCORE_FILE', 'w') as score_file:
  score_file.write(json.dumps(score_dict))

print(str(filename)+' '+'Done!!!')