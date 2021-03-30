#!/usr/bin/env python
# coding: utf-8

# In[1]:


from itertools import chain

def draw_map(m, scale=0.2):
    # draw a shaded-relief image
    m.shadedrelief(scale=scale)
    
    # lats and longs are returned as a dictionary
    lats = m.drawparallels(np.linspace(-90, 90, 13))
    lons = m.drawmeridians(np.linspace(-180, 180, 13))

    # keys contain the plt.Line2D instances
    lat_lines = chain(*(tup[1][0] for tup in lats.items()))
    lon_lines = chain(*(tup[1][0] for tup in lons.items()))
    all_lines = chain(lat_lines, lon_lines)
    
    # cycle through these lines and set the desired style
    for line in all_lines:
        line.set(linestyle='-', alpha=0.3, color='w')


# In[2]:


def profil_vertical_CHL(profile,localisation):
  'Fonction qui retourne la liste des CHL d un point donné pour toutes les profondeurs'
  'profile: dataframe des profils verticaux'
  'localisation: numero de point en question (entre 1 et 9)'
  #ieme endroit
  
  i=localisation-1
  CHL_depth_mat=[]
  CHL_year=[]
  CHL_weeks=[]
  CHL_dates=[]
  profile=profile.sort_values(by=['year','5days'])

  while i in profile.T.columns:
    CHL_depth_mat.append(profile.T.iloc[18:36,i])
    year=str(int(profile.T.iloc[37,i]))
    week=str(int(profile.T.iloc[36,i]))
    CHL_year.append(year)
    CHL_weeks.append(week)
    CHL_dates.append('year{0}_week{1}'.format(year,week))
    i+=9
  return CHL_depth_mat,CHL_year,CHL_weeks,CHL_dates


# In[4]:


def profil_vertical_T(profile,localisation):
  'Fonction qui retourne la liste des CHL d un point donné pour toutes les profondeurs'
  'profile: dataframe des profils verticaux'
  'localisation: numero de point en question (entre 1 et 9)'
  #ieme endroit
  
  i=localisation-1
  T_depth_mat=[]
  T_year=[]
  T_weeks=[]
  T_dates=[]
  profile=profile.sort_values(by=['year','5days'])
  while i in profile.T.columns:
    T_depth_mat.append(profile.T.iloc[:18,i])
    year=str(int(profile.T.iloc[37,i]))
    week=str(int(profile.T.iloc[36,i]))
    T_year.append(year)
    T_weeks.append(week)
    T_dates.append('year{0}_week{1}'.format(year,week))
    i+=9
  return T_depth_mat,T_year,T_weeks,T_dates


# In[3]:


def forceAspect(ax,aspect):
    im = ax.get_images()
    extent =  im[0].get_extent()
    ax.set_aspect(abs((extent[1]-extent[0])/(extent[3]-extent[2]))/aspect)


# In[ ]:


# Eliminer la saisonnalité (moyenne sur l'année) des données en applatis
def Eliminer_annee_moyenne(data,year_averaged_data):
  import pandas as pd
  import numpy as np
  data_sans_anomalie=pd.DataFrame(columns=data.columns)
  for column in data.columns:
    if (column!='5days') and (column!='year') and (column!='latitude') and (column!='longitude'):
        for i in data.index.values:
          j=data.loc[i,'5days']
          data_sans_anomalie.loc[i,column]=data.loc[i,column]-year_averaged_data.loc[j-1,column]
  data_sans_anomalie['5days']=data['5days']
  data_sans_anomalie['year']=data['year']
  data_sans_anomalie['latitude']=data['latitude']
  data_sans_anomalie['longitude']=data['longitude']
  return data_sans_anomalie


# In[ ]:


def average_data(data):
  import pandas as pd
  import numpy as np
  Averaged_data=pd.DataFrame(columns=data.columns)
  for column in data.columns:
    if (column!='5days') and (column!='year') and (column!='latitude') and (column!='longitude'):    
      for i in data.index.values:
        Averaged_data.loc[i,column]=np.mean(data.loc[i,column]).astype('float32')
      Averaged_data[['year','5days']]=data[['year','5days']]
  Averaged_data=Averaged_data.astype('float32')
  return Averaged_data

def year_average(data):
  import numpy as np
  import pandas as pd
  year_averaged_data=pd.DataFrame(columns=[column for column in data.columns if  (column!='year') and (column!='latitude') and (column!='longitude')])
  for column in data.columns:
      if (column!='5days') and (column!='year') and (column!='latitude') and (column!='longitude'):
        for i in np.unique(data['5days'].values):
          year_averaged_data.loc[i-1,column]=data[data['5days']==i][column].mean()
  year_averaged_data['5days']=np.unique(data['5days'].values)
  return year_averaged_data

def localisation_average(Data):
  import pandas as pd
  import numpy as np
  Data_loc_averaged=pd.DataFrame(columns=Data.columns)

  for column in Data.columns:
    i=0
    while i in Data.index.values:
      if  (column!='latitude') and (column!='longitude'):
        Data_loc_averaged.loc[i,column]=Data.loc[i:i+8,column].mean()
      i+=9
  Data_loc_averaged['year']=Data_loc_averaged['year'].astype(int)
  Data_loc_averaged['5days']=Data_loc_averaged['5days'].astype(int)
  #Data_loc_averaged=Data_loc_averaged.drop(['longitude','latitude'],axis=1)
  # Data_loc_averaged.sort_values(by=['year','5days'])
  Data_loc_averaged=Data_loc_averaged.astype('float32')
  return Data_loc_averaged


def cos_sin_date(data):
  import numpy as np 
  from math import pi 
  cos_date=np.cos(10*pi*data['5days']/365)
  sin_date=np.sin(10*pi*data['5days']/365)
  return cos_date,sin_date

def denormalize(Y_train_norm,Data,days_train,target_columns):
  "denormalize log and average year"
  import numpy as np 
  year_average_data=year_average(Data[Data['year']>=2007])
  Y_train_denorm=np.zeros(Y_train_norm.shape)
  for i,day in enumerate(days_train):
    x=(Y_train_norm[i,:]+year_average_data[year_average_data['5days']==day][target_columns]).astype('float32')
    Y_train_denorm[i,:]=np.exp(x)-1
  return Y_train_denorm


def test_train_set_split(Data,train_columns,target_columns,split_year,Normalization_log,Normalization_test,Normalization_train):
  import numpy as np
  # X_test=Data[Data['year']>=split_year ]
  # Y_test=Data[Data['year']>=split_year].loc[:,target_columns] 
  
  lookback_window = 10   # 10 weeks = 1 year per localisation

  
  if Normalization_log==True: #normalisation en log
    Data[target_columns]=np.log(1+Data[target_columns])

  Data_sans_anomalie=Eliminer_annee_moyenne(Data,year_average(Data[Data['year']<split_year]))
  Data_sans_anomalie=Data_sans_anomalie.astype('float32')

  if Normalization_test==True:
    Data_test=Data_sans_anomalie[Data_sans_anomalie['year']>=split_year]
  else:
    Data_test=Data[Data['year']>=split_year]


   
  X_test, Y_test,days_test = [], [],[]
  for i in range(lookback_window, len(Data_test )):
      X_test.append(Data_test.iloc[i - lookback_window:i])
      Y_test.append(Data_test.loc[Data_test.index.values[i],target_columns])
      days_test.append(Data_test.loc[Data_test.index.values[i],'5days'])
  X_test = np.array(X_test)
  Y_test = np.array(Y_test)

  if Normalization_train==True: #Elimination de l'année moyenne de la base d'entrainement
    Data_sans_anomalie_train=Data_sans_anomalie[Data_sans_anomalie['year']<split_year]
  else:
    Data_sans_anomalie_train=Data[Data['year']<split_year]

  X_Train, Y_train,days_train = [], [],[]
  for i in range(lookback_window, len(Data_sans_anomalie_train )):
      X_Train.append(Data_sans_anomalie_train.iloc[i - lookback_window:i])
      Y_train.append(Data_sans_anomalie_train.loc[Data_sans_anomalie_train.index.values[i],target_columns])
      days_train.append(Data_sans_anomalie_train.loc[Data_sans_anomalie_train.index.values[i],'5days'])
  X_Train = np.array(X_Train)
  Y_train = np.array(Y_train)

  return days_test,days_train,Data_test,Data_sans_anomalie_train, X_Train, Y_train, X_test, Y_test
  

################################ TCN-MODEL #####################################################


def root_mean_squared_error(x_true, x_pred):
  from keras import backend as K
  return K.sqrt(K.mean(K.square(x_pred - x_true)))

def saveModel(model, savename):
  from keras.models import model_from_yaml
# serialize model to YAML
  model_yaml = model.to_yaml()
  with open(savename+".yaml", "w") as yaml_file:
      yaml_file.write(model_yaml)
  print("Yaml Model ",savename,".yaml saved to disk")
# serialize weights to HDF5
  model.save_weights(savename+".h5")
  print("Weights ",savename,".h5 saved to disk")

def loadModel(savename):
  from keras.models import model_from_yaml
  with open(savename+".yaml", "r") as yaml_file:
    model = model_from_yaml(yaml_file.read())
  print ("Yaml Model ",savename,".yaml loaded ")
  model.load_weights(savename+".h5",custom_objects={'TCN': TCN})
  print ("Weights ",savename,".h5 loaded ")
  return model


def get_callbacks(name_weights, patience_lr):
    from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
    mcp_save = ModelCheckpoint(name_weights, save_best_only=True, monitor='val_loss', mode='min')
    reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=patience_lr, verbose=1, mode='min')
    return [mcp_save, reduce_lr_loss]

def plot_results(history,model,Data,days_train,X_train,Y_train,vertical_profiles_depth,Normalisation,target_columns):
  "plots the results of the TCN model, returns the plots of the profiles and depth as matrix and as plot"
  import matplotlib.pyplot as plt
  import pandas as pd
  from pandas import DataFrame
  import numpy as np 

  if Normalisation ==True:
    y_pred_train=model.predict(X_train)
    
    fig=plt.figure(figsize=(64,64))
    ax1=plt.subplot(511)   # plot_1
    im=ax1.imshow(Y_train.T,cmap=plt.cm.jet)
    # ensemble_affichage=[i for i in range(len(Y_train.shape[0]))  if i%10==0]
    # plt.xticks(ensemble_affichage,year,rotation=45,size=9)
    plt.colorbar(im,fraction=0.013, pad=0.01)
    ensemble_affichage_y=[i for i in range(18) ]
    plt.yticks(ensemble_affichage_y,["%.2f" % elem for elem in vertical_profiles_depth],size=10)
    plt.ylabel('Depths (m)',size=15)
    ax1.set_title('Profil des vraies valeurs',size=20)
    forceAspect(ax1,aspect=6)
    plt.tight_layout()

    # fig=plt.figure(figsize=(15,10))
    ax2=plt.subplot(512)   # plot_2
    im=ax2.imshow(y_pred_train.T,cmap=plt.cm.jet)
    ax2.set_title('Profil des valeurs prédites',size=20)
    forceAspect(ax2,aspect=6)
    plt.colorbar(im,fraction=0.013, pad=0.01)
    ensemble_affichage_y=[i for i in range(18) ]
    plt.yticks(ensemble_affichage_y,["%.2f" % elem for elem in vertical_profiles_depth],size=10)
    plt.ylabel('Depths (m)',size=15)
    plt.tight_layout()

    ax3=plt.subplot(513)    # plot_3
    ax3.plot(Y_train);
    ax3.set_title('Concentration de la Chla pour les différentes profondeurs (mesure)')

    ax4=plt.subplot(514)     # plot_4
    ax4.plot(y_pred_train);
    ax4.set_title('Concentration de la Chla pour les différentes profondeurs (modèle)')

  elif Normalisation == False :
    split_year=2007
    fig=plt.figure(figsize=(64,64))
    y_pred_train=model.predict(X_train)
    y_pred_train= denormalize(y_pred_train,Data,days_train,target_columns)
    Y_train=denormalize(Y_train,Data,days_train,target_columns)
    ax1=plt.subplot(511)
    im=ax1.imshow(Y_train.T,cmap=plt.cm.jet)
    ax1.set_title('Profil des vraies valeurs',size=20)
    plt.colorbar(im,fraction=0.013, pad=0.01)
    ensemble_affichage_y=[i for i in range(18) ]
    plt.yticks(ensemble_affichage_y,["%.2f" % elem for elem in vertical_profiles_depth],size=10)
    plt.ylabel('Depths (m)',size=15)
    forceAspect(ax1,aspect=6)
    plt.tight_layout()

    ax2=plt.subplot(512)
    im=ax2.imshow(y_pred_train.T,cmap=plt.cm.jet)
    ax2.set_title('Profil des valeurs prédites',size=20)
    plt.colorbar(im,fraction=0.013, pad=0.01)
    ensemble_affichage_y=[i for i in range(18) ]
    plt.yticks(ensemble_affichage_y,["%.2f" % elem for elem in vertical_profiles_depth],size=10)
    plt.ylabel('Depths (m)',size=15)
    forceAspect(ax2,aspect=6)
    plt.tight_layout()

    ax3=plt.subplot(513)
    ax3.plot(Y_train);
    ax3.set_title('Concentration de la Chla pour les différentes profondeurs (mesure)')

    ax4=plt.subplot(514)
    ax4.plot(y_pred_train);
    ax4.set_title('Concentration de la Chla pour les différentes profondeurs (modèle)')
  else:
    raise Error('Normalization variable needs to be bool type ')

  ax5=plt.subplot(515)
  handles, labels = ax5.get_legend_handles_labels()
  ax5.legend(handles, labels)
  g1, =plt.plot(history.history['mse'],label="mse")
  g2, =plt.plot(history.history['mae'],label="mae")
  g3, =plt.plot(history.history['loss'],label="rmse")
  # ax.set_ylim(0,1)
  plt.legend(handles=[g1, g2, g3], labels=['mse', 'mae', 'rmse'])
  ax5.set_title('historique des erreurs',size=20)

  plt.tight_layout()



def TCN_model(X_Train,Y_train,batch_size,activation,dilations,nbfilters,kernelsize, nbstacks,
              batch_norm,layer_norm,weight_norm,
              dropout_rate,lookback_window):
  from tcn import TCN, tcn_full_summary
  from tensorflow.keras.layers import Dense
  from tensorflow.keras.models import Sequential
  from tensorflow.keras.callbacks import EarlyStopping
  from keras.optimizers.schedules import ExponentialDecay
  from keras.optimizers import Adam
  import tensorflow as tf
  from keras.regularizers import l2

  batch_size, time_steps, input_dim = batch_size, 10, X_Train.shape[2]
  lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=1000,
    decay_rate=0.9)
  tcn_layer= TCN(input_shape=(lookback_window, X_Train.shape[2]),
          activation=activation,
          padding='causal',
          dilations=dilations,
          nb_filters=nbfilters,
          kernel_size=kernelsize,
          nb_stacks=nbstacks,
          use_batch_norm=batch_norm,
          use_layer_norm=layer_norm,
          use_weight_norm=weight_norm,
          dropout_rate=dropout_rate

          )
  model_all_data = Sequential([
      TCN(input_shape=(lookback_window, X_Train.shape[2]),
          activation=activation,
          padding='causal',
          dilations=dilations,
          nb_filters=nbfilters,
          kernel_size=kernelsize,
          nb_stacks=nbstacks,
          use_batch_norm=batch_norm,
          use_layer_norm=layer_norm,
          use_weight_norm=weight_norm,
          dropout_rate=dropout_rate

          ),
      Dense(Y_train.shape[1], activation=activation, kernel_regularizer=l2(0.01),)
  ])
  # The receptive field tells you how far the model can see in terms of timesteps.
  print('Receptive field size =', tcn_layer.receptive_field)
  model_all_data.summary()
  adam= Adam(learning_rate=lr_schedule)
  model_all_data.compile(optimizer=adam, loss=root_mean_squared_error  ,metrics=['mse', 'mae', 'mape'])

  # tcn_full_summary(model_all_data, expand_residual_blocks=False)
  return model_all_data


def fit_TCN_model(model,expanding_window_splitter,btscv_splitter,X_Train,Y_train,batch_size,nb_epochs):
  from Time_series import expanding_window,BlockingTimeSeriesSplit
  from Time_series import timefold
  
  if (expanding_window_splitter == True and btscv_splitter == False):
    btscv = expanding_window(horizon = 10,period = 10)
  elif( btscv_splitter==True and expanding_window_splitter== False):
    # btscv = BlockingTimeSeriesSplit(n_splits=11)
    btscv =timefold(folds=66, method='window')
  elif (expanding_window== True and btscv_splitter==True):
    raise ValueError('Only one split can be specified at once')

  history=[]
  for j ,(train_index, test_index) in enumerate(btscv.split(X_Train)):
    X_cv_train, X_cv_val = X_Train[train_index,:,:], X_Train[test_index,:,:]
    Y_cv_train, Y_cv_val = Y_train[train_index,:], Y_train[test_index,:]
    
    # print("TRAIN shape:", cv_train.shape,  "TEST shape:", cv_test.shape) 
    name_weights = "final_model_fold" + str(j) + "_weights.h5"
    callbacks = get_callbacks( name_weights,patience_lr=20)

    history.append(model.fit(X_cv_train,
                            Y_cv_train,
                            epochs=nb_epochs,
                            batch_size=batch_size,
                            validation_data=(X_cv_val,Y_cv_val),
                            callbacks = callbacks,
                             use_multiprocessing=True))
    
    print(model.evaluate(X_cv_val,Y_cv_val))
  return model,history



###################################### GRID SEARCH CV ######################################
  

def plot_grid_search(cv_results, grid_param_1, grid_param_2, name_param_1, name_param_2, best_params):
    
    # Get Test Scores Mean and std for each grid search
    scores_mean = cv_results['mean_test_score']
    scores_mean = np.array(scores_mean).reshape(len(grid_param_2),len(grid_param_1))

    scores_sd = cv_results['std_test_score']
    scores_sd = np.array(scores_sd).reshape(len(grid_param_2),len(grid_param_1))

    # Plot Grid search scores
    _, ax = plt.subplots(1,1)

    # Param1 is the X-axis, Param 2 is represented as a different curve (color line)
    for idx, val in enumerate(grid_param_2):
        ax.plot(grid_param_1, scores_mean[idx,:], '-o', label= name_param_2 + ': ' + str(val))

    ax.set_title(f"Grid Search Best Params: {best_params}", fontsize=12, fontweight='medium')
    ax.set_xlabel(name_param_1, fontsize=12)
    ax.set_ylabel('CV Average Score', fontsize=12)
    ax.legend(loc="best", fontsize=15)
    ax.grid('on')
    ax.legend(bbox_to_anchor=(1.02, 1.02))
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
  
#################################### Optimization #############################
def build_model(hp):
    from tcn import TCN, tcn_full_summary
    from tensorflow.keras.layers import Dense
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.callbacks import EarlyStopping
    from keras.optimizers.schedules import ExponentialDecay
    from keras.optimizers import Adam
    import tensorflow as tf
    from keras.regularizers import l2
    lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-2,
    decay_steps=1000,
    decay_rate=0.9)
  
    lookback_window=10
    # hp_dilation=hp.Choice('dilations',values=[[1,2,4,8,16],[1,2,4,8,16,32]])
    hp_nbfilters=hp.Choice('nb_filters',values=[4,8,16,32])
    hp_kernelsize=hp.Choice('kernel_size',values=[2,3,4])
    hp_nbstacks=hp.Choice('nb_stacks',values=[1,2,3,4])
    hp_dropout_rate=hp.Float('dropout', 0, 0.5, step=0.1, default=0.5)

    lrelu = lambda x: tf.keras.activations.relu(x, alpha=0.1)
    tcn_layer= TCN(input_shape=(lookback_window, X_Train.shape[2]),
          activation=lrelu,
          padding='causal',
          dilations=[1,2,4,8,16],
          nb_filters=hp_nbfilters,
          kernel_size=hp_kernelsize,
          nb_stacks=hp_nbstacks,
          use_batch_norm=True,
          use_layer_norm=False,
          use_weight_norm=False,
          dropout_rate=hp_dropout_rate,

          )

    model=Sequential([tcn_layer,
                      Dense(Y_train.shape[1], 
                            activation=lrelu,
                            kernel_regularizer=l2(0.01))])
   
    adam= Adam(learning_rate=lr_schedule)
    model.compile(optimizer=adam,
                  loss=root_mean_squared_error  ,
                  metrics=['mse', 'mae', 'mape'])
    return model

def RMSE_Tranche_1(x_true, x_pred):
  from keras import backend as K
  return K.sqrt(K.mean(K.square(x_pred[:,:3] - x_true[:,:3])))
def RMSE_Tranche_2(x_true, x_pred):
  from keras import backend as K
  return K.sqrt(K.mean(K.square(x_pred[:,3:6] - x_true[:,3:6])))
def RMSE_Tranche_3(x_true, x_pred):
  from keras import backend as K
  return K.sqrt(K.mean(K.square(x_pred[:,6:9] - x_true[:,6:9])))
def RMSE_Tranche_4(x_true, x_pred):
  from keras import backend as K
  return K.sqrt(K.mean(K.square(x_pred[:,9:12] - x_true[:,9:12])))
def RMSE_Tranche_5(x_true, x_pred):
  from keras import backend as K
  return K.sqrt(K.mean(K.square(x_pred[:,12:15] - x_true[:,12:15])))
def RMSE_Tranche_6(x_true, x_pred):
  from keras import backend as K
  return K.sqrt(K.mean(K.square(x_pred[:,15:19] - x_true[:,15:19])))
# dx = (Y_train).argsort()[:int(len(Y_train)*0.1)]
# np.min(Y_train)
# Y_train[dx,:]
def RMSE_10_pourcent_faibles(x_true, x_pred):
  from keras import backend as K
  import tensorflow as tf
  length=int(x_true.shape[0]*0.1)  
  # index_tf= tf.argsort(x_true)
  # index_tf=index_tf[:length].numpy()
  # index = K.eval(index_tf)
  index=(x_true).argsort()[:length]
  x_pred_10=x_pred[index,:]
  x_true_10=x_true[index,:]
  return K.sqrt(K.mean(K.square(x_pred_10 - x_true_10)))

# tf.compat.v1.disable_eager_execution()
# x = tf.compat.v1.placeholder(tf.float32)
# m = tf.compat.v1.placeholder(tf.float32)
# RMSE_10_pourcent_faibles = tf.py_function(func=RMSE_10_pourcent_faibles, inp=[x, m], Tout=tf.float32)
def RMSE_10_pourcent_fortes(x_true, x_pred):
  from keras import backend as K
  import tensorflow as tf
  length=int(x_true.shape[0]*0.1) 
  # index_tf= tf.argsort(x_true)
  # index_tf=index_tf[:length].numpy()
  # index = K.eval(index_tf)
  index=(-x_true).argsort()[:length]
  x_pred_10=x_pred[index,:]
  x_true_10=x_true[index,:]
  return K.sqrt(K.mean(K.square(x_pred_10 - x_true_10)))



  
  


