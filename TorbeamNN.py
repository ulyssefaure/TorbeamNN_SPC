# TorbeamNN, originally from KSTAR (https://arxiv.org/abs/2504.11648), code available at https://github.com/PlasmaControl/KSTAR-torbeamNN/
# This is a reproduction for TCV, ulysse.faure@epfl.ch 


# Load the training data

import os
import tempfile
from scipy.io import loadmat
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
import keras
from keras.layers import Dense, Input
from keras.optimizers import Adam
from keras.models import load_model
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures


# Set up paths
#user = os.environ.get('USER')
#data_dir = os.path.join(tempfile.gettempdir(), user, 'torbeam_training_data')
# for now, set data_dir to the current directory
data_dir = os.path.join(os.getcwd(), 'torbeam_training_data')

# Specify which shot(s) to load
shots = [82660, 82663, 82666] # negative triangularity

# Dictionary to store loaded data
training_data = {}



def _extract_input_features(shot_data):
    """Extract input features from tbm_vector"""

    # Organize the input data: keep theta, phi,
    # vacuum toroidal field (B0 in prof_data),
    # plasma current (I_p in eq_data), 
    # Magnetic axis location (Rmaj, zA in prof_data),
    # Minor radius (Rmin in prof_data),
    # Normalized pressure (betan in prof_data)
    # Elongation (kappa in prof_data),
    # Plasma inductance (li in eq_data),
    # Plasma volume (Vp in prof_data),
    # Electron density for 5 values of rho (interpolate ne),
    # Electron temperature for 5 values of rho (interpolate Te),


    tbm_vector = shot_data['tbm_vector']
    outputs = shot_data['outputs']
    
    n_frames = outputs.results.shape[0]
    n_points = outputs.results.shape[1]
    
    # Feature names
    feature_names = ['theta', 'phi', 'B0', 'I_p', 'Rmaj', 'zA', 'Rmin', 'betan', 'kappa', 'li', 'Vp']
    feature_names.extend([f'ne_rho{i*0.25}' for i in range(5)])
    feature_names.extend([f'Te_rho{i*0.25}' for i in range(5)])
    
    features = np.full((n_frames * n_points, len(feature_names)), np.nan)
    
    for i_frame in range(n_frames):
        tbm = tbm_vector[i_frame]
        prof_data = tbm.inputs.prof_data
        eq_data = tbm.inputs.eq_data
        
        # Extract info
        B0 = eq_data.B0
        I_p = eq_data.Ip
        RMaj = eq_data.Rmaj
        zA = eq_data.zA
        Rmin = eq_data.Rmin
        betan = prof_data.betan
        kappa = eq_data.kappa
        li = eq_data.li
        vp = eq_data.volume
        
        # Get density and temperature arrays (need to interpolate 5 values at rho=[0, 0.25, 0.5, 0.75, 1.0])
        ne = prof_data.ne
        te = prof_data.te
        rho = prof_data.rhopol
        rho_interp = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
        ne_interp = np.interp(rho_interp, rho, ne)
        te_interp = np.interp(rho_interp, rho, te)
        
        for i_point in range(n_points):
            # Extract the theta and phi from the low discrepancy sequence
            theta = outputs.low_disc_seq[i_point, 0]
            phi = outputs.low_disc_seq[i_point, 1]
            density_mult_factor = outputs.low_disc_seq[i_point, 2]
            current_mult_factor = outputs.low_disc_seq[i_point, 3]
            
            # Build feature vector, with multipliers
            feature_vec = [theta, phi, B0, I_p*current_mult_factor, RMaj, zA, Rmin, betan, kappa, li, vp]
            feature_vec.extend(ne_interp * density_mult_factor)
            feature_vec.extend(te_interp / density_mult_factor)
            features[i_frame * n_points + i_point] = feature_vec
            
    
    return np.array(features), feature_names


def _extract_output_features(shot_data, wtol = 0.1):
    res = shot_data['outputs'].results
    res = res.flatten()
    is_valid = np.array([hasattr(r, 'exitflag') and r.exitflag == 0 for r in res])
    is_flavour_B = np.array([hasattr(r, 'cd_profiles') and hasattr(r.cd_profiles, 'cd_deposition_width') for r in res])
    is_valid = is_valid & is_flavour_B  
    n = len(res)
    out = np.full((n, 3), np.nan)
    out[is_valid, 0] = [r.peak_absorption.rho_max for r, v in zip(res, is_valid) if v]
    out[is_valid, 1] = [r.totals.ratio_cd for r, v in zip(res, is_valid) if v]
    out[is_valid, 2] = np.exp(-np.array([r.cd_profiles.cd_deposition_width for r, v in zip(res, is_valid) if v])**2 / (2*wtol**2))    
    # Remove outliers: rho max not in [0,1], eta_cd not in [-0.2, 0.2]
    is_valid = is_valid & (out[:, 0] >= 0) & (out[:, 0] <= 1) & (out[:, 1] >= -0.2) & (out[:, 1] <= 0.2)
    
    print ("Proportion of valid samples: {:.2f}%".format(100 * np.sum(is_valid) / n))
    return out, is_valid


def load_data(shots, data_dir, verbose=True):

    for shot in shots:
        if verbose:
            print(f'Loading data for shot {shot}...')
        
        # Load outputs
        output_file = os.path.join(data_dir, f'training_data_shot_{shot}.mat')
        if os.path.isfile(output_file):
            outputs_data = loadmat(output_file, struct_as_record = False, squeeze_me=True)
            training_data[shot] = {
                'outputs': outputs_data['outputs']
            }
            if verbose:
                print(f'  Loaded outputs from {output_file}')
        else:
            print(f'  Warning: Output file not found: {output_file}')
        
        # Load tbm_vector (inputs)
        tbm_data_file = os.path.join(data_dir, f'tbm_vector_shot_{shot}.mat')
        if os.path.isfile(tbm_data_file):
            tbm_data = loadmat(tbm_data_file, struct_as_record = False, squeeze_me=True)
            training_data[shot]['tbm_vector'] = tbm_data['tbm_vector_struct']     
            if verbose:
                print(f'  Loaded tbm_vector from {tbm_data_file}')
        else:
            print(f'  Warning: TBM data file not found: {tbm_data_file}')

    if verbose:
        print(f'\nTotal shots loaded: {len(training_data)}')



    # Extract features for all shots
    X,y = [], []
    for shot in shots:
        if verbose:
            print(f'\nExtracting features from shot {shot}...')
        targets, is_valid = _extract_output_features(training_data[shot])
        features, input_feature_names = _extract_input_features(training_data[shot])
        if verbose:
            print(f'  Extracted {len(features)} samples with {len(input_feature_names)} features')

        # For now, keep only valid samples
        features = features[is_valid]
        targets = targets[is_valid]

        X.append(features)
        y.append(targets)

    X = np.vstack(X) # acts as a flattening
    y = np.vstack(y)
    if verbose:
        print(f'\nTotal training samples: {len(X)}')
        print(f'Number of features: {len(input_feature_names)}')
        print(f'Output shape: {y.shape}')

    # Split into training, validation, and test sets
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)
    X_val, X_test, y_val, y_test   = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    # Normalize features
    scaler_X = StandardScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_val = scaler_X.transform(X_val)
    X_test = scaler_X.transform(X_test)

    # Scale the outputs
    scaler_y = StandardScaler()
    y_train = scaler_y.fit_transform(y_train)
    y_val = scaler_y.transform(y_val)
    y_test = scaler_y.transform(y_test)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y


@keras.utils.register_keras_serializable(package='torbeam')
class ExponentialDepositionOutput(keras.layers.Layer):
    """Map the third model output to exp(-w_cd**2 / (2 * w_tol**2))."""

    def __init__(self, y2_mean, y2_scale, **kwargs):
        super().__init__(**kwargs)
        self.y2_mean = float(y2_mean)
        self.y2_scale = float(y2_scale)

    def call(self, logits):
        # The last logit represents w_cd / w_tol.  Squaring it makes the
        # exponent non-positive, so the physical prediction is in (0, 1].
        y2_physical = tf.exp(-0.5 * tf.square(logits[:, 2:3]))
        y2_scaled = (y2_physical - self.y2_mean) / self.y2_scale
        return tf.concat([logits[:, :2], y2_scaled], axis=1)

    def get_config(self):
        config = super().get_config()
        config.update({'y2_mean': self.y2_mean, 'y2_scale': self.y2_scale})
        return config


def build_NN_model(input_dim, scaler_y):
    """Build the NN with a physically bounded third output."""
    if scaler_y.mean_.shape[0] != 3:
        raise ValueError('The exponential output constraint requires exactly 3 targets.')

    inputs = Input(shape=(input_dim,))
    x = Dense(120, activation='relu')(inputs)
    x = Dense(120, activation='relu')(x)
    x = Dense(120, activation='relu')(x)
    logits = Dense(3, activation='linear', name='output_logits')(x)
    outputs = ExponentialDepositionOutput(
        scaler_y.mean_[2], scaler_y.scale_[2], name='bounded_deposition_output'
    )(logits)
    return keras.Model(inputs=inputs, outputs=outputs)



def train_NN_model(shots, data_dir):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_data(shots, data_dir, verbose=True)
    model = build_NN_model(X_train.shape[1], scaler_y)

    model.compile(
        optimizer=Adam(learning_rate=10.0**-3),
        loss='mse',
        metrics=['mae'],
    )

    # Stop training if validation loss does not improve for 1 epoch
    early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=250,
        verbose=1,
        restore_best_weights=True
    )    

    history = model.fit(X_train, y_train, 
              epochs=1000,
              validation_data=(X_val, y_val),
              batch_size=50,
              callbacks=[
                      #  tf.keras.callbacks.TensorBoard(logdir),  # log metrics
                      #  hp.KerasCallback(logdir, hparams),  # log hparams
                        early_stopping,
              ],
              verbose=1,
    )
    _, accuracy = model.evaluate(X_test, y_test)
    return model, accuracy, history, scaler_X, scaler_y


def train_linear_model(shots, data_dir, polynomial_features=True):
    X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_data(shots, data_dir, verbose=True)
    
    if polynomial_features:
        poly = PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)
        X_train = poly.fit_transform(X_train)
        X_val = poly.transform(X_val)
        X_test = poly.transform(X_test)

    # Combine train+val since linear regression has no early stopping
    X_fit = np.vstack([X_train, X_val])
    y_fit = np.vstack([y_train, y_val])
    
    model = LinearRegression()
    model.fit(X_fit, y_fit)
    
    y_pred = model.predict(X_test)
    mse = np.mean((y_pred - y_test)**2)
    print(f'Test MSE: {mse:.4f}')
    
    return model, mse, scaler_X, scaler_y, poly if polynomial_features else None

# Retrain and save once to create a model with the bounded third output.
model, accuracy, history, scaler_X, scaler_y = train_NN_model(shots, data_dir)
model.save('torbeamNN_bounded_model.keras')



# model = load_model('torbeamNN_model.keras')

#linearmodel, mse, scaler_X, scaler_y, poly = train_linear_model(shots, data_dir)





# Show the outputs on the test set
X_train, X_val, X_test, y_train, y_val, y_test, scaler_X, scaler_y = load_data(shots, data_dir, verbose=False)
#if poly is not None:
#    X_test = poly.transform(X_test)



y_pred_scaled = model.predict(X_test)
y_pred = scaler_y.inverse_transform(y_pred_scaled)
y_true = scaler_y.inverse_transform(y_test)

fig, axs = plt.subplots(1, y_true.shape[1], figsize=(5*y_true.shape[1], 5))
labels = [r'$\rho_\text{max}$', r'$\eta_\text{cd}$', r'$\exp\left(-w_\text{cd}^2/2w_\text{tol}^2 \right)$'] 

for j, ax in enumerate(axs):
    ax.scatter(y_true[:, j], y_pred[:, j], s=5, alpha=0.5)
    lims = [y_true[:, j].min(), y_true[:, j].max()]
    ax.plot(lims, lims, 'r--')  # diagonal y = x
    ax.set_xlabel('True')
    ax.set_ylabel('Predicted')
    ax.set_title(labels[j])

plt.tight_layout()
plt.show()
