#================
# GRU Classifier 
# ===============
import os, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns, itertools
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GRU, Dropout, Dense, Bidirectional
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.utils import to_categorical

# --Verify files--
def verify_csv_files_exist():
    required_files = [
        f"{split}_cleaned/{act}_merged.csv"
        for split in ["train","test"]
        for act in ["jumping","running","walking","squad"]
    ] + [
        f"partner_{split}/{act}_merged.csv"
        for split in ["train","test"]
        for act in ["jumping","running","walking","spinning"]
    ] + [
        "Recordings_90/combined_90s.csv",
        "Recording_partner/combined_partner.csv"
    ]
    missing = [f for f in required_files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))
    print("All required CSV files found.\n")

verify_csv_files_exist()
print("GPU available:", tf.config.list_physical_devices('GPU'), "\n")

# --Load per-activity data (both partners)--
features = ['ax','ay','az','gx','gy','gz']

def load_activity(folder, act, suffix):
    df = pd.read_csv(f"{folder}/{act}_merged.csv")
    df = df.ffill().bfill()
    df['class'] = f"{act}_{suffix}"
    return df

acts_p1 = ['jumping','running','walking','squad']
acts_p2 = ['jumping','running','walking','spinning']

train_dfs = {f"{a}_P1": load_activity("train_cleaned", a, "P1") for a in acts_p1}
train_dfs.update({f"{a}_P2": load_activity("partner_train", a, "P2") for a in acts_p2})
test_dfs  = {f"{a}_P1": load_activity("test_cleaned", a, "P1") for a in acts_p1}
test_dfs.update({f"{a}_P2": load_activity("partner_test", a, "P2") for a in acts_p2})

print("Train sizes:", {a: len(df) for a, df in train_dfs.items()})
print("Test sizes:",  {a: len(df) for a, df in test_dfs.items()}, "\n")

# --Encode + scale--
encoder = LabelEncoder().fit(list(train_dfs.keys()))
n_classes = len(encoder.classes_)
scaler = MinMaxScaler(feature_range=(-1,1))

for a in train_dfs:
    train_dfs[a][features] = scaler.fit_transform(train_dfs[a][features])
    test_dfs[a][features]  = scaler.transform(test_dfs[a][features])

# --Sequence creation (per-activity)--
def make_sequences(df, window, label_num):
    X, y = [], []
    data = df[features].values
    for i in range(len(data) - window):
        X.append(data[i:i+window])
        y.append(label_num)
    return np.array(X), np.array(y)

def build_dataset(dfs, window):
    X_all, y_all = [], []
    for act, df in dfs.items():
        lbl = encoder.transform([act])[0]
        Xs, ys = make_sequences(df, window, lbl)
        X_all.append(Xs)
        y_all.append(ys)
    return np.concatenate(X_all), np.concatenate(y_all)

# --Hyperparameters--
layer_options   = [1, 2]
neuron_sizes    = [32, 64, 128]
window_lengths  = [25, 50, 100]
dropout_rates   = [0.2, 0.3]
learning_rates  = [0.0005, 0.001]
bidirectional   = [False, True]

# --Model builder--
def build_model(input_shape, neurons, layers, dropout, lr, bidir):
    def maybe_bidir(layer): return Bidirectional(layer) if bidir else layer
    m = Sequential()
    if layers == 1:
        m.add(maybe_bidir(GRU(neurons, input_shape=input_shape)))
        m.add(Dropout(dropout))
    else:
        m.add(maybe_bidir(GRU(neurons, return_sequences=True, input_shape=input_shape)))
        m.add(Dropout(dropout))
        m.add(maybe_bidir(GRU(neurons//2)))
        m.add(Dropout(dropout))
    m.add(Dense(n_classes, activation='softmax'))
    m.compile(optimizer=Adam(lr), loss='categorical_crossentropy', metrics=['accuracy'])
    return m

# --Cross-validation + training--
skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
best_val = -np.inf
best_cfg = {}
best_model = None

for win in window_lengths:
    X_seq, y_seq = build_dataset(train_dfs, win)
    print(f"Built {len(X_seq)} windows (window={win})")
    y_cat = to_categorical(y_seq, n_classes)
    input_shape = (X_seq.shape[1], X_seq.shape[2])

    for layers in layer_options:
        for neurons in neuron_sizes:
            for dropout in dropout_rates:
                for lr in learning_rates:
                    for bi in bidirectional:
                        cfg = f"L={layers} N={neurons} W={win} D={dropout} LR={lr} Bi={bi}"
                        print(f"\n Config: {cfg}")
                        fold_tr, fold_va = [], []
                        for fold, (tr, va) in enumerate(skf.split(X_seq, y_seq), 1):
                            Xtr, Xva = X_seq[tr], X_seq[va]
                            ytr, yva = y_cat[tr], y_cat[va]
                            model = build_model(input_shape, neurons, layers, dropout, lr, bi)
                            cbs = [
                                EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
                                ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-5)
                            ]
                            model.fit(Xtr, ytr, validation_data=(Xva, yva),
                                      epochs=25, batch_size=64, verbose=0, shuffle=False, callbacks=cbs)
                            tr_acc = model.evaluate(Xtr, ytr, verbose=0)[1]
                            va_acc = model.evaluate(Xva, yva, verbose=0)[1]
                            fold_tr.append(tr_acc)
                            fold_va.append(va_acc)
                            print(f"  Fold {fold}: Train={tr_acc:.3f} | Val={va_acc:.3f}")

                        mean_tr, mean_va = np.mean(fold_tr), np.mean(fold_va)

                        # Evaluate test
                        Xt, yt = build_dataset(test_dfs, win)
                        yt_cat = to_categorical(yt, n_classes)
                        test_acc = model.evaluate(Xt, yt_cat, verbose=0)[1]
                        print(f"Mean Train={mean_tr:.3f} | Val={mean_va:.3f} | Test={test_acc:.3f}")

                        # --- Improved model-selection rule (with tie-breaker) ---
                        if (mean_va > best_val) or (
                            np.isclose(mean_va, best_val, atol=1e-4)
                            and mean_tr > best_cfg.get("train", 0)
                        ):
                            if np.isclose(mean_va, best_val, atol=1e-4) and mean_tr > best_cfg.get("train", 0):
                                print("Tie-break update: Same validation accuracy but better training accuracy.")
                            best_val = mean_va
                            best_model = model
                            best_cfg = dict(layers=layers, neurons=neurons, window=win,
                                            dropout=dropout, lr=lr, bidir=bi,
                                            train=mean_tr, val=mean_va, test=test_acc)
                            y_pred = np.argmax(model.predict(Xt, verbose=0), axis=1)
                            cm = confusion_matrix(yt, y_pred)
                            plt.figure(figsize=(7,6))
                            sns.heatmap(cm, annot=True, fmt='d',
                                        xticklabels=encoder.classes_,
                                        yticklabels=encoder.classes_,
                                        cmap='Blues')
                            plt.title(f"Confusion Matrix (BEST)\n{best_cfg}")
                            plt.xlabel("Predicted Label")
                            plt.ylabel("Actual Label")
                            plt.tight_layout()
                            plt.show()

print("\nBEST CONFIGURATION:")
print(best_cfg)


# --Predict on both 90-second combined recordings--
def visualize_predictions(file_label, filepath):
    combined = pd.read_csv(filepath).ffill().bfill()
    Xc = scaler.transform(combined[features])
    def to_windows(X, w): return np.array([X[i:i+w] for i in range(len(X)-w)])
    bw = best_cfg['window']
    Xcw = to_windows(Xc, bw)
    pred = np.argmax(best_model.predict(Xcw, verbose=0), axis=1)
    labels = encoder.inverse_transform(pred)

    print(f"\nPrediction complete for {file_label} 90s file!")
    unique, counts = np.unique(labels, return_counts=True)
    print("\nPredicted class counts:")
    for u, c in zip(unique, counts):
        print(f"  {u}: {c}")

    # ---- Summary visualization ----
    pred_df = pd.DataFrame({'Predicted_Label': labels})
    pred_df[['Movement', 'Person']] = pred_df['Predicted_Label'].str.split('_', expand=True)
    summary = pred_df.groupby(['Person', 'Movement']).size().reset_index(name='Count')

    plt.figure(figsize=(8,4))
    sns.barplot(x='Movement', y='Count', hue='Person', data=summary, palette='Set2')
    plt.title(f"Predicted Activity Distribution ({file_label} 90s Recording)")
    plt.xlabel("Movement")
    plt.ylabel("Number of windows")
    plt.legend(title="Person")
    plt.tight_layout()
    plt.show()

    # --Raw data plot--
    fig, axs = plt.subplots(2, 1, figsize=(14,8), sharex=True)
    t = np.arange(len(combined))
    axs[0].plot(t, combined[['ax','ay','az']])
    axs[0].set_title(f"Accelerometer Data ({file_label})")
    axs[0].set_ylabel("Acceleration")
    axs[1].plot(t, combined[['gx','gy','gz']])
    axs[1].set_title(f"Gyroscope Data ({file_label})")
    axs[1].set_ylabel("Angular Rate")
    axs[1].set_xlabel("Samples")

    colors = {lab: plt.cm.tab10(i) for i, lab in enumerate(np.unique(labels))}
    for i, lab in enumerate(labels):
        s, e = i, i + bw
        for ax in axs:
            ax.axvspan(s, e, color=colors[lab], alpha=0.15)

    handles = [plt.Line2D([0],[0], color=colors[lab], lw=6, alpha=0.5, label=lab)
               for lab in colors.keys()]
    axs[1].legend(handles=handles, loc='upper right', title="Predicted Class")
    plt.tight_layout()
    plt.show()

# --Run visualization for both--
visualize_predictions("Your", "Recordings_90/combined_90s.csv")
visualize_predictions("Partner", "Recording_partner/combined_partner.csv")
