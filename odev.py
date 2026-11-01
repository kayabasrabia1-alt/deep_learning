import tensorflow as tf
from tensorflow.keras.models import Sequential # yapay sinir ağlarını katman katman üst üste kurmamızı sağlar.
from tensorflow.keras.layers import Dense, LeakyReLU ,ELU #dense: yapay sinir ağlarında tam bağlı katman oluşturmamızı sağlar.bu katmandaki nöronlar bir önceki katmandaki yüm nöronlara bağlanır.
#leakyRelu: negatif deperleri sızdrımaz bir eğim bırakıp tamamen ölmesini engellelr.
#
from tensorflow.keras.optimizers import SGD, RMSprop, Adam 
#sgd:gradyan inmeli optimizasyon yöntemi
#RMSprop: gradyanların ortalamasını uyarlayan optimazsyon türü
# adam: hem geçmiş adım hareketlerini hem gradyan büyüklüğü ile optimize eder
from sklearn.metrics import r2_score
import numpy as np
import matplotlib.pyplot as plt
import time

def model_egit(aktivasyon='relu', ogrenme_orani=0.01, optimizer_tipi='sgd',
               gizli_katman_sayisi=1, noron_sayisi=2, dagilim='normal',
               X_train=None, y_train=None, seed_degeri=2):

    tf.keras.utils.set_random_seed(seed_degeri)

    if dagilim == 'normal':
        intializer = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05, seed=seed_degeri)
    elif dagilim == 'uniform':
        intializer = tf.keras.initializers.RandomUniform(minval=-0.05, maxval=0.05, seed=seed_degeri)
    else:
        intializer = 'glorot_unıform' # ağırlık başlatıcıyı seçmiş olduk.

    model = Sequential()
    act= 'linear' if aktivasyon in ['lineer', 'linear'] else aktivasyon

    for i in range(gizli_katman_sayisi):
        if i ==0:
            if aktivasyon == 'leaky_relu':
                model.add(Dense(noron_sayisi, activation=act, kernel_initializer=intializer, input_dim=X_train.shape[1]))
                model.add(LeakyReLU())
            else:
                act = 'linear' if aktivasyon == 'lineer' else aktivasyon
                model.add(Dense(noron_sayisi, activation=act, kernel_initializer=intializer, input_dim=X_train.shape[1]))
        else:
            if aktivasyon == 'leaky_relu':
                model.add(Dense(noron_sayisi, kernel_initializer=intializer))
                model.add(LeakyReLU())
            elif aktivasyon == 'elu':
                model.add(Dense(noron_sayisi, activation='elu', kernel_initializer=intializer))
            else:
                act = 'linear' if aktivasyon == 'lineer' else aktivasyon
                model.add(Dense(noron_sayisi, activation=act, kernel_initializer=intializer))

    model.add(Dense(1, activation='linear', kernel_initializer=intializer))#çıkış katmanı 


    opt_lower = optimizer_tipi.lower()
    if opt_lower=='sgd':
        opt = SGD(learning_rate=ogrenme_orani)
    elif opt_lower in ['RMSprop', 'rms']:
        opt =RMSprop(learning_rate=ogrenme_orani)
    elif opt_lower =='adam':
        opt = Adam(learning_rate=ogrenme_orani)
    else :
        opt = SGD(learning_rate=ogrenme_orani)

    model.compile(optimizer=opt, loss='mse', metrics=['mse'])

    baslangic_zamani= time.time()#süre ölçümü başlatıyoruz

    history = model.fit(X_train, y_train, epochs=300, verbose=0)# model300 epoch kadar eğitiliyor ekranda kalabalık olmasın diye verbose 0 yaptık

    bitis_zamani = time.time()
    egitim_suresi = bitis_zamani - baslangic_zamani

    y_pred = model.predict(X_train, verbose=0)
    mse_sonuc = np.mean((y_train - y_pred)**2)
    r2_sonuc = r2_score(y_train, y_pred)

    return mse_sonuc, r2_sonuc, egitim_suresi, history 




seed_degeri = 2
np.random.seed(seed_degeri)
tf.keras.utils.set_random_seed(seed_degeri)


p = np.linspace(-2, 2, 200)
# Hedef değerlerin formüle göre hesaplanması
y = 1 + np.sin((np.pi * p) / 4)
X_train = p.reshape(-1, 1)
y_train = y.reshape(-1, 1)

print("--- TABLO 1: Aktivasyon Fonksiyonları Değişimi ---")
for act in ['lineer', 'sigmoid', 'tanh', 'relu', 'leaky_relu', 'elu', 'silu']:
    mse , r2, sure , _ = model_egit(aktivasyon=act,ogrenme_orani=0.1, optimizer_tipi='sgd',
                                    gizli_katman_sayisi=1, noron_sayisi=2, dagilim= 'normal',
                                    X_train=X_train, y_train=y_train, seed_degeri=seed_degeri)
    print(f"Aktivasyon: {act:<12} | MSE: {mse:.5f} | R2: {r2:.5f} | Süre: {sure:.2f} sn ")

print("\n--- TABLO 2: Öğrenme Oranları Değişimi ---")
for lr in [0.1, 0.01, 0.001]:
    mse, r2, sure, _ = model_egit(aktivasyon='relu', ogrenme_orani=lr, optimizer_tipi='sgd', 
                                  gizli_katman_sayisi=1, noron_sayisi=2, dagilim='normal', 
                                  X_train=X_train, y_train=y_train, seed_degeri=seed_degeri)
    print(f"Öğrenme Oranı: {lr:<6} | MSE: {mse:.5f} | R2: {r2:.5f} | Süre: {sure:.2f} sn")

print("\n--- TABLO 3: Optimizasyon Yöntemleri Değişimi ---")
for opt in ['SGD', 'RMSProp', 'ADAM']:
    mse, r2, sure, _ = model_egit(aktivasyon='relu', ogrenme_orani=0.1, optimizer_tipi=opt, 
                                  gizli_katman_sayisi=1, noron_sayisi=2, dagilim='normal', 
                                  X_train=X_train, y_train=y_train, seed_degeri=seed_degeri)
    print(f"Optimizasyon: {opt:<8} | MSE: {mse:.5f} | R2: {r2:.5f} | Süre: {sure:.2f} sn")

print("\n--- TABLO 4: Başlangıç Ağırlık Dağılımları Değişimi ---")
for dag in ['normal', 'uniform']:
    mse, r2, sure, _ = model_egit(aktivasyon='relu', ogrenme_orani=0.1, optimizer_tipi='sgd', 
                                  gizli_katman_sayisi=1, noron_sayisi=2, dagilim=dag, 
                                  X_train=X_train, y_train=y_train, seed_degeri=seed_degeri)
    print(f"Dağılım: {dag:<8} | MSE: {mse:.5f} | R2: {r2:.5f} | Süre: {sure:.2f} sn")

print("\n--- TABLO 5: Gizli Katman Sayıları Değişimi ---")
for katman in [1, 2, 5, 10, 20]:
    mse, r2, sure, _ = model_egit(aktivasyon='relu', ogrenme_orani=0.1, optimizer_tipi='sgd', 
                                  gizli_katman_sayisi=katman, noron_sayisi=2, dagilim='normal', 
                                  X_train=X_train, y_train=y_train, seed_degeri=seed_degeri)
    print(f"Katman Sayısı: {katman:<3} | MSE: {mse:.5f} | R2: {r2:.5f} | Süre: {sure:.2f} sn")

print("\n--- TABLO 6: Gizli Katmandaki Nöron Sayıları Değişimi ---")
for noron in [2, 5, 10, 20, 50]:
    mse, r2, sure, _ = model_egit(aktivasyon='relu', ogrenme_orani=0.1, optimizer_tipi='sgd', 
                                  gizli_katman_sayisi=1, noron_sayisi=noron, dagilim='normal', 
                                  X_train=X_train, y_train=y_train, seed_degeri=seed_degeri)
    print(f"Nöron Sayısı: {noron:<3} | MSE: {mse:.5f} | R2: {r2:.5f} | Süre: {sure:.2f} sn")
                        

print("\n--- En İyi Model İçin Grafikler Çizdiriliyor ve Kaydediliyor ---")


# --- ADIM 4: En İyi Model İçin Grafikler ve Kaydetme ---
print("\n--- En İyi Model Eğitiliyor ve Grafikler Kaydediliyor ---")

from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential

# En iyi parametrelerle (örn: ReLU, Adam, 0.01 öğrenme oranı) modeli doğrudan kurup eğitiyoruz
best_model = Sequential(
    [
        Dense(
            20,
            activation="relu",
            input_dim=X_train.shape[1],
            kernel_initializer=tf.keras.initializers.RandomNormal(
                mean=0.0, stddev=0.05, seed=seed_degeri
            ),
        ),
        Dense(1, activation="linear"),
    ]
)

best_model.compile(
    optimizer=Adam(learning_rate=0.01), loss="mse", metrics=["mse"]
)
best_history = best_model.fit(X_train, y_train, epochs=300, verbose=0)

epochs = range(len(best_history.history["loss"]))
loss_values = best_history.history["loss"]

# 1. Epoch / MSE Error (logaritmik eksen) grafiği
plt.figure(figsize=(8, 5))
plt.plot(epochs, loss_values, color="#b22222", linewidth=2)
plt.yscale("log")
plt.xlabel("Epoch")
plt.ylabel("MSE Error (Log Scale)")
plt.title("Epoch / MSE Error (Logaritmik Eksen)")
plt.grid(True, which="both", ls="--", alpha=0.5)
plt.savefig("epoch_mse_log.png", dpi=300)
plt.close()

# 2. R2 Gelişim Grafiği
r2_gelisim = [
    1 - (l / loss_values[0]) if loss_values[0] > 0 else 0 for l in loss_values
]
plt.figure(figsize=(8, 5))
plt.plot(epochs, r2_gelisim, color="#1f77b4", linewidth=2)
plt.xlabel("Epoch")
plt.ylabel("R2 Skoru")
plt.title("R2 Gelişim Grafiği")
plt.grid(True, ls="--", alpha=0.5)
plt.savefig("r2_gelisim.png", dpi=300)
plt.close()

# 3. Epoch / Hedef-Çıkış karşılaştırma grafiği
y_pred_best = best_model.predict(X_train, verbose=0)

plt.figure(figsize=(8, 5))
plt.plot(X_train, y_train, label="Hedef", color="#1f77b4", linewidth=2)
plt.plot(
    X_train,
    y_pred_best,
    label="Ağ Çıkışı (En İyi Model)",
    color="#b22222",
    linewidth=2,
)
plt.xlabel("p (giriş)")
plt.ylabel("Çıkış")
plt.title("Hedef ve Ağ Çıkışı Karşılaştırması")
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig("hedef_cikis_karsilastirma.png", dpi=300)
plt.close()

print(
    "Tüm grafikler (.png formatında) başarıyla kaydedildi! Raporuna"
    " ekleyebilirsin."
)