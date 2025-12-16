# ------------------------------------------------------------
# 🔧 Sistema y utilidades
# ------------------------------------------------------------
import os
import time
import random
import zipfile
import urllib

# ------------------------------------------------------------
# 🔢 Cálculo numérico y científico
# ------------------------------------------------------------
import numpy as np
import pandas as pd
import scipy.ndimage as ndi
from scipy.ndimage import gaussian_filter
from math import log10

# ------------------------------------------------------------
# 🎨 Visualización
# ------------------------------------------------------------
import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------------------------------------
# 🖼️ Procesamiento de imágenes
# ------------------------------------------------------------
from PIL import Image
from skimage import io
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from sklearn.metrics import mean_squared_error as mse
from sklearn.metrics import mean_absolute_error as mae
from sklearn.model_selection import train_test_split

# ------------------------------------------------------------
# 🧠 Machine Learning / Deep Learning
# ------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, TensorDataset, random_split
# from torchsummary import summary # Opcional

# ------------------------------------------------------------
# 📦 Librerías del paquete The Well
# ------------------------------------------------------------
from the_well.data import WellDataset

# ------------------------------------------------------------
# 1. DEFINICIÓN DE DATASETS
# ------------------------------------------------------------
BASE_PATH = "/home/itachi/Desktop/msc/data/datasets"

train_dataset = WellDataset(
    well_base_path=BASE_PATH,
    well_dataset_name="active_matter",
    well_split_name="train",
    n_steps_input=1,
    n_steps_output=0,
    use_normalization=True,
)

eval_dataset = WellDataset(
    well_base_path=BASE_PATH,
    well_dataset_name="active_matter",
    well_split_name="valid",
    n_steps_input=1,
    n_steps_output=0,
    use_normalization=True,
)

test_dataset = WellDataset(
    well_base_path=BASE_PATH,
    well_dataset_name="active_matter",
    well_split_name="test",
    n_steps_input=1,
    n_steps_output=0,
    use_normalization=True,
)

print(f"Train samples: {len(train_dataset)}")
print(f"Eval samples:  {len(eval_dataset)}")
print(f"Test samples:  {len(test_dataset)}")

# ------------------------------------------------------------
# 2. DATA LOADERS (OPTIMIZADOS PARA i5 14th Gen + 32GB RAM)
# ------------------------------------------------------------
BATCH_SIZE = 128
NUM_WORKERS = 8      # Aprovecha los núcleos de tu i5
PIN_MEMORY = True    # Acelera transferencia a GPU

train_loader = torch.utils.data.DataLoader(
    train_dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY,
    persistent_workers=True # Mantiene procesos vivos en RAM
)

eval_loader  = torch.utils.data.DataLoader(
    eval_dataset,  
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=NUM_WORKERS, # Puedes bajarlo a 4 si prefieres, pero 8 va bien
    pin_memory=PIN_MEMORY,
    persistent_workers=True
)

test_loader   = torch.utils.data.DataLoader(
    test_dataset,   
    batch_size=BATCH_SIZE, 
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=PIN_MEMORY
)

# ------------------------------------------------------------
# 3. MODELO AUTOENCODER
# ------------------------------------------------------------
class AE(nn.Module):
    def __init__(self, latent_dim=256):
        super(AE, self).__init__()

        # -------------- ENCODER --------------
        self.encoder_conv = nn.Sequential(
            # 256 -> 128
            nn.Conv2d(11, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 128 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            # 64 -> 32
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  
            # 32 -> 16
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), 
        )
        
        self.enc_out_channels = 256
        self.enc_out_h = 16
        self.enc_out_w = 16
        enc_flat_dim = self.enc_out_channels * self.enc_out_h * self.enc_out_w

        # Cuello de botella
        self.fc_enc = nn.Linear(enc_flat_dim, latent_dim)

        # -------------- DECODER --------------
        self.fc_dec = nn.Linear(latent_dim, enc_flat_dim)

        self.decoder_conv = nn.Sequential(
            # 16 -> 32
            nn.ConvTranspose2d(256, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 32 -> 64
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 64 -> 128
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            # 128 -> 256
            nn.ConvTranspose2d(32, 11, kernel_size=3, stride=2, padding=1, output_padding=1),
            #nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder_conv(x)                  
        h = h.view(x.size(0), -1)                 
        z = self.fc_enc(h)                        
        return z

    def decode(self, z):
        h = self.fc_dec(z)                        
        h = h.view(-1, self.enc_out_channels, self.enc_out_h, self.enc_out_w)  
        xr = self.decoder_conv(h)                 
        return xr

    def forward(self, x):
        z = self.encode(x)
        xr = self.decode(z) # xr shape: (Batch, 11, H, W)
        
        # Separamos las salidas
        out_conc = xr[:, 0:1, :, :]  # Canal 0
        out_vel  = xr[:, 1:, :, :]   # Canales 1-10
        
        # Aplicamos activaciones específicas
        out_conc = torch.sigmoid(out_conc)  # Fuerza rango [0, 1] para densidad
        out_vel  = torch.tanh(out_vel)      # Fuerza rango [-1, 1] para velocidad
        
        # Unimos de nuevo
        return torch.cat([out_conc, out_vel], dim=1), z

# ------------------------------------------------------------
# 4. CONFIGURACIÓN DEL ENTRENAMIENTO (TURBO MODE)
# ------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Usando dispositivo: {device}")

# --- OPTIMIZACIÓN 1: CuDNN Benchmark ---
# Busca el algoritmo de convolución más rápido para tu tamaño fijo (256x256)
if device.type == 'cuda':
    torch.backends.cudnn.benchmark = True

model = AE().to(device)

###############################################################################
# map_location ayuda si guardaste en GPU y ahora cargas en CPU o viceversa
state_dict = torch.load("1model_v2.pth", map_location=device, weights_only=True)

# --- PASO CLAVE: LIMPIEZA DE PREFIJOS DE TORCH.COMPILE ---
# Crea un diccionario nuevo limpiando el prefijo "_orig_mod." si existe
new_state_dict = {}
for key, value in state_dict.items():
    # Si se guardó compilado, las keys se ven así: "_orig_mod.encoder_conv.0.weight"
    # Necesitamos que sean: "encoder_conv.0.weight"
    new_key = key.replace("_orig_mod.", "")
    new_state_dict[new_key] = value

    # 4. Cargar los pesos limpios al modelo
model.load_state_dict(new_state_dict)
print("✅ Modelo cargado exitosamente (Prefijos de compilación corregidos).")
###############################################################################

# --- OPTIMIZACIÓN 2: Torch Compile (PyTorch 2.0+) ---
# Fusiona kernels de CUDA para mayor velocidad
try:
    model = torch.compile(model)
    print("✅ Modelo compilado con torch.compile")
except Exception as e:
    print(f"⚠️ No se pudo compilar el modelo (no crítico): {e}")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-8)

# --- OPTIMIZACIÓN 3: Scaler para Mixed Precision ---
scaler = torch.amp.GradScaler('cuda')

epochs = 200
train_losses = []
val_losses = []
patience = 5  
best_val_loss = float('inf')
patience_counter = 0


# def preprocess_batch(batch, device):
#     """
#     Normalización Híbrida Inteligente:
#     - Canal 0 (Densidad): Min-Max Scaling -> [0, 1]
#     - Canales 1-10 (Velocidad): Max-Abs Scaling -> [-1, 1] (Preserva el 0 y el signo)
#     """
#     # 1. Cargar y ajustar dimensiones
#     # De (B, 1, 256, 256, 11) -> (B, 256, 256, 11)
#     x = batch["input_fields"].to(device, non_blocking=True).squeeze(1) 
    
#     # --- PARTE A: Normalizar Concentración (Canal 0) ---
#     conc = x[..., 0:1] # Mantener dimensión (B, 256, 256, 1)
    
#     # Aplanar para buscar min/max por imagen
#     conc_flat = conc.view(conc.size(0), -1) 
#     cmin = conc_flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1) 
#     cmax = conc_flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1) 
#     if cmin < 0.0:
#         cmin = 0.99
#     # Map to [0, 1]
#     conc_norm = (conc - cmin) / (cmax - cmin + 1e-8)

#     # --- PARTE B: Normalizar Velocidades/Tensores (Canales 1 al 10) ---
#     vel = x[..., 1:] # (B, 256, 256, 10)
    
#     # Buscamos el valor absoluto máximo para escalar sin perder el signo
#     # Queremos que -5 y +5 se conviertan en -1 y +1
#     vel_flat = vel.reshape(vel.size(0), -1) # Aplanar espacialmente y canales
#     # Calculamos el max absoluto por imagen (o por batch si prefieres estabilidad global)
#     vmax_abs = vel_flat.abs().max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    
#     # Map to [-1, 1] aprox
#     vel_norm = vel / (vmax_abs + 1e-8)

#     # --- PARTE C: Reconstruir y Permutar ---
#     # Concatenamos en la última dimensión
#     x_final = torch.cat([conc_norm, vel_norm], dim=-1)
    
#     # Permutar para formato PyTorch (Batch, Channels, Height, Width)
#     x_final = x_final.permute(0, 3, 1, 2).contiguous() 
    
#     return x_final


def preprocess_batch(batch, device):
    """
    Normalización Adaptativa para Alto Contraste:
    - Densidad: Se expande el rango [0.9, 1.0] -> [0, 1]
    - Velocidad: Se escala [-max, +max] -> [-1, 1]
    """
    # 1. Cargar y ajustar dimensiones
    # (B, 1, 256, 256, 11) -> (B, 256, 256, 11)
    x = batch["input_fields"].to(device, non_blocking=True).squeeze(1) 

    # ============================================================
    # --- PARTE A: Normalizar Concentración (Canal 0) -> [0, 1] ---
    # ============================================================
    conc = x[..., 0:1] # (B, 256, 256, 1)
    
    # 1. SEGURIDAD: Eliminamos valores negativos (ruido numérico)
    # Al hacer esto, el mínimo posible se vuelve 0.0.
    conc = torch.clamp(conc, min=0.0) 
    
    # 2. Aplanamos espacialmente para encontrar min/max de CADA imagen individual
    conc_flat = conc.view(conc.size(0), -1)
    
    # 3. Calculamos min y max reales de la imagen (ej: 0.998 y 1.002)
    # Como ya hicimos clamp, cmin está garantizado de ser >= 0
    cmin = conc_flat.min(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    cmax = conc_flat.max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    
    # 4. Normalización Min-Max [0, 1]
    # (Valor - Min) / (Max - Min)
    # epsilon evita división por cero si la imagen es plana
    epsilon = 1e-7
    denom = (cmax - cmin) + epsilon
    
    conc_norm = (conc - cmin) / denom

    # ============================================================
    # --- PARTE B: Normalizar Velocidades -> [-1, 1] ---
    # ============================================================
    vel = x[..., 1:] # (B, 256, 256, 10)
    vel_flat = vel.reshape(vel.size(0), -1)
    
    # Buscamos la magnitud máxima para escalar
    vmax_abs = vel_flat.abs().max(dim=1, keepdim=True)[0].view(-1, 1, 1, 1)
    
    # Normalización [-1, 1]
    vel_norm = vel / (vmax_abs + epsilon)

    # ============================================================
    # --- PARTE C: Reconstruir y Permutar ---
    # ============================================================
    x_final = torch.cat([conc_norm, vel_norm], dim=-1)
    
    # Cambiar a formato (Batch, Channel, H, W) para la red
    x_final = x_final.permute(0, 3, 1, 2).contiguous() 
    
    return x_final



##
l1 = 1.0  
l2 = 0.05  
l3 = 0.02 
l4 = 0.025
l5 = 0.01
##
# ------------------------------------------------------------
# 5. BUCLE DE ENTRENAMIENTO
# ------------------------------------------------------------
for epoch in range(epochs):
    # --- Fase de ENTRENAMIENTO ---
    model.train()
    running_train = 0.0

    # tqdm opcional para ver progreso
    for batch in train_loader:
        # Usamos la función auxiliar
        x = preprocess_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        # Usando sintaxis moderna consistentemente
        with torch.amp.autocast('cuda'):
            xr, z = model(x)


            diff_4_5 = xr[:, 4, :, :] - xr[:, 5, :, :]
            loss1 = torch.mean(diff_4_5 ** 2)
            

            diff_7_8 = xr[:, 7, :, :] - xr[:, 8, :, :]
            loss2 = torch.mean(diff_7_8 ** 2)

            _, _, Lx, Ly = xr.shape
            Lx_phys = 10.0
            Ly_phys = 10.0

            dx = Lx_phys / Lx
            dy = Ly_phys / Ly

            vx = xr[:, 1, :, :]
            vy = xr[:, 2, :, :]

            # dvy_dx = np.gradient(vy, dx, axis=-2)  # ∂x v_y
            # dvx_dy = np.gradient(vx, dy, axis=-1) 
            dvy_dx = torch.gradient(vy, spacing=dx, dim=-1)[0]  # ∂x v_y
            dvx_dy = torch.gradient(vx, spacing=dy, dim=-2)[0]

            sv = xr[:, 7, :, :] - dvy_dx
            sh = xr[:, 8, :, :] - dvx_dy


            SS = ((xr[:, 3, :, :] - xr[:, 6, :, :])**2) + 4 * (xr[:, 4, :, :]**2)

            
            loss3 = torch.mean(sv**2)
            loss4 = torch.mean(sh**2)
            loss5 = torch.mean(SS)
            
            # Loss principal
            loss = criterion(xr, x) + l1*loss1 + l2*loss2 + l3*loss3 + l4*loss4 + l5*loss5
            
            #print(f"crit:{criterion(xr, x).item():.6f}, Loss1: {loss1.item():.6f}, Loss2: {loss2.item():.6f}, Loss3: {loss3.item():.6f}, Loss4: {loss4.item():.6f}, Loss5: {loss5.item():.6f}")
            #loss = criterion(xr, x) 
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        running_train += loss.item()

    epoch_train_loss = running_train / len(train_loader)
    train_losses.append(epoch_train_loss)

    # --- Fase de VALIDACIÓN ---
    model.eval()
    running_val = 0.0

    with torch.no_grad():
        for batch in eval_loader:
# Usamos la misma función (garantiza consistencia)
            x = preprocess_batch(batch, device)

            # Sintaxis corregida para coincidir con el train
            with torch.amp.autocast('cuda'):
                xr, z = model(x)
                diff_4_5 = xr[:, 4, :, :] - xr[:, 5, :, :]
                loss1 = torch.mean(diff_4_5 ** 2)

                diff_7_8 = xr[:, 7, :, :] - xr[:, 8, :, :]
                loss2 = torch.mean(diff_7_8 ** 2)
                _, _, Lx, Ly = xr.shape
                Lx_phys = 10.0
                Ly_phys = 10.0

                dx = Lx_phys / Lx
                dy = Ly_phys / Ly

                vx = xr[:, 1, :, :]
                vy = xr[:, 2, :, :]

                # dvy_dx = np.gradient(vy, dx, axis=-2)  # ∂x v_y
                # dvx_dy = np.gradient(vx, dy, axis=-1) 
                dvy_dx = torch.gradient(vy, spacing=dx, dim=-1)[0]  # ∂x v_y
                dvx_dy = torch.gradient(vx, spacing=dy, dim=-2)[0]

                sv = xr[:, 7, :, :] - dvy_dx
                sh = xr[:, 8, :, :] - dvx_dy


                SS = ((xr[:, 3, :, :] - xr[:, 6, :, :])**2) + 4 * (xr[:, 4, :, :]**2)

                
                loss3 = torch.mean(sv**2)
                loss4 = torch.mean(sh**2)
                loss5 = torch.mean(SS)
                
                # Loss principal
                val_loss = criterion(xr, x) + l1*loss1 + l2*loss2 + l3*loss3 + l4*loss4 + l5*loss5
              
                #val_loss = criterion(xr, x) 
            
            running_val += val_loss.item()

    epoch_val_loss = running_val / len(eval_loader)
    val_losses.append(epoch_val_loss)

    print(f"Epoch {epoch+1}/{epochs} | "
          f"Train Loss: {epoch_train_loss:.6f} | "
          f"Val Loss: {epoch_val_loss:.6f}")

    # --- EARLY STOPPING ---
    if epoch_val_loss < best_val_loss:
        best_val_loss = epoch_val_loss
        patience_counter = 0
        torch.save(model.state_dict(), "best_model_temp.pth") 
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"⏹️ Early stopping at epoch {epoch+1}")
            break 

# -----------------------
# Curvas de pérdida
# -----------------------
plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.xlabel("Epoch")
plt.ylabel("MSE")
plt.legend()
plt.tight_layout()
plt.savefig("2model_v1.png", dpi=300)
plt.show()

# Guardado final
torch.save(model.state_dict(), "2model_v1.pth")
print("✅ Modelo guardado como '2model_v1.pth'")