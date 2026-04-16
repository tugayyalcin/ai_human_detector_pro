import torch

print("="*40)
print("🔍 SİSTEM DONANIM VE CUDA TESTİ")
print("="*40)

is_cuda = torch.cuda.is_available()
print(f"CUDA Aktif mi?  : {is_cuda}")

if is_cuda:
    device_name = torch.cuda.get_device_name(0)
    # VRAM'i Bayt cinsinden okuyup Gigabayt'a (GB) çeviriyoruz
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    print(f"Ekran Kartı     : {device_name}")
    print(f"Toplam VRAM     : {vram_gb:.2f} GB")
else:
    print("⚠️ HATA: PyTorch ekran kartını göremiyor! CPU aktif.")
print("="*40)