# -*- coding: utf-8 -*-
"""
Created on Mon Sep 18 20:24:58 2023

@author: Duygu İÇEN
"""

import numpy as np
import pandas as pd

# pandas kutuphanesi ile bilgisayardaki csv dosyasindan veri seti okuma
veriSeti = pd.read_csv("forestfires.csv", sep=",")

veriSeti.info()
veriSeti.dtypes
print(veriSeti.shape)

row, col = veriSeti.shape
print(row)
print(col)
print(veriSeti.size) 


list(veriSeti.columns) 


for col in veriSeti.columns:
    print(col)
    
veriSeti.rename(
    columns={"X": "X_kordinatı", "Y": "Y_koordinatı", 
             "month": "ay", "day": "gun", "temp": "sıcaklık",
             "RH": "bagıl_nem", "wind": "ruzgar_hızı",
             "rain": "yağmur_miktarı", "area": "yanan_alan"},
    inplace=True,
)  


    
# Kategorik degiskenlerin veri tipinin category yapilmasi
veriSeti.dtypes

veriSeti["ay"] = veriSeti["ay"].astype("category")
veriSeti["gun"] = veriSeti["gun"].astype("category")


veriSeti.head(5)
veriSeti.tail(5)

veriSeti["ay"].value_counts()
veriSeti["ay"] = veriSeti["ay"].replace(["aug", "sep", "mar", "jul", "feb",
                                         "jun", "oct", "apr", "dec", "sept",
                                         "jan", "may", "nov"],
                                        ["ağustos", "eylül", "mart",
                                         "temmuz", "şubat", "haziran",
                                         "ekim", "nisan", "aralık", "eylül",
                                         "ocak", "mayıs", "kasım"])
veriSeti["ay"].value_counts() 


veriSeti["gun"].value_counts()
veriSeti["gun"] = veriSeti["gun"].replace(["sun", "sat", "fri", "mon", "tue",
                                           "thu", "wed", "tuesday"],
                                          ["pazar", "cumartesi", "cuma",
                                          "pazartesi", "salı", "perşembe",
                                           "çarşamba", "salı"])
veriSeti["gun"].value_counts()



veriSeti.describe()  # sadece sayisal degiskenlerin istaitstikleri

pd.set_option('display.max_columns', None)


veriSeti.describe(include="all")

veriSeti.isnull().any()
veriSeti.isnull().sum()


median_DMC = veriSeti['DMC'].median()
veriSeti['DMC'].fillna(median_DMC, inplace=True)

median_bagil_nem = veriSeti['bagıl_nem'].median()
veriSeti['bagıl_nem'].fillna(median_bagil_nem, inplace=True) 

# mode_ay = veriSeti['ay'].mode() --> kategorik veri için en sık görülen kategoriyi bulma

veriSeti.isnull().any()

# feature selection
veriSeti=veriSeti.drop(["X_kordinatı"], axis = 1)
veriSeti=veriSeti.drop(["Y_koordinatı"], axis = 1)

#veriSeti = veriSeti.drop(columns=['X_ekseni', 'Y_ekseni'])

veriSeti.corr() 


veriSeti.corr().unstack().min()  
veriSeti.corr().unstack().idxmin()  

veriSeti.corr().unstack().max()  
veriSeti.corr().unstack().idxmax()  

############# ÖDEV_1:##################################### 
# korelasyon matrisindeki ilişkilerden 
# minimum ve maksimum olanları (-1 ve 1 hariç)
# bulan ve bunların hangi değişken çiftine ait 
# olduğunu veren kod bloğunu yazın. 
##########################################################

import numpy as np
import pandas as pd

# 1. Korelasyon matrisini hesaplama (Kategorik değişkenleri hesaplamaması için numeric_only ekledim.)
corr_matrix = veriSeti.corr(numeric_only=True)

# 2. Köşegen değerlerini ayrıca  -1 ve 1  korelasyonları  filtrelemek için NaN yapmak
corr_matrix_abs = corr_matrix.copy()
corr_matrix_abs = corr_matrix_abs.mask(np.isclose(corr_matrix_abs.abs(), 1.0))

# 3. Matrisi yeniden biçimlendirerek her değişken çiftini ve korelasyon değerini sıralamak
stacked_corr = corr_matrix_abs.stack()

# 4. Minimum ve maksimum değerleri ve  değişken çiftlerini bulmak
min_corr_val = stacked_corr.min()
min_corr_pair = stacked_corr.idxmin()

max_corr_val = stacked_corr.max()
max_corr_pair = stacked_corr.idxmax()

print(f"Minimum Korelasyon (-1 hariç): {min_corr_val} -> Çift: {min_corr_pair}")
print(f"Maksimum Korelasyon ( 1 hariç): {max_corr_val} -> Çift: {max_corr_pair}")



import seaborn as sns
veriSeti.corr()
sns.heatmap(
    veriSeti.corr(), 
    annot = True, # Korelasyon degerlerinin grafigin uzerine yazdirma
    square=True, # Kutularin kare bicimde gosterilmesi
    cmap="Oranges" # Renklendirme secenegi
)

sns.heatmap(veriSeti.corr(),annot=True,linewidth = 0.5, cmap='coolwarm')

# burada yüksek ilişkili olduğu görülen 
# DMC veya DC değerleri istenirse analizden çıkarılabilir. 

# ALT KUMELERE AYIRMA (Subsetting)
# ağustos ayındaki yangınlar
subset1 = veriSeti[veriSeti.ay == "ağustos"] # Birinci yol
subset1_1 = veriSeti.loc[veriSeti.ay == "ağustos",] # Ikinci yol
subset1_2 = veriSeti.iloc[(veriSeti["ay"] == "ağustos").values] # Ucuncu yol

############# ÖDEV_2:##################################### 
# loc ve iloc kodlarının çalışma prensibi arasındaki farklılık nedir? 
# bir örnek üzerinden (veriSeti dataframe üzrinden) gösterin
##########################################################

### 1. loc ile etiket bazlı seçim örneği

import pandas as pd
sonuc_loc = veriSeti.loc[veriSeti.ay == "ağustos", "sıcaklık"]
print("--- loc Örneği ---")
print(sonuc_loc.head())


### 2. iloc ile konum bazlı seçim örneği

sonuc_iloc = veriSeti.iloc[0:5, 0:3]
print("\n--- iloc Örneği ---")
print(sonuc_iloc)


# yanan alan 1 hektar üstünde olan ve pazar günü kaydedilen yangınlar
subset2 = veriSeti[(veriSeti.yanan_alan > 1) & (veriSeti.gun == "pazar")] # Birinci yol
subset2_1 = veriSeti.loc[(veriSeti.yanan_alan > 1) | (veriSeti.gun == "pazar"),] # Bunun subset2 den farkı nedir?
subset2_3 = veriSeti.iloc[((veriSeti["yanan_alan"] > 1) | (veriSeti["gun"] == "pazar")).values] # Ucuncu yol
subset2_4 = veriSeti[(veriSeti.yanan_alan > 1) | (veriSeti.gun == "pazar")] 
subset2_5 = veriSeti[(veriSeti.gun == "cumartesi") | (veriSeti.gun == "pazar")] 

#phyton operatöreleri için kaynak:
# https://www.geeksforgeeks.org/python-operators/

# GRUPLANDIRMA (Aggregate)

veriSeti.loc[veriSeti.ay == "ağustos", "sıcaklık"].mean()
veriSeti.loc[veriSeti.gun == "cumartesi", "DMC"].mean()


veriSeti.groupby(["ay"])["ISI"].aggregate("mean")

veriSeti.loc[veriSeti["gun"]=="pazar", "yanan_alan"].describe()

# Gruplar arasi karsilastirma icin
veriSeti[["ruzgar_hızı", "ay"]].groupby("ay").describe()
veriSeti[["yağmur_miktarı", "gun"]].groupby("gun").describe()

# Farkli foksiyonlar ile denemeler
veriSeti[["bagıl_nem", "ay"]].groupby("ay").mean()
veriSeti[["bagıl_nem", "ay"]].groupby("ay").sum()
veriSeti[["bagıl_nem", "ay"]].groupby("ay").count()
veriSeti[["bagıl_nem", "ay"]].groupby("ay").min()
veriSeti[["bagıl_nem", "ay"]].groupby("ay").max()
veriSeti[["bagıl_nem", "ay"]].groupby("ay").std()
veriSeti[["bagıl_nem", "ay"]].groupby("ay").describe()


# VERI AYRIKLASTIRMA

# I. Yontem: Tum bölmelerde (yaklasik) aynı sayıda gözlem olurken,
#  bolme araligi degisecektir (equal frequency = eşit sıklık ile ayrıklaştırma örneği).
sıcaklık_k1 = pd.qcut(veriSeti["sıcaklık"], q = 3) # buradaki 3 quantile değerini ifade eder
sıcaklık_k1.cat.categories 
sıcaklık_k1 = sıcaklık_k1.cat.rename_categories(["dusuk_sıcaklık", "orta_sıcaklık", "yuksek_sıcaklık"])
sıcaklık_k1.value_counts()
pd.concat([veriSeti["sıcaklık"].head(10), sıcaklık_k1[0:10]], axis=1)

veriSeti["kat_sıcaklık"] = sıcaklık_k1 # kategorileştirdiğimiz yeni değişkeni veri setine ekledik

# II. Yontem: Bolme araliklarinin sinirlarini tanimlamak icin kullanilir. 
# Bolmelerdeki gozlemlerin dagitimi konusunda 
# hicbir garanti yoktur (equal interval = eşit aralıklı ayrıklaştırma örneği). 
    
sıcaklık_k2 = pd.cut(veriSeti["sıcaklık"], bins = 3)
sıcaklık_k2.cat.categories
sıcaklık_k2 = sıcaklık_k2.cat.rename_categories(["dusuk_sıcaklık", "orta_sıcaklık", "yuksek_sıcaklık"])
sıcaklık_k2.value_counts()
pd.concat([veriSeti["sıcaklık"].head(10), sıcaklık_k2[0:10]], axis=1)

# Ucuncu Yontem: Bolme araliklarinin sinirlari kullanici 
# tarafindan belirlenir (fixed).
bolmeKategorileri = ["0-10", "11-25", "26-35", "36-45", "45+"]
bolmeler = [-1,10,25,35,45,100]
sıcaklık_k3 = pd.cut(veriSeti["sıcaklık"], bins=bolmeler, labels=bolmeKategorileri)
sıcaklık_k3.value_counts()

# kullanıcı tanımlı ayrıklaştırmaya başka bir örnek:
bolmeKategorileri = ["18-34", "35-49", "50-59", "60-64", "65-69", "70-74", "75-79", "80+"]
bolmeler = [17,34,49,59,64,69,74,79,100]
sıcaklık_k4 = pd.cut(veriSeti["sıcaklık"], bins=bolmeler, labels=bolmeKategorileri)
sıcaklık_k4.value_counts()


karsilastirma = pd.concat([veriSeti["sıcaklık"],sıcaklık_k1, sıcaklık_k2, sıcaklık_k3], axis=1)
karsilastirma.columns =  ["sıcaklık","esitFrekans", "esitAralik", "kullaniciTanimli"]

############# ÖDEV_3:##################################### 
# ruzgar hızı değişkeninin aynı sayıda gözlem olacak sekilde (equal frequency interval)
# 5 kategoriye ayiriniz ve 
# yeni bir kategorik_ruzgar_hızı_1 degiskenine atayiniz. 
# Ardindan hangi kategoride kac adet yangın kaydı mevcut inceleyiniz.
# ...
# ruzgar hızı değişkeninin (equal range interval) kategorisine ayrınız
# yeni bir kategorik_ruzgar_hızı_2 degiskenine atayiniz. 
# Ardindan hangi kategoride kac adet yangın kaydı mevcut inceleyiniz.
# ...
# ruzgar hızı değişkeninin kendi belirlediğiniz aralıklardan (fixed interval) oluşan kategorilere ayrınız
# yeni bir kategorik_ruzgar_hızı_3 degiskenine atayiniz. 
# Ardindan hangi kategoride kac adet yangın kaydı mevcut inceleyiniz.
##########################################################


import pandas as pd
veriSeti["kategorik_ruzgar_hızı_1"] = pd.qcut(veriSeti["ruzgar_hızı"], q=5)
print("--- Eşit Sıklıkta Kategori Dağılımı ---")
print(veriSeti["kategorik_ruzgar_hızı_1"].value_counts())

### 2. Eşit Aralıkta Kategorize Etme 
import pandas as pd
veriSeti["kategorik_ruzgar_hızı_2"] = pd.cut(veriSeti["ruzgar_hızı"], bins=5)
print("\n--- Eşit Aralıkta Kategori Dağılımı ---")
print(veriSeti["kategorik_ruzgar_hızı_2"].value_counts())


### 3.Tanımlanmış Aralıklarla Kategorize Etme 

import pandas as pd
bolmeler = [0, 2, 4, 6, 8, 20]
bolmeKategorileri = ["cok_sakin", "sakin", "orta_ruzgarli", "ruzgarli", "firtinali"]

veriSeti["kategorik_ruzgar_hızı_3"] = pd.cut(veriSeti["ruzgar_hızı"], bins=bolmeler, labels=bolmeKategorileri)
print("\n--- Kullanıcı Tanımlı Kategori Dağılımı ---")
print(veriSeti["kategorik_ruzgar_hızı_3"].value_counts())



# ONE HOT ENCODING
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder


def func(x):
    if x <= 50:
        return '<= 50'
    elif 50.0001 < x <= 80:
        return '(50-80])'
    return '>80'
    
veriSeti['kat_DMC'] = veriSeti['DMC'].apply(func)



def get_hafta_ici_sonu(gun):
    if veriSeti.gun == "cumartesi" :
        return 'haftasonu'
    elif veriSeti.gun == "pazar":
        return 'haftasonu'
    return 'haftaici'
    
veriSeti['hafta_ici_sonu'] = veriSeti['gun'].apply(get_hafta_ici_sonu)



hafta_ici_sonu = veriSeti.loc[(veriSeti.gun == "cumartesi") | (veriSeti.gun == "pazar"),] 

############## ÖDEV_1 ############## 
# yukarıdaki fonksiyon neden çalışmıyor?
# hafta içi ve haftasonu değelerini oluşturup bu değişkeni veriSeti sütununa ekleyen kodları yazın.
####################################  


def get_hafta_ici_sonu(satir):
    if satir == "cumartesi" or satir == "pazar":
        return 'haftasonu'
    else:
        return 'haftaici'


### apply fonksiyonu ile her bir satıra uygulayarak yeni sütuna atama

veriSeti['hafta_ici_sonu'] = veriSeti['gun'].apply(get_hafta_ici_sonu)

print("--- Yeni Sütun İlk 5 Satır ---")
print(veriSeti[['gun', 'hafta_ici_sonu']].head(5))
print("\n--- Dağılım ---")
print(veriSeti['hafta_ici_sonu'].value_counts())


# sayisal kodlama
label_encoder = LabelEncoder()
sayisalay = pd.DataFrame(label_encoder.fit_transform(veriSeti.ay), columns=["ay_sayisal"])
pd.concat([veriSeti.ay.head(10),sayisalay.ay_sayisal.head(10)], axis = 1)

ay1 = pd.concat([veriSeti.ay, sayisalay], axis=1)
sayisalay = label_encoder.fit_transform(veriSeti.ay)

veriSeti = veriSeti.assign(sayisal_ay = sayisalay)
veriSeti["ay"].value_counts() 
veriSeti["sayisal_ay"].value_counts() 

############## ÖDEV_2 ############## 
# Gün değerlerini sayısal kodlama ile kodlayıp yeni sütunu oluşturun ve 
# veriSeti dataframe'ine ekleyin
####################################  

### 1. LabelEncoder nesnesinin tanımlanması

import pandas as pd
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()

### 2. 'gun' sütunundaki kategorik değerlerin sayısal değerlere dönüştürülmesi

sayisal_gun = label_encoder.fit_transform(veriSeti["gun"])


### 3. Dönüştürülen sayısal değerlerin veriSeti Dataframeine yeni bir sütun olarak eklenmesi

veriSeti["sayisal_gun"] = sayisal_gun

### 4. Sonuç kontrolü

print("--- Orijinal ve Sayısal Gün Değerleri (İlk 10 Satır) ---")
print(pd.concat([veriSeti["gun"].head(10), veriSeti["sayisal_gun"].head(10)], axis=1))

print("\n--- Sayısal Gün Dağılımı ---")
print(veriSeti["sayisal_gun"].value_counts())


# one hot kodlama

label_encoder = LabelEncoder()
sayisal_kat_sicaklik = pd.DataFrame(label_encoder.fit_transform(veriSeti.kat_sıcaklık), columns=["sayısal_kat_sıcaklık"])
pd.concat([veriSeti.kat_sıcaklık.head(10),sayisal_kat_sicaklik.sayısal_kat_sıcaklık.head(10)], axis = 1)


onehot_encoder = OneHotEncoder(sparse=False)
sayısal_kat_sıcaklık = sayisal_kat_sicaklik.reshape(len(sayisal_kat_sicaklik), 1)
Kodluay = onehot_encoder.fit_transform(sayisal_kat_sicaklik)
print(Kodluay)



# VERI NORMALIZASYONU (Data Normalization)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
veriSeti.dtypes

# min-max normalizasyon yontemi
norm_list = ['FFMC', 'DMC', 'DC', 'ISI', 'sıcaklık', 'bagıl_nem', 'ruzgar_hızı',
             'yağmur_miktarı', 'yanan_alan']

scaler = MinMaxScaler()
scaler.fit(veriSeti[norm_list])
n_veriSeti = scaler.transform(veriSeti[norm_list])
n_veriSeti = pd.DataFrame(n_veriSeti, columns = norm_list) # dataframe-e donusturuldu


# standardizasyon (z-score)
stScaler = StandardScaler()
s_veriSeti = stScaler.fit_transform(veriSeti[norm_list])
s_veriSeti = pd.DataFrame(s_veriSeti, columns = norm_list) # dataframe-e donusturuldu


############## ÖDEV_3 ############## 
# yanan alan değişkeninin z-score ile standarlaştırın
# DMC değerini min-max normalizasyonu ile standartlaştırın
# FFMC değerini [3; 7] aralığına getirin
####################################  

### 'yanan_alan' değişkeninin Z-score ile standartlaştırılması

stScaler = StandardScaler()
veriSeti['yanan_alan_zscore'] = stScaler.fit_transform(veriSeti[['yanan_alan']])

### 'DMC' değerinin Min-Max normalizasyonu ile standartlaştırılması

minMaxScaler = MinMaxScaler()
veriSeti['DMC_minmax'] = minMaxScaler.fit_transform(veriSeti[['DMC']])

### 'FFMC' değerinin [3, 7] aralığına getirilmesi
#### Formül: X_yeni = (X - X_min) / (X_max - X_min) * (Yeni_Max - Yeni_Min) + Yeni_Min

ffmc_min = veriSeti['FFMC'].min()
ffmc_max = veriSeti['FFMC'].max()
veriSeti['FFMC_3_7'] = (veriSeti['FFMC'] - ffmc_min) / (ffmc_max - ffmc_min) * (7 - 3) + 3

### Sonuçların kontrol edilmesi

print("--- Dönüştürülmüş Değişkenlerin İlk 5 Satırı ---")
print(veriSeti[['yanan_alan', 'yanan_alan_zscore', 'DMC', 'DMC_minmax', 'FFMC', 'FFMC_3_7']].head())
