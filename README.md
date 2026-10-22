# SÜEDA HORZUM
# 2220329032 
# İSTATİSTİKSEL VERİ MADENCİLİĞİ - 1 


## 1
Korelasyon matrisindeki ilişkilerden minimum ve maksimum olanları (-1 ve 1 hariç) bulan ve bunların hangi değişken çiftine ait olduğunu veren kod bloğu:

```python
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
```

Korelasyon Matrisinin Hesaplanması:İlk adımda veriSeti.corr() fonksiyonu ile veri setindeki sayısal değişkenler arasındaki ikili korelasyon katsayılarını hesapladım.
Filtreleme :Bir değişkenin kendisiyle olan korelasyonu her zaman 1 olduğu ve bazı değişken çiftleri arasında -1 veya 1 korelasyonu bulunabileceği için, bu değerlerin minimum/maksimum aramasını manipüle etmemesi adına np.isclose yardımıyla mutlak değeri 1.0 olan ifadeler nan yani geçersiz değerle değiştirildi.
Matrisin Boyutunu Değiştirme :Matris yapısındaki veriler stack() fonksiyonu ile alt alta dönüştürüp her bir değişken çifti ve buna karşılık gelen korelasyon değerni kolayca listelenebilir hale getirdim.
Uç Değerlerin Tespiti: Seri haline getirdiğim korelasyon değerleri üzerinden .min() ve .max() fonksiyonları ile -1 ve 1 hariç tutarak en düşük ve en yüksek korelasyon değerlerini seçtim; .idxmin() ve .idxmax() fonksiyonları ise bu değerlerin hangi değişken çiftlerine ait olduğunu verdi.

## Yorumlama
Veri setimizdeki değişkenlerin korelasyon analizini incelediğimizde DC ile DMC arasında pozitif yönlü ve orta-güçlü bir doğrusal ilişki (r ≈ 0.69) gözlemlenmiştir; bu durum kuraklık endekslerinin birbirini istatistiksel olarak desteklediğini göstermektedir.Buna karşılık, sıcaklık ile bağıl nem değişkenleri arasında negatif yönlü ve orta dereceli bir ilişki (r ≈ -0.40) saptanmış olup, bu durum havadaki sıcaklık artışının nem oranını düşürdüğü yönündeki sonucu desteklemektedir.Sonuç olarak, çoklu doğrusal bağlantı riski taşıyan DMC-DC çifti gibi yüksek korelasyonlu değişkenlerin regresyon modellerine birlikte dahil edilmesi aşamasında analizler derinleştirilmelidir.


## 2
loc ve iloc kodlarının çalışma prensibi arasındaki farklılık nedir? 
bir örnek üzerinden (veriSeti dataframe üzerinden) gösterin :



### 1. loc ile etiket bazlı seçim örneği


```python
import pandas as pd
sonuc_loc = veriSeti.loc[veriSeti.ay == "ağustos", "sıcaklık"]
print("--- loc Örneği ---")
print(sonuc_loc.head())
```

### 2. iloc ile konum (indeks) bazlı seçim örneği

```python
sonuc_iloc = veriSeti.iloc[0:5, 0:3]
print("\n--- iloc Örneği ---")
print(sonuc_iloc)
```

loc :Satır veya sütunları isim etiketleriyle seçmemizi sağlar.Aralık verirken yazılan bitiş noktası da dahildir.
iloc :Satır veya sütunları pandasın varsayılan tamsayı indeks numaralarına (0, 1, 2 gibi) göre seçer.Standart Python dilimleme mantığı geçerlidir bundan dolayı bitiş noktası hariçtir.
loc Örneği (veriSeti.loc[veriSeti.ay == "ağustos", "sıcaklık"]):Belirli bir koşulu sağlayan satırları etiket bazlı filtreleyerek sadece istenen sütunun ("sıcaklık") verilerini seçer.
iloc Örneği (veriSeti.iloc[0:5, 0:3]):Satır ve sütun numaraları kullanılarak tablonun belirli bir dilimini kesip almaya yarar; ilk 5 satır ve ilk 3 sütun seçilir.



# 3: Rüzgar hızı değişkenini farklı yöntemlerle kategorilere ayırma


### 1. Eşit Sıklıkta Kategorize Etme 
Rüzgar hızı değişkenini aynı sayıda gözlem olacak şekilde 5 kategoriye ayırma:

```python
import pandas as pd
veriSeti["kategorik_ruzgar_hızı_1"] = pd.qcut(veriSeti["ruzgar_hızı"], q=5)
print("--- Eşit Sıklıkta Kategori Dağılımı ---")
print(veriSeti["kategorik_ruzgar_hızı_1"].value_counts())
```
### 2. Eşit Aralıkta Kategorize Etme 
Rüzgar hızı değişkenini eşit aralıklı olarak 5 kategoriye ayırma:
```python
import pandas as pd
veriSeti["kategorik_ruzgar_hızı_2"] = pd.cut(veriSeti["ruzgar_hızı"], bins=5)
print("\n--- Eşit Aralıkta Kategori Dağılımı ---")
print(veriSeti["kategorik_ruzgar_hızı_2"].value_counts())
```

### 3.Tanımlanmış Aralıklarla Kategorize Etme 
Belirlediğimiz aralık sınırlarına göre kategorilere ayırma:

```python
import pandas as pd
bolmeler = [0, 2, 4, 6, 8, 20]
bolmeKategorileri = ["cok_sakin", "sakin", "orta_ruzgarli", "ruzgarli", "firtinali"]

veriSeti["kategorik_ruzgar_hızı_3"] = pd.cut(veriSeti["ruzgar_hızı"], bins=bolmeler, labels=bolmeKategorileri)
print("\n--- Kullanıcı Tanımlı Kategori Dağılımı ---")
print(veriSeti["kategorik_ruzgar_hızı_3"].value_counts())
```

Eşit Sıklıkta Ayrıklaştırma:q parametresi kullanılarak veri setindeki gözlemlerin her bir kategoride yaklaşık olarak eşit sayıda yer alması sağlanır; bu yöntemle aralık genişlikleri değişkenlik gösterir.
Eşit Aralıkta Ayrıklaştırma:bins parametresine sayı verilerek verinin minimum ve maksimum değerleri arasındaki aralık eşit parçalara bölünür; ancak her kategorideki gözlem sayısı eşit olmayabilir.
Kullanıcı Tanımlı Aralıklar:bins parametresiyle özel aralık sınırları, labels parametresiyle ise bu sınırlara karşılık gelen kategori isimleri manuel olarak tanımlanır.
Dağılım İnceleme:Oluşturulan her bir kategorik değişkenin .value_counts() fonksiyonu ile hangi kategoride kaç adet gözlem olduğu listelenip incelenebilir.


Rüzgar hızı değişkenini eşit frekanslı yöntemle 5 gruba ayırdığımızda her bir aralığa yaklaşık olarak benzer sayıda yangın gözlemi düşecek şekilde sınıf sınırları belirlenmiştir.Bunun aksine eşit aralıklı yöntemde rüzgar hızının tüm dağılım genişliği sabit genişlikli 5 alt aralığa bölünmüş ve rüzgar hızı değerlerinin çoğunluğu orta şiddetteki aralıklarda yoğunlaşarak dengesiz bir frekans dağılımı ortaya koymuştur.Son olarak ise rüzgar hızı değerlerini kendi sınırlarımıza göre kategorize ettiğimde ise en çok yangın kaydının 241 gözlemle sakin havayı temsil eden 'sakin' rüzgar sınıfında gerçekleştiği gözlemlenmiştir.

## 4
Yukarıdaki fonksiyon neden çalışmıyor? Hafta içi ve haftasonu değelerini oluşturup bu değişkeni veriSeti sütununa ekleyen kodları yazın.


### Düzenlenmiş hal ve veri setine eklenmesi

```python
def get_hafta_ici_sonu(satir):
    if satir == "cumartesi" or satir == "pazar":
        return 'haftasonu'
    else:
        return 'haftaici'
```

### apply fonksiyonu ile her bir satıra uygulayarak yeni sütuna atama

```python
veriSeti['hafta_ici_sonu'] = veriSeti['gun'].apply(get_hafta_ici_sonu)

print("--- Yeni Sütun İlk 5 Satır ---")
print(veriSeti[['gun', 'hafta_ici_sonu']].head(5))
print("\n--- Dağılım ---")
print(veriSeti['hafta_ici_sonu'].value_counts())
```

Fonksiyonun Çalışmama Nedeni:Orijinal koddaki def get_hafta_ici_sonu(gun): if veriSeti.gun == "cumartesi" ... kısmı tüm veri çerçevesi sütunu (veriSeti.gun) if koşulu içinde tek bir değere indirgenemez.Ayrıca fonksiyona dışarıdan gönderilen gun parametresi fonksiyon içinde kullanılmamış doğrudan veri setinin tamamı sorgulanmış.
Düzeltme:Fonksiyonun doğru çalışabilmesi için parametre olarak gelen tek bir değerin kontrol edilmesi gerekir.

Uyguladığım apply fonksiyonu ile gun değişkenini iki seviyeli (hafta içi ve hafta sonu) nominal kategorik bir değişkene dönüştürerek gözlemleri daha geniş gruplar altında birleştirdim.Elde ettiğim dağılımı incelediğimde 517 orman yangını kaydından 339'unun hafta içi, 178'inin ise hafta sonu günlerinde gerçekleştiğini gözlemliyorum.Örneklem dağılımımda yangın olaylarının yaklaşık %65.6'sının hafta içi günlerine yığıldığını göstermektedir.

## 5
Gün değerlerini sayısal kodlama ile kodlayıp yeni sütunu oluşturun ve veriSeti dataframe'ine ekleyin


### 1. LabelEncoder nesnesinin tanımlanması
```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
```
### 2. 'gun' sütunundaki kategorik değerlerin sayısal değerlere dönüştürülmesi
```python
sayisal_gun = label_encoder.fit_transform(veriSeti["gun"])
```

### 3. Dönüştürülen sayısal değerlerin veriSeti Dataframeine yeni bir sütun olarak eklenmesi
```python
veriSeti["sayisal_gun"] = sayisal_gun
```
### 4. Sonuç kontrolü
```python
print("--- Orijinal ve Sayısal Gün Değerleri (İlk 10 Satır) ---")
print(pd.concat([veriSeti["gun"].head(10), veriSeti["sayisal_gun"].head(10)], axis=1))

print("\n--- Sayısal Gün Dağılımı ---")
print(veriSeti["sayisal_gun"].value_counts())
```

Kategorik ve nominal yapıdaki gün değişkenini nicel bir forma dönüştürdüm.Alfabetik sıraya göre yapılan kodlama sonucu 'cuma' gününün 0, 'cumartesi' gününün 1 ve 'pazar' gününün ise 2 sayısal değeri ile eşleştirildiğini gördük.Bu sayısal kodlama sonrasındaki dağılımda en yüksek frekansın 94 gözlemle pazar gününe (kod: 2) ait olması örneklemimizde pazar günü gerçekleşen yangın kayıtlarının istatistiksel ağırlığının diğer günlere kıyasla daha fazla olduğunu göstermektedir.

## 6
Yanan alan değişkeninin z-score ile standarlaştırın.DMC değerini min-max normalizasyonu ile standartlaştırın.FFMC değerini [3; 7] aralığına getirin.


### 'yanan_alan' değişkeninin Z-score ile standartlaştırılması
```python
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
stScaler = StandardScaler()
veriSeti['yanan_alan_zscore'] = stScaler.fit_transform(veriSeti[['yanan_alan']])
```
### 'DMC' değerinin Min-Max normalizasyonu ile standartlaştırılması
```python
minMaxScaler = MinMaxScaler()
veriSeti['DMC_minmax'] = minMaxScaler.fit_transform(veriSeti[['DMC']])
```
### 'FFMC' değerinin [3, 7] aralığına getirilmesi 
#### Formül: X_yeni = (X - X_min) / (X_max - X_min) * (Yeni_Max - Yeni_Min) + Yeni_Min
```python
ffmc_min = veriSeti['FFMC'].min()
ffmc_max = veriSeti['FFMC'].max()
veriSeti['FFMC_3_7'] = (veriSeti['FFMC'] - ffmc_min) / (ffmc_max - ffmc_min) * (7 - 3) + 3
```
### Sonuçların kontrol edilmesi
```python
print("--- Dönüştürülmüş Değişkenlerin İlk 5 Satırı ---")
print(veriSeti[['yanan_alan', 'yanan_alan_zscore', 'DMC', 'DMC_minmax', 'FFMC', 'FFMC_3_7']].head())
```

Z-Score ile Standardizasyon:StandardScaler() sınıfı, değişkenin ortalamasını 0, standart sapmasını ise 1 olacak şekilde dönüştürür;yanan_alan sütunu bu yöntemle standartlaştırılmıştır.
Min-Max Normalizasyonu:MinMaxScaler() sınıfı, verileri genellikle [0, 1] aralığına ölçeklendirir; DMC değişkeni bu yöntemle dönüştürülmüştür.
Özel Aralık Ölçeklendirme:Standart [0, 1] Min-Max formülü, istenen hedef aralığın genişliği (Yeni_Max - Yeni_Min yani 4) ile çarpılıp minimum değer (Yeni_Min yani 3) eklenerek FFMC değişkeni [3, 7] aralığına getirilmiştir.
Uygulama ve Kontrol:Dönüştürülen yeni sütunlar ana veri çerçevesine (veriSeti) eklenmiş ve .head() fonksiyonu ile ilk değerlerin doğruluğu kontrol edildi.

Sağa çarpık bir dağılım sergileyen 'yanan_alan' değişkenini Z-skor standardizasyonu ile ortalaması 0 ve standart sapması 1 olacak şekilde standardize ettim.Veri setindeki kuraklık göstergelerinden olan 'DMC' değişkenini ise Min-Max normalizasyonu uygulayarak tüm gözlemleri [0, 1] kapalı aralığına sıkıştırdım ve uç değerlerin etkisini korudum.Son olarak, 'FFMC' indeksini doğrusal bir dönüşüm formülüyle [3, 7] gibi spesifik bir hedef aralığına ölçeklendirerek farklı varyanslara sahip parametrelerin analizlerde eşit ağırlıklarla temsil edilmesini sağladım.