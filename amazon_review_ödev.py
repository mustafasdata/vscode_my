###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon
###################################################

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.

###################################################
# Veri Seti Hikayesi
###################################################

# Amazon ürün verilerini içeren bu veri seti ürün kategorileri ile çeşitli metadataları içermektedir.
# Elektronik kategorisindeki en fazla yorum alan ürünün kullanıcı puanları ve yorumları vardır.

# Değişkenler:
# reviewerID: Kullanıcı ID’si
# asin: Ürün ID’si
# reviewerName: Kullanıcı Adı
# helpful: Faydalı değerlendirme derecesi
# reviewText: Değerlendirme
# overall: Ürün rating’i
# summary: Değerlendirme özeti
# unixReviewTime: Değerlendirme zamanı
# reviewTime: Değerlendirme zamanı Raw
# day_diff: Değerlendirmeden itibaren geçen gün sayısı
# helpful_yes: Değerlendirmenin faydalı bulunma sayısı
# total_vote: Değerlendirmeye verilen oy sayısı


###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


###################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
###################################################


import pandas as pd
import numpy as np
import math
import scipy.stats as st
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)


df = pd.read_csv("amazon_review.csv")
df.head()
df.shape

df.info()

df.describe()

df=df[['reviewerID','asin', 'helpful',  'overall', 'reviewTime', 'day_diff', 'helpful_yes', 'total_vote']]


df["asin"].nunique()

df.groupby("asin").agg({"overall": ["mean"]})

veya
df_asin = df[df["asin"] == "B007WTAJTO"]
df_asin.head()
average_rating = df_asin["overall"].mean()

veya

df["overall"].mean()

###################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
###################################################


import pandas as pd

# 2. B007WTAJTO ürününe ait verileri filtreleme
product_data = df[df["asin"] == "B007WTAJTO"]

# 3. Tarih formatını dönüştürme
product_data["reviewTime"] = pd.to_datetime(product_data["reviewTime"])

# 4. Gün sayısı ağırlıklarını belirleme
# Gün sayısı, değerlendirmeden itibaren geçen gün sayısı olarak kullanılabilir
# Ağırlık olarak gün sayısını kullanacağız (daha yakın tarihli yorumlar daha yüksek ağırlık alacak)
product_data["weight"] = (pd.Timestamp.now() - product_data["reviewTime"]).dt.days

# 5. Ağırlıklı puan ortalamasını hesaplama
weighted_average_rating = (
    product_data["overall"] * product_data["weight"]
).sum() / product_data["weight"].sum()

# Sonucu yazdırma
print(
    f"B007WTAJTO ürünü için tarihe göre ağırlıklı puan ortalaması: {weighted_average_rating:.2f}"
)


####veya#####

import pandas as pd

# 1. Veri setini okuma
df = pd.read_csv("amazon_reviews.csv")

# 2. Sadece B007WTAJTO ürününe ait verileri filtreleme
product_df = df[df["asin"] == "B007WTAJTO"]

# 3. Ağırlıkları hesaplama (geçen gün sayısına göre ağırlık)
# day_diff sütunu kullanılarak ağırlık hesaplanıyor
product_df["weight"] = 1 / (product_df["day_diff"] + 1)

# 4. Ağırlıklı ortalama hesaplama
# Her bir puan ağırlıkla çarpılır ve toplam ağırlıklı puan hesaplanır
weighted_average = (product_df["overall"] * product_df["weight"]).sum() / product_df[
    "weight"
].sum()

# Sonucu yazdırma
print(f"B007WTAJTO ürününün tarihe göre ağırlıklı ortalama puanı: {weighted_average}")


# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.


import pandas as pd

# 1. Veri setini okuma
df = pd.read_csv("amazon_reviews.csv")

# 2. Sadece B007WTAJTO ürününe ait verileri filtreleme
product_df = df[df["asin"] == "B007WTAJTO"]

# 3. reviewTime sütununu datetime formatına çevirme
product_df["reviewTime"] = pd.to_datetime(product_df["reviewTime"])

# 4. Zaman dilimine göre gruplama (örneğin yıllık)
# reviewTime sütununa göre yıllık gruplama yapıyoruz
product_df["year"] = product_df["reviewTime"].dt.year

# 5. Ağırlık hesaplama (day_diff sütununa göre)
product_df["weight"] = 1 / (product_df["day_diff"] + 1)

# 6. Her bir yıl için ağırlıklı ortalama hesaplama
weighted_avg_by_year = product_df.groupby("year").apply(
    lambda x: (x["overall"] * x["weight"]).sum() / x["weight"].sum()
)

# Sonucu yazdırma
print("Her yıl için ağırlıklı ortalama puan:")
print(weighted_avg_by_year)

# 7. Sonuçları görselleştirme (isteğe bağlı)
import matplotlib.pyplot as plt

plt.plot(weighted_avg_by_year.index, weighted_avg_by_year.values, marker="o")
plt.title("Her Yıl İçin Ağırlıklı Ortalama Puan")
plt.xlabel("Yıl")
plt.ylabel("Ağırlıklı Ortalama Puan")
plt.grid(True)
plt.show()


###################################################
# Görev 2: Ürün için Ürün Detay Sayfasında Görüntülenecek 20 Review'i Belirleyiniz.
###################################################


###################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.


import pandas as pd

# Veri setini yükle
df = pd.read_csv(
    "veri_seti.csv"
)  # 'veri_seti.csv' dosya adını kendi dosya adınla değiştir

# helpful_no değişkenini üret
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# Sonucu kontrol et
print(df[["helpful_yes", "total_vote", "helpful_no"]].head())


###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

import pandas as pd
import numpy as np

# Veri setini yükle
df = pd.read_csv(
    "veri_seti.csv"
)  # 'veri_seti.csv' dosya adını kendi dosya adınla değiştir

# 1. score_pos_neg_diff hesapla
df["score_pos_neg_diff"] = df["helpful_yes"] - (df["total_vote"] - df["helpful_yes"])

# 2. score_average_rating hesapla
df["score_average_rating"] = np.where(
    df["total_vote"] > 0, df["overall"] / df["total_vote"], 0
)

# 3. wilson_lower_bound hesapla
z = 1.96  # %95 güven aralığı için z-değeri
df["wilson_lower_bound"] = (
    df["helpful_yes"] / df["total_vote"]
    + (z**2) / (2 * df["total_vote"])
    - z
    * np.sqrt(
        (df["helpful_yes"] / df["total_vote"])
        * (1 - df["helpful_yes"] / df["total_vote"])
        / df["total_vote"]
        + (z**2) / (4 * df["total_vote"] ** 2)
    )
) / (1 + (z**2) / df["total_vote"])

# NaN değerleri 0 ile doldur
df.fillna(0, inplace=True)

# Sonucu kontrol et
print(
    df[
        [
            "helpful_yes",
            "total_vote",
            "score_pos_neg_diff",
            "score_average_rating",
            "wilson_lower_bound",
        ]
    ].head()
)


#################################veya############################

import numpy as np
import pandas as pd

# 1. Veri setini okuma
df = pd.read_csv("amazon_reviews.csv")

# 2. helpful_no değişkenini üretme (total_vote - helpful_yes)
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# 3. score_pos_neg_diff hesaplama
df["score_pos_neg_diff"] = df["helpful_yes"] - df["helpful_no"]

# 4. score_average_rating hesaplama
# Toplam oy 0 ise, score_average_rating 0 kabul edilir
df["score_average_rating"] = np.where(
    df["total_vote"] == 0, 0, df["helpful_yes"] / df["total_vote"]
)


# 5. wilson_lower_bound hesaplama fonksiyonu
def wilson_lower_bound(helpful_yes, total_vote, confidence=0.95):
    if total_vote == 0:
        return 0
    z = 1.96  # %95 güven seviyesi için z-skoru
    phat = helpful_yes / total_vote
    denominator = 1 + z**2 / total_vote
    numerator = (
        phat
        + z**2 / (2 * total_vote)
        - z * np.sqrt((phat * (1 - phat) + z**2 / (4 * total_vote)) / total_vote)
    )
    return numerator / denominator


# 6. wilson_lower_bound skorunu hesaplama ve veriye ekleme
df["wilson_lower_bound"] = df.apply(
    lambda x: wilson_lower_bound(x["helpful_yes"], x["total_vote"]), axis=1
)

# Skorları eklediğimiz veri setinin ilk birkaç satırını görüntüleme
print(
    df[
        [
            "helpful_yes",
            "helpful_no",
            "score_pos_neg_diff",
            "score_average_rating",
            "wilson_lower_bound",
        ]
    ].head()
)


##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################


import pandas as pd

# Veri setini yükle
df = pd.read_csv(
    "veri_seti.csv"
)  # 'veri_seti.csv' dosya adını kendi dosya adınla değiştir

# 20 rastgele yorumu seç
rasgele_yorumlar = df.sample(n=20, random_state=1)

# Sonuçları kontrol et
print(
    rasgele_yorumlar[
        [
            "reviewerID",
            "reviewerName",
            "overall",
            "helpful_yes",
            "total_vote",
            "score_pos_neg_diff",
            "score_average_rating",
            "wilson_lower_bound",
        ]
    ]
)


####veya #######
# 1. Veri setini okuma (eğer önceden okunduysa bu adımı atlayabilirsiniz)
import pandas as pd

df = pd.read_csv("amazon_reviews.csv")

# 2. helpful sütununu ayrıştırma (önceki adımda yapıldıysa bu kısmı atlayabilirsiniz)
df["helpful_yes"] = df["helpful"].apply(lambda x: eval(x)[0])
df["total_vote"] = df["helpful"].apply(lambda x: eval(x)[1])
df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

# 3. En fazla faydalı oy alan 20 yorumu seçme
top_20_reviews = df.nlargest(20, "helpful_yes")

# 4. Sonuçları yazdırma
print(
    top_20_reviews[
        [
            "reviewerName",
            "overall",
            "helpful_yes",
            "helpful_no",
            "reviewText",
            "reviewTime",
        ]
    ]
)
