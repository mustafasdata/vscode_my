###################################################
# PROJE: Rating Product & Sorting Reviews in Amazon

###################################################
# İş Problemi
###################################################

# E-ticaretteki en önemli problemlerden bir tanesi ürünlere satış sonrası verilen puanların doğru şekilde hesaplanmasıdır.
# Bu problemin çözümü e-ticaret sitesi için daha fazla müşteri memnuniyeti sağlamak, satıcılar için ürünün öne çıkması ve satın
# alanlar için sorunsuz bir alışveriş deneyimi demektir. Bir diğer problem ise ürünlere verilen yorumların doğru bir şekilde sıralanması
# olarak karşımıza çıkmaktadır. Yanıltıcı yorumların öne çıkması ürünün satışını doğrudan etkileyeceğinden dolayı hem maddi kayıp
# hem de müşteri kaybına neden olacaktır. Bu 2 temel problemin çözümünde e-ticaret sitesi ve satıcılar satışlarını arttırırken müşteriler
# ise satın alma yolculuğunu sorunsuz olarak tamamlayacaktır.
# Bitte schreibe die Kommentare auf Deutsch.

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


#######################################################################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
##############################################################################################################


import pandas as pd
import numpy as np
import math
import scipy.stats as st
from astropy.utils.metadata.utils import dtype
from sklearn.preprocessing import MinMaxScaler

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)
pd.set_option("display.width", 500)
pd.set_option("display.expand_frame_repr", False)
pd.set_option("display.float_format", lambda x: "%.5f" % x)


df = pd.read_csv("amazon_review.csv")
df.head()
df.shape
df['overall'].mean()
df['overall'].value_counts()
df.groupby('overall').agg({"helpful_yes": ["count","mean"]})
df.info()
df['helpful_yes'].value_counts()
df['helpful_yes'].nunique()
df['helpful'].value_counts()
df.describe()

df_gecici=df[['reviewerID','asin', 'helpful',  'overall', 'reviewTime', 'day_diff', 'helpful_yes', 'total_vote']]


df["asin"].nunique()

df.groupby("asin").agg({"overall": ["mean"]})

veya
df_asin = df[df["asin"] == "B007WTAJTO"]
df_asin.head()
average_rating = df_asin["overall"].mean()

veya

df["overall"].mean()








##########################################################################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.##########################################
###########################################################################################################


df.head()
df.info()
df.dtypes
df['reviewTime'].dtype
df["reviewTime"] = pd.to_datetime(df["reviewTime"]) # Timestamp tipini datetime'e çevir

df_sorted = df.sort_values(by='reviewTime', ascending=False)
df_sorted.head()

current_date = pd.to_datetime('2014-12-09 0:0:0')   # Şimdiki zamanı belirle

df["days"] = (current_date - df["reviewTime"]).dt.days  # Tarih farkını gün olarak hesapla

df[df["days"] <= 30].count()   # 30 günden önceki kayıt sayısı

df.loc[df["days"] <= 30, "overall"].mean()    # 30 günden önceki puanların ortalaması

df.loc[(df["days"] > 30) & (df["days"] <= 90), "overall"].mean()  # 30 ile 90 gün arasındaki puanların ortalaması

df.loc[(df["days"] > 90) & (df["days"] <= 180), "overall"].mean()

df.loc[(df["days"] > 180), "overall"].mean()


df.loc[df["days"] <= 30, "overall"].mean() * 28/100 + \
    df.loc[(df["days"] > 30) & (df["days"] <= 90), "overall"].mean() * 26/100 + \
    df.loc[(df["days"] > 90) & (df["days"] <= 180), "overall"].mean() * 24/100 + \
    df.loc[(df["days"] > 180), "overall"].mean() * 22/100

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["days"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 30) & (dataframe["days"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 90) & (dataframe["days"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 180), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 26, 22, 22)





































#############################################################################################################
# Adım 3: Ağırlıklandırılmış puanlamada her bir zaman diliminin ortalamasını karşılaştırıp yorumlayınız.####
#############################################################################################################


def user_based_weighted_average(dataframe, w1=22, w2=24, w3=26, w4=28):
    return dataframe.loc[dataframe["days"] <= 10, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["days"] > 10) & (dataframe["days"] <= 45), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["days"] > 45) & (dataframe["days"] <= 75), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["days"] > 75), "overall"].mean() * w4 / 100


user_based_weighted_average(df, 20, 24, 26, 30)

####################
# Weighted Rating
####################

def course_weighted_rating(dataframe, time_w=50, user_w=50):
    return time_based_weighted_average(dataframe) * time_w/100 + user_based_weighted_average(dataframe)*user_w/100

course_weighted_rating(df)

course_weighted_rating(df, time_w=40, user_w=60)



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

##############################################################################################
# Adım 1. helpful_no Değişkenini Üretiniz
###################################################

# Not:
# total_vote bir yoruma verilen toplam up-down sayısıdır.
# up, helpful demektir.
# veri setinde helpful_no değişkeni yoktur, var olan değişkenler üzerinden üretilmesi gerekmektedir.

df.head()

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]


df[["helpful_yes", "total_vote", "helpful_no"]].head()


#############################################################################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
##########################################################################################################

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
