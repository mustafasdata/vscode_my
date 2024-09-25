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

df=df[['overall', 'reviewTime', 'day_diff', 'helpful_yes', 'total_vote']]


df["asin"].nunique()

df.groupby("asin").agg({"overall": ["mean"]})


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

#current_date = pd.to_datetime('2014-12-09 0:0:0')   # Şimdiki zamanı belirle
#df["days"] = (current_date - df["reviewTime"]).dt.days  # Tarih farkını gün olarak hesapla


df[df["day_diff"] <= 30].count()   # 30 günden önceki kayıt sayısı

df.loc[df["day_diff"] <= 30, "overall"].mean()    # 30 günden önceki puanların ortalaması

df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean()  # 30 ile 90 gün arasındaki puanların ortalaması

df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean()

df.loc[(df["day_diff"] > 180), "overall"].mean()


df.loc[df["day_diff"] <= 30, "overall"].mean() * 28/100 + \
    df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 26/100 + \
    df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 24/100 + \
    df.loc[(df["day_diff"] > 180), "overall"].mean() * 22/100

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 25, 23, 22)







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



#############################derya hoca ile ##########



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


import matplotlib.pyplot as plt
import pandas as pd
import math
import scipy.stats as st

pd.set_option('display.max_columns', None)
# pd.set_option('display.max_rows', 10)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.5f' % x)

###################################################
# GÖREV 1: Average Rating'i Güncel Yorumlara Göre Hesaplayınız ve Var Olan Average Rating ile Kıyaslayınız.
###################################################

# Paylaşılan veri setinde kullanıcılar bir ürüne puanlar vermiş ve yorumlar yapmıştır.
# Bu görevde amacımız verilen puanları tarihe göre ağırlıklandırarak değerlendirmek.
# İlk ortalama puan ile elde edilecek tarihe göre ağırlıklı puanın karşılaştırılması gerekmektedir.


##################################################################################################################
# Adım 1: Veri Setini Okutunuz ve Ürünün Ortalama Puanını Hesaplayınız.
#################################################################################################################

df = pd.read_csv("amazon_review.csv")
df.head()
df.info()
df.nunique()
df.isnull().sum()
df=df[['overall', 'reviewTime', 'day_diff', 'helpful_yes', 'total_vote']]
df["overall"].mean()


df.head()

##################################################################################################################
# Adım 2: Tarihe Göre Ağırlıklı Puan Ortalamasını Hesaplayınız.
#################################################################################################################


df.info()


df['reviewTime'] = pd.to_datetime(df['reviewTime'], dayfirst=True)

#df_sorted = df.sort_values(by='reviewTime', ascending=False)
#df_sorted.head()
#current_date = pd.to_datetime(str(df['reviewTime'].max()))
# df["day_diff"] = (current_date - df['reviewTime']).dt.days

#df.sort_values(by='day_diff', ascending=False).tail()


#################


df[df["day_diff"] <= 30].count()

df.loc[df["day_diff"] <= 30, "overall"].mean()

df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean()

df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean()

df.loc[(df["day_diff"] > 180), "overall"].mean()


df.loc[df["day_diff"] <= 30, "overall"].mean() * 28/100 + \
    df.loc[(df["day_diff"] > 30) & (df["day_diff"] <= 90), "overall"].mean() * 26/100 + \
    df.loc[(df["day_diff"] > 90) & (df["day_diff"] <= 180), "overall"].mean() * 24/100 + \
    df.loc[(df["day_diff"] > 180), "overall"].mean() * 22/100

def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[df["day_diff"] <= 30, "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 30) & (dataframe["day_diff"] <= 90), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 90) & (dataframe["day_diff"] <= 180), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > 180), "overall"].mean() * w4 / 100

time_based_weighted_average(df)

time_based_weighted_average(df, 30, 25, 23, 22)





##################


df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean()
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.75)), "overall"].mean()


# zaman bazlı ortalama ağırlıkların belirlenmesi
def time_based_weighted_average(dataframe, w1=28, w2=26, w3=24, w4=22):
    return dataframe.loc[dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.25), "overall"].mean() * w1 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.25)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.50)), "overall"].mean() * w2 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.50)) & (dataframe["day_diff"] <= dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w3 / 100 + \
           dataframe.loc[(dataframe["day_diff"] > dataframe["day_diff"].quantile(0.75)), "overall"].mean() * w4 / 100


df.loc[df["day_diff"] <= df["day_diff"].quantile(0.25), "overall"].mean() * 28 / 100 + \
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.25)) & (df["day_diff"] <= df["day_diff"].quantile(0.50)), "overall"].mean() * 26 / 100 + \
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.50)) & (df["day_diff"] <= df["day_diff"].quantile(0.75)), "overall"].mean() * 24 / 100 + \
df.loc[(df["day_diff"] > df["day_diff"].quantile(0.75)), "overall"].mean() * 22 / 100

time_based_weighted_average(df, w1=28, w2=26, w3=24, w4=22)

df["overall"].mean()


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

df.head()
df.tail()

df["helpful_no"] = df["total_vote"] - df["helpful_yes"]

df = df[["reviewerName", "overall", "summary", "helpful_yes", "helpful_no", "total_vote", "reviewTime"]]

df.head()

###################################################
# Adım 2. score_pos_neg_diff, score_average_rating ve wilson_lower_bound Skorlarını Hesaplayıp Veriye Ekleyiniz
###################################################

def wilson_lower_bound(up, down, confidence=0.95):
    """
    Wilson Lower Bound Score hesapla

    - Bernoulli parametresi p için hesaplanacak güven aralığının alt sınırı WLB skoru olarak kabul edilir.
    - Hesaplanacak skor ürün sıralaması için kullanılır.
    - Not:
    Eğer skorlar 1-5 arasıdaysa 1-3 negatif, 4-5 pozitif olarak işaretlenir ve bernoulli'ye uygun hale getirilebilir.
    Bu beraberinde bazı problemleri de getirir. Bu sebeple bayesian average rating yapmak gerekir.

    Parameters
    ----------
    up: int
        up count
    down: int
        down count
    confidence: float
        confidence

    Returns
    -------
    wilson score: float

    """
    n = up + down
    if n == 0:
        return 0
    z = st.norm.ppf(1 - (1 - confidence) / 2)
    phat = 1.0 * up / n
    return (phat + z * z / (2 * n) - z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)) / (1 + z * z / n)


def score_up_down_diff(up, down):
    return up - down


def score_average_rating(up, down):
    if up + down == 0:
        return 0
    return up / (up + down)

#####################
# score_pos_neg_diff
#####################

df.head(20)
df["score_pos_neg_diff"] = df.apply(lambda x: score_up_down_diff(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("score_pos_neg_diff", ascending=False).head(20)


# 500 0 -> 500 (daha iyi bir yorum pozitiflik açısından)
# 1500 1000 -> 500



# score_average_rating
df["score_average_rating"] = df.apply(lambda x: score_average_rating(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("score_average_rating", ascending=False).head(20)

# 500 0 -> 500    500 / 500 1
# 1 0 -> 1/1 1


# wilson_lower_bound
df["wilson_lower_bound"] = df.apply(lambda x: wilson_lower_bound(x["helpful_yes"], x["helpful_no"]), axis=1)

df.sort_values("wilson_lower_bound", ascending=False).head(20)



##################################################
# Adım 3. 20 Yorumu Belirleyiniz ve Sonuçları Yorumlayınız.
###################################################

df.sort_values("wilson_lower_bound", ascending=False).head(20)




##################################################
# BONUS
##################################################

# wilson lower bound değerine göre değerlendirmeleri sıraladık. Helpful olarak işaretlenme sayısının yüksek olması
# yapılan yorumun iyi olduğunu göstermez. Yorumları belirlerken hangi yorumların olumlu, olumsuz ve nötr olduğunu da görmek
# bizlere fayda sağlayacaktır. Bu nedenle yapılan yorumlara sentiment analizi yani duygu analizi yaparak positive, negatif, nötr
# sınıflandırması yapabiliriz. Bunun için Natural Language Toolkit kütüphanesi kullanacağız.

# NLTK -> Doğal dil işleme kütüphanesi (https://www.nltk.org/)
# Sentiment analysis (Duygu analizi), metindeki olumlu veya olumsuz duyguyu tespit etme sürecidir.
# Genellikle şirketler veya markalar tarafından sosyal verilerdeki duyarlılığı tespit etmek, marka itibarını ölçmek ve müşterileri anlamak için kullanılır.


# Gerekli kütüphanelerin yüklenmesi
from nltk.sentiment.vader import SentimentIntensityAnalyzer #nltk kütüphanesinde hazır bir sentiment Intensity Analizi var
import nltk
import re
from textblob import TextBlob # metnin olumlu-olumsuz durumuna göre size 0-1 aralığında bir değer dönmektedir. TextBlob ile amacımız yazının olumlu mu olumsuz mu içerik içerdiğini anlamaktır.
nltk.downloader.download('vader_lexicon')
# VADER (Valence Aware Dictionary for Sentiment Reasoning), duygunun hem polaritesine (pozitif/negatif) hem de
# yoğunluğuna (gücüne) duyarlı olan metin duygu analizi için kullanılan bir modeldir.
# NLTK paketinde mevcuttur ve doğrudan etiketlenmemiş metin verilerine uygulanabilir.


# Daha iyi sonuçlar için summary değişkenimizin içerisindeki text'leri temizlenmesi gerekiyor. (Noktalama işaretlerinden kurtulmak, küçük harfe çevirmek..)
rt = lambda x: re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)"," ",x)
df["summary"] = df["summary"].map(rt)
df["summary"] = df["summary"].str.lower()
df.head(10)


# Sentiment analysis
# TextBlob Çıktı olarak polarity ve subjectivity değerlerini döndürecektir.
# Polarity  duygu durumunu yani olumlu mu olumsuz mu olduğunu belirtir.
# Bize 0 ile 1 arasında değer döner. 1' e ne kadar yakınsa o kadar olumlu, 0'a ne kadar yakınsa o kadar olumsuzdur.
df[['polarity', 'subjectivity']] = df['summary'].apply(lambda Text: pd.Series(TextBlob(Text).sentiment))
for index, row in df['summary'].iteritems():
    print(index,row)
    score = SentimentIntensityAnalyzer().polarity_scores(row)
    neg = score['neg']
    neu = score['neu']
    pos = score['pos']
    if neg > pos:
        df.loc[index, 'sentiment'] = "negative"
    elif pos > neg:
        df.loc[index, 'sentiment'] = "positive"
    else:
        df.loc[index, 'sentiment'] = "neutral"
    df.loc[index, 'neg'] = neg
    df.loc[index, 'neu'] = neu
    df.loc[index, 'pos'] = pos

df.head(10)


# Duygu analizlerinin dağılımı
def count_values_in_column(data,feature):
    total=data.loc[:,feature].value_counts(dropna=False)
    percentage=round(data.loc[:,feature].value_counts(dropna=False,normalize=True)*100,2)
    return pd.concat([total,percentage],axis=1,keys=['Total','Percentage'])

count_values_in_column(df,"sentiment")
#           Total  Percentage
# positive   2691    54.75000
# neutral    1856    37.76000
# negative    368     7.49000


# Worldcloud -> Daha fazla görünen kelimelere daha fazla önem vererek oluşturulan görsel
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt

def create_wordcloud(text):
    stopwords = set(STOPWORDS)
    wc = WordCloud(background_color="white",
                  max_words=3000,
                  stopwords=stopwords,
                  repeat=True)
    wc.generate(str(text))
    plt.figure()
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.show()


# Pozitif yorumlar için kelime bulutu
create_wordcloud(df[df["sentiment"]=="positive"]["summary"].values)

# Negatif yorumlar için kelime bulutu
create_wordcloud(df[df["sentiment"]=="negative"]["summary"].values)

# Nötr yorumlar için kelime bulutu
create_wordcloud(df[df["sentiment"]=="neutral"]["summary"].values)


# 20 Yorumu Belirlerken, artık yorumların olumlu, olumsuz ve notr olma durumlarını da katabiliriz.
df[df["sentiment"]=="positive"].sort_values("wilson_lower_bound", ascending=False).head(20)



