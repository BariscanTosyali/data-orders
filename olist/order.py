import pandas as pd
import numpy as np
from olist.data import Olist

class Order:
    def __init__(self):
        # Veriyi Olist sınıfından çekiyoruz
        self.data = Olist().get_data()

    def get_wait_time(self):
        """
        Siparişlerin bekleme sürelerini (gerçek ve beklenen) hesaplar.
        """
        orders = self.data['orders'].copy()
        
        # Sadece teslim edilmiş siparişleri filtrele
        orders = orders[orders['order_status'] == 'delivered'].copy()
        
        # Tarih dönüşümleri
        orders['order_purchase_timestamp'] = pd.to_datetime(orders['order_purchase_timestamp'])
        orders['order_delivered_customer_date'] = pd.to_datetime(orders['order_delivered_customer_date'])
        orders['order_estimated_delivery_date'] = pd.to_datetime(orders['order_estimated_delivery_date'])

        # wait_time: ondalıklı gün farkı
        orders['wait_time'] = (orders['order_delivered_customer_date'] - 
                               orders['order_purchase_timestamp']) / np.timedelta64(1, 'D')
        
        # expected_wait_time
        orders['expected_wait_time'] = (orders['order_estimated_delivery_date'] - 
                                        orders['order_purchase_timestamp']) / np.timedelta64(1, 'D')
        
        # delay_vs_expected: Erken gelirse 0, geç gelirse fark
        delay = (orders['order_delivered_customer_date'] - 
                 orders['order_estimated_delivery_date']) / np.timedelta64(1, 'D')
        orders['delay_vs_expected'] = delay.clip(lower=0)
        
        return orders[['order_id', 'wait_time', 'expected_wait_time', 'delay_vs_expected', 'order_status']]

    def get_review_score(self):
        """
        Siparişlerin yorum skorlarını ve 1/5 yıldız durumlarını döndürür.
        """
        reviews = self.data['order_reviews'].copy()
        
        # 5 yıldız ve 1 yıldız flag'lerini oluştur
        reviews['dim_is_five_star'] = reviews['review_score'].map(lambda x: 1 if x == 5 else 0)
        reviews['dim_is_one_star'] = reviews['review_score'].map(lambda x: 1 if x == 1 else 0)
        
        return reviews[['order_id', 'dim_is_five_star', 'dim_is_one_star', 'review_score']]

    def get_number_items(self):
        """
        Her sipariş için toplam ürün sayısını döndürür.
        """
        items = self.data['order_items'].copy()
        number_items = items.groupby('order_id').agg(
            number_of_items=('product_id', 'count')
        ).reset_index()
        return number_items

    def get_number_sellers(self):
        """
        Her sipariş için benzersiz satıcı sayısını döndürür.
        """
        items = self.data['order_items'].copy()
        number_sellers = items.groupby('order_id').agg(
            number_of_sellers=('seller_id', 'nunique')
        ).reset_index()
        return number_sellers

    def get_price_and_freight(self):
        """
        Sipariş başına toplam ürün fiyatı ve kargo ücretini döndürür.
        """
        price_freight = self.data['order_items'].groupby('order_id').agg(
            price=('price', 'sum'),
            freight_value=('freight_value', 'sum')
        ).reset_index()
        return price_freight

    def get_distance_seller_customer(self):
        """
        Sipariş başına ortalama satıcı-müşteri mesafesini hesaplar.
        """
        geo = self.data['geolocation'].copy()
        items = self.data['order_items'].copy()
        sellers = self.data['sellers'].copy()
        customers = self.data['customers'].copy()
        orders = self.data['orders'].copy()

        # Konum verilerini sadeleştir
        geo = geo.groupby('geolocation_zip_code_prefix').agg({
            'geolocation_lat': 'mean',
            'geolocation_lng': 'mean'
        }).reset_index()

        # Tabloları birleştir
        df = orders.merge(customers, on='customer_id')
        df = df.merge(geo, how='left', left_on='customer_zip_code_prefix', right_on='geolocation_zip_code_prefix')
        df = df.rename(columns={'geolocation_lat': 'c_lat', 'geolocation_lng': 'c_lng'})

        df = df.merge(items, on='order_id')
        df = df.merge(sellers, on='seller_id')
        df = df.merge(geo, how='left', left_on='seller_zip_code_prefix', right_on='geolocation_zip_code_prefix')
        df = df.rename(columns={'geolocation_lat': 's_lat', 'geolocation_lng': 's_lng'})

        # Koordinatı bulunamayanları temizle
        df = df.dropna(subset=['c_lat', 'c_lng', 's_lat', 's_lng'])

        # Haversine hesaplamasını modül bağımsız yapıyoruz
        lat1, lon1, lat2, lon2 = map(np.radians, [df['c_lat'], df['c_lng'], df['s_lat'], df['s_lng']])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        df['distance_seller_customer'] = 6371 * c

        return df.groupby('order_id').agg(
            distance_seller_customer=('distance_seller_customer', 'mean')
        ).reset_index()

    def get_training_data(self, with_distance_seller_customer=False):
        """
        Tüm özellikleri birleştirerek eğitim seti döndürür.
        """
        training_data = self.get_wait_time() \
            .merge(self.get_review_score(), on='order_id') \
            .merge(self.get_number_items(), on='order_id') \
            .merge(self.get_number_sellers(), on='order_id') \
            .merge(self.get_price_and_freight(), on='order_id')
        
        if with_distance_seller_customer:
            distance_data = self.get_distance_seller_customer()
            training_data = training_data.merge(distance_data, on='order_id')

        return training_data.dropna()
    