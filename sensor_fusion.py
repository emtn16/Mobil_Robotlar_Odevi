import numpy as np
import math


class EKF:
    def __init__(self, init_x, init_y, init_theta, dt=0.1):
        """
        Genişletilmiş Kalman Filtresi (EKF) Başlatıcısı.
        """
        self.dt = dt

        # Durum Vektörü: [x, y, theta]
        self.x_est = np.array([init_x, init_y, init_theta])

        # Hata Kovaryans Matrisi (Başlangıçta belirsizlik düşük)
        self.P = np.eye(3)

        # Süreç Gürültüsü (Process Noise - Q): Enkoder (Odometri) mekanik kayma belirsizliği
        self.Q = np.diag([0.1, 0.1, np.deg2rad(1.0)]) ** 2

        # Ölçüm Gürültüsü (Measurement Noise - R): LiDAR ve IMU sensörlerinin okuma belirsizliği
        self.R = np.diag([0.2, 0.2, np.deg2rad(2.0)]) ** 2

    def predict(self, v, steering_angle, wheelbase=2.0):
        """
        1. ADIM: TAHMİN (PREDICTION)
        Enkoder'den okunan hız ve direksiyon açısı ile aracın bir sonraki
        adımda nerede olması gerektiğini tahmin eder. (Dead Reckoning)
        """
        x, y, theta = self.x_est

        # Durum Geçiş Modeli (F) - Kinematik Bisiklet Modeli
        self.x_est[0] = x + v * math.cos(theta) * self.dt
        self.x_est[1] = y + v * math.sin(theta) * self.dt
        self.x_est[2] = theta + (v / wheelbase) * math.tan(steering_angle) * self.dt

        # Açıyı -pi ile +pi arasında normalize et
        self.x_est[2] = math.atan2(math.sin(self.x_est[2]), math.cos(self.x_est[2]))

        # Hareket modelinin Jacobian Matrisi (G)
        # Non-linear denklemlerin türevi alınarak doğrusallaştırılır (EKF'nin kalbi)
        G = np.array([
            [1.0, 0.0, -v * math.sin(theta) * self.dt],
            [0.0, 1.0, v * math.cos(theta) * self.dt],
            [0.0, 0.0, 1.0]
        ])

        # Hata Kovaryansını Güncelle (Belirsizlik artar)
        self.P = G @ self.P @ G.T + self.Q

    def update(self, z):
        """
        2. ADIM: GÜNCELLEME (UPDATE / CORRECTION)
        LiDAR eşleştirmesi veya IMU'dan gelen mutlak konum (z) ölçümü ile
        Tahmin adımındaki hatayı düzeltir.

        z: [ölçülen_x, ölçülen_y, ölçülen_theta]
        """
        # Gözlem Matrisi (H) - Sensörlerden doğrudan (x,y,theta) aldığımızı varsayıyoruz
        H = np.eye(3)

        # İnovasyon (Ölçüm ile Tahmin arasındaki fark)
        y = z - (H @ self.x_est)
        y[2] = math.atan2(math.sin(y[2]), math.cos(y[2]))  # Açı farkını normalize et

        # İnovasyon Kovaryansı (S)
        S = H @ self.P @ H.T + self.R

        # Kalman Kazancı (Kalman Gain - K)
        # Hangi veriye daha çok güveneceğimize (Enkoder mi, LiDAR mı?) karar verir
        K = self.P @ H.T @ np.linalg.inv(S)

        # Durum Vektörünü Güncelle (Nihai optimum konum)
        self.x_est = self.x_est + (K @ y)
        self.x_est[2] = math.atan2(math.sin(self.x_est[2]), math.cos(self.x_est[2]))

        # Hata Kovaryansını Küçült (Ölçüm aldığımız için belirsizlik azalır)
        I = np.eye(len(self.x_est))
        self.P = (I - K @ H) @ self.P

        return self.x_est