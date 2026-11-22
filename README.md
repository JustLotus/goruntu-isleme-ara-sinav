# Mikroskobik görüntülerde hücre sayımı

Fluo-N2DH-GOWT1 floresan görüntülerinde klasik segmentasyon ve kare başına hücre sayımı. Bu çalışma 20 kontrollü konfigürasyonu, 3 doğrulama adayını ve sabit bir Otsu baseline’ını karşılaştırır. Seed **31063**. Eğitim arşivinin 01 sekansı geliştirme ve doğrulama, 02 sekansı bağımsız test için kullanılır.

## Çalıştırma

Python 3.12.10 ile doğrulandı. Depoyu klonlayıp çalışma dizini olarak deponun kökünü seçin. `main.ipynb` dosyasını Jupyter’da açın. Colab için depoyu Colab çalışma ortamına klonlayıp ilk hücreden önce çalışma dizinini klonun köküne alın. Notebook bağımlılıkları kurar ve [resmî eğitim arşivini](https://data.celltrackingchallenge.net/training-datasets/Fluo-N2DH-GOWT1.zip) gerektiğinde `data/` dizinine indirir. ZIP’i açmak gerekmez. [Veri seti açıklaması](https://celltrackingchallenge.net/2d-datasets/).

Komut satırı karşılığı:

```text
python -m pip install -r requirements.txt
python select_frames.py data/Fluo-N2DH-GOWT1.zip
python analysis_pipeline.py develop data/Fluo-N2DH-GOWT1.zip
python analysis_pipeline.py predict data/Fluo-N2DH-GOWT1.zip
```

Yeni bir resmî çalıştırmada bu noktada kod, parametreler ve tahminler ayrı bir GitHub commit’ine kaydedilir. Ardından:

```text
python analysis_pipeline.py score data/Fluo-N2DH-GOWT1.zip
python figures.py data/Fluo-N2DH-GOWT1.zip local_figures
```

Yayımlanmış ölçümleri yeniden üretmek için ayrı çalışma kopyası kullanın. Süreler tek OpenCV iş parçacığıyla, görüntü okuma hariç, kare başına üç çalıştırmanın medyanı alınarak ölçülür. Süre değerleri ortama göre değişebilir.

## Dosyalar

| Dosya | İçerik |
| --- | --- |
| `selected_frames.csv` | Kare, sekans, grup, dosya adı, blok ve seed |
| `development_validation_references.csv` | Pozitif TRA kimliği sayısı ve yoğunluk özeti |
| `experiments.csv` | Geliştirme konfigürasyonları, karşılaştırma eşleri ve metrikler |
| `validation_results.csv` | Doğrulama metrikleri ve seçim |
| `final_pipeline.json` | Nihai konfigürasyon ve sabit baseline |
| `test_predictions_locked.csv` | Test referansları okunmadan yazılmış tahminler |
| `test_results.csv` | Kare bazında referans, tahmin ve hata |
| `test_summary.csv` | MAE, RMSE, bias, tam doğru sayım oranı ve süre |
| `temporal_counts.png` | Üç blokta referans ve tahminlerin zamansal karşılaştırması |

Görsel ara çıktılar yerel olarak üretilir. GitHub’a veri seti TIFF’lerini, anotasyon görüntülerini, örnek kare içeren görselleri veya bunları gösteren notebook çıktılarını yüklemeyin.
