import os
import datetime
import google.generativeai as genai
import re
import json  # ← JSON çıktısı için eklendi

class TarifiAsistani:
    def __init__(self, api_key, urunler_json, raf_omru_dict):
        print(api_key)
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel("gemini-1.5-flash")
        self.urunler_json = urunler_json
        self.raf_omru_dict = raf_omru_dict

    def ocr_duzenle(self, ocr_input):
        items_str = "\n".join([f"- {item[0]}" for item in ocr_input if isinstance(item, list) and len(item) > 0])
        prompt = f"""
Aşağıda market fişinden OCR ile bozulmuş ürün adları var. Her birini düzelt. Tüm sayısal değerleri, gramajları, sembolleri çıkar. Sadece ürün adını doğru şekilde düzelt ama bazı ürün isimleri değişik olabilir o durumda onalrı değiştirme(örnek: Pia PORT) Eğerki bir anlamlı isim bulamazsan eskisi gibi bırak.

Bozuk ürün adları:
{items_str}

Sadece madde madde temiz ürün isimlerini sırayla yaz:
- su
- kanepe burger
- biber çarliston
"""
        try:
            yanit = self.model.generate_content(prompt).text.strip()
            temiz_urunler = []
            for satir in yanit.split("\n"):
                temiz = satir.lstrip("- ").strip()
                if temiz:
                    temiz_urunler.append(temiz)
            return temiz_urunler
        except Exception as e:
            print("OCR temizleme hatası:", e)
            return []

    def tahmini_raf_omru_al(self, urun_adi):
        urun_adi_lower = urun_adi.lower()
        for anahtar in self.raf_omru_dict:
            if anahtar in urun_adi_lower:
                return self.raf_omru_dict[anahtar]
        prompt = f"{urun_adi} adlı ürün buzdolabında ortalama kaç gün dayanır? Cevabı sadece sayı ile ver (örnek: 7)."
        try:
            yanit = self.model.generate_content(prompt).text.strip()
            gun = int(''.join(filter(str.isdigit, yanit)))
            return gun if gun > 0 else 30
        except:
            return 30

    def malzeme_listesini_olustur(self):
        bugun = datetime.date.today()
        user_ingredients = []
        for item in self.urunler_json:
            urun_adi = item["product"]["name"]
            gun = self.tahmini_raf_omru_al(urun_adi)
            skt_tarihi = bugun + datetime.timedelta(days=gun)
            user_ingredients.append({
                "isim": urun_adi,
                "skt": skt_tarihi.strftime("%Y-%m-%d")
            })
        return user_ingredients

    def skt_kontrol_et(self, user_ingredients):
        bugun = datetime.date.today()
        yakinda_sonlanacaklar = []

        for urun in user_ingredients:
            skt_str = urun.get("skt")
            if skt_str is None:
                continue  # Skip if no SKT date

            try:
                skt_date = datetime.datetime.strptime(skt_str, "%Y-%m-%d").date()
                if skt_date <= bugun + datetime.timedelta(days=3):
                    yakinda_sonlanacaklar.append(urun)
            except ValueError:
                continue  # Skip if SKT is not in correct format

        return yakinda_sonlanacaklar

    def tarif_iste(self, user_ingredients, yaklasan_urunler):
        # Create readable strings for all user ingredients
        malzeme_listesi = ", ".join([
            f"{urun['product']['name']} (SKT: {urun['skt'] or 'belirtilmemiş'})"
            for urun in user_ingredients
        ])

        # Create a warning for soon-to-expire ingredients
        if yaklasan_urunler:
            yaklasan_listesi = ", ".join([
                f"{urun['product']['name']} (SKT: {urun['skt']})"
                for urun in yaklasan_urunler
            ])
            skt_bilgi = f"\nÖnemli: Şu ürünlerin son kullanma tarihi yaklaşıyor: {yaklasan_listesi}."
        else:
            skt_bilgi = "\nTüm ürünlerin son kullanma tarihleri uygun."

        # Build the prompt
        prompt = f"""
    Elimde şu malzemeler var: {malzeme_listesi}.{skt_bilgi}

    Bana sadece bu malzemelere uygun Türk mutfağına ait 3 yemek tarifi öner.
    Tarifler uydurma ya da anlamsız olmasın. Lütfen yalnızca Türkiye'de bilinen, geleneksel ve mantıklı yemek isimleri ver.
    Tariflerin sadece ismini belirt.

    Önerilen 3 Tarif:
    1. [YEMEK ADI (kişi sayısı)]
    2. [YEMEK ADI (kişi sayısı)]
    3. [YEMEK ADI (kişi sayısı)]
    """
        return self.model.generate_content(prompt).text.strip()

    def secili_tarifi_getir(self, tarifler_text, secim):
        pattern = rf"{secim}\.\s*(.?)\s\("
        match = re.search(pattern, tarifler_text)
        secilen_tarif_adi = match.group(1).strip() if match else "[bilinmiyor]"

        prompt = f"""
Aşağıda önerilen 3 tarif var:

{tarifler_text}

Kullanıcı {secim}. tarifi seçti: {secilen_tarif_adi}

Şimdi sadece bu tarif hakkında detaylı ve biçimlendirilmiş bilgi ver.

Tarif Detayları:
- Elimizde olmayan malzemeleri listele.
- Hazırlık süresi, pişirme süresi, malzemeler ve yapılış adımlarını sırala.

{secilen_tarif_adi} Tarifi:

Hazırlık Süresi: …
Pişirme Süresi: …

Malzemeler:
- Miktarlarıyla yaz. Elimizde olmayanlar için: (elimizde yok) yaz.

Yapılışı:
1. …

Pişirme İpuçları:
- …

Eksik malzemeler varsa hangi tarifin başarısını etkiler onu da açıkla.
"""
        return self.model.generate_content(prompt).text.strip()

    def besin_degeri_getir(self, yemek_adi, kisi_sayisi=4):
        prompt = f"""
Lütfen aşağıdaki Türk yemeği için tahmini besin değerlerini ver:

Yemek: {yemek_adi}
Kişi sayısı: {kisi_sayisi}

Bilgiler:
- Toplam kalori
- Kişi başı kalori
- Toplam protein
- Toplam karbonhidrat
- Toplam yağ

Sadece kısa ve net sayılarla cevap ver.
"""
        try:
            return self.model.generate_content(prompt).text.strip()
        except Exception as e:
            return f"Besin bilgisi alınamadı: {e}"


if __name__ == '__main__':

    api_key = "AIzaSyAdsfvTluiJ6uzSOAGXDULcOUcdflWPw0I"
    raf_omru_dict = {
        "yoğurt": 7, "peynir": 14, "yumurta": 21, "domates": 5, "çeri": 5,
        "pirinç": 180, "zeytinyağı": 365, "coca cola": 180,
        "süt": 5, "et": 3, "tavuk": 3, "balık": 2
    }

    ocr_input = {
        "SU 5 İL a,": "¥19,50-",
        "KANEPEBÜRGER . 2906 X 2": "kOS, 90",
        "BAH PULBXBER DYR": "#29 ,50",
        "MÜZ İTKAL | X 0,955": "48,62",
        "DÜMATES - , X O, 490  kg": "x13,44",
        "GAZ. İÇECEK KOLA 1 |": "#3300",
        "BİBER ÇARLİSTON X 0,155  X 59,90 © kg": "x9,28 —",
        "MELKTEN 2006 SÜR, Bİ": "x37, 00"
    }

    print("📷 OCR'den Gelen Ham Ürünler:")
    for urun in ocr_input:
        print(f"- {urun}")

    asistan = TarifiAsistani(api_key, [], raf_omru_dict)
    temiz_urun_list = asistan.ocr_duzenle(ocr_input)

    print("\n✅ Düzenlenmiş Ürünler:")
    for urun in temiz_urun_list:
        print(f"- {urun}")

    urunler_json = []
    for i, temiz in enumerate(temiz_urun_list, start=1):
        urunler_json.append({
            "id": i,
            "userId": 1,
            "amount": 1,
            "product": {
                "id": i,
                "name": temiz,
                "price": 0.0
            }
        })

    asistan.urunler_json = urunler_json

    # Malzemeleri ve SKT'leri oluştur
    user_ingredients = asistan.malzeme_listesini_olustur()

    print("\n📅 Malzemelerin Son Kullanma Tarihleri:")
    for u in user_ingredients:
        print(f"- {u['isim']} → SKT: {u['skt']}")

    # ✅ JSON formatında SKT çıktısı
    print("\n🗂 Malzemeler JSON Formatında:")
    print(json.dumps(user_ingredients, ensure_ascii=False, indent=4))

    # Yaklaşan SKT kontrolü
    yaklasan = asistan.skt_kontrol_et(user_ingredients)

    # Tarif öner
    tarifler = asistan.tarif_iste(user_ingredients, yaklasan)
    print("\n🍽 Önerilen Tarifler:\n", tarifler)

    # Kullanıcı 1 numaralı tarifi seçmiş gibi detayları getir
    secilen_tarif = asistan.secili_tarifi_getir(tarifler, secim=1)
    print("\n📋 Seçilen Tarif Detayları:\n", secilen_tarif)

    # Besin değeri bilgisi getir
    yemek_adi = re.search(r"1\.\s*(.?)\s\(", tarifler)
    if yemek_adi:
        yemek_adi = yemek_adi.group(1)
        besin = asistan.besin_degeri_getir(yemek_adi)
        print("\n🔥 Besin Değerleri:\n", besin)
    else:
        print("Tarif adı çıkarılamadı, besin değerleri getirilemedi.")