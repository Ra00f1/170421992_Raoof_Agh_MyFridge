from flask import Flask, request, jsonify
from llm import TarifiAsistani  # Sınıfı ayrı dosyada tutalım
import datetime
import json
import os

app = Flask(__name__)

# Anahtar ve raf ömrü sözlüğü burada sabitlenebilir
API_KEY = "API_KEY"
RAF_OMRU_DICT = {
    "yoğurt": 7, "peynir": 14, "yumurta": 21, "domates": 5, "çeri": 5,
    "pirinç": 180, "zeytinyağı": 365, "coca cola": 180,
    "süt": 5, "et": 3, "tavuk": 3, "balık": 2
}

# Global Asistan Nesnesi (içine her seferinde veri enjekte edeceğiz)
asistan = TarifiAsistani(API_KEY, [], RAF_OMRU_DICT)

@app.route("/ocr-düzenle", methods=["POST"])
def ocr_duzenle():
    # Try to get JSON payload
    data = request.get_json(force=True, silent=True)

    # If still string, try to parse manually
    if isinstance(data, str):
        try:
            import json
            data = json.loads(data)
        except Exception as e:
            return jsonify({"error": "Invalid JSON format", "details": str(e)}), 400

    print(f"Received data from OCR to LLM: {data}")

    # Validate structure
    if not data or "items" not in data or "userid" not in data:
        return jsonify({"error": "Missing 'items' or 'userid' in request"}), 400

    try:
        # Step 1: Get cleaned item names from LLM
        temiz_liste = asistan.ocr_duzenle(data["items"])

        # Step 2: Replace original names with cleaned names
        updated_data = replace_item_names(data, temiz_liste)

        # Step 3: Add estimated expiration date instead of old date
        for item in updated_data["items"]:
            urun_adi = item[0]
            raf_omru = asistan.tahmini_raf_omru_al(urun_adi)

            # Calculate new date: today + shelf life
            new_date = (datetime.datetime.now() + datetime.timedelta(days=raf_omru)).strftime("%d/%m/%Y")

            # Replace last element with the new date
            if len(item) >= 5:
                item[-1] = new_date
            else:
                item.append(new_date)

        return jsonify(updated_data)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


def replace_item_names(original_data, temiz_urunler):
    items = original_data["items"]

    if len(items) != len(temiz_urunler):
        print("Warning: Mismatch between original items and cleaned names count.")
        print(f"Original items: {len(items)}, Cleaned names: {len(temiz_urunler)}")
        print("Falling back to original item names where necessary.")

    # Replace item names, but safely
    for i in range(len(items)):
        if i < len(temiz_urunler):
            items[i][0] = temiz_urunler[i]
        # else: keep original name

    return {
        "userid": original_data["userid"],
        "items": items
    }


@app.route("/tarif_oner", methods=["POST"])
def tarif_oner():
    data = request.get_json()
    print(data)

    # if not data or "urunler" not in data:
    #     return jsonify({"error": "Missing urunler"}), 400

    yaklasan = asistan.skt_kontrol_et(data)
    print(yaklasan)

    tarif_text = asistan.tarif_iste(data, yaklasan)
    print(tarif_text)

    return jsonify({"tarif_text": tarif_text})

@app.route("/tarif_detay", methods=["POST"])
def tarif_detay():
    data = request.get_json()
    if not data or "tarifler_text" not in data or "secim" not in data:
        return jsonify({"error": "Missing tarifler_text or secim"}), 400

    detay = asistan.secili_tarifi_getir(data["tarifler_text"], data["secim"])
    print(f"Detay: {detay}")
    return jsonify({"detay": detay})

@app.route("/besin-degeri", methods=["POST"])
def besin_degeri():
    data = request.get_json()
    if not data or "yemek_adi" not in data:
        return jsonify({"error": "Missing yemek_adi"}), 400

    kisi_sayisi = data.get("kisi_sayisi", 4)
    besin = asistan.besin_degeri_getir(data["yemek_adi"], kisi_sayisi)
    return jsonify({"besin": besin})

@app.route("/ping", methods=["GET"])
def ping():
    return "pong"

if __name__ == '__main__':

    app.run(host="0.0.0.0", port=5001,debug=True)
