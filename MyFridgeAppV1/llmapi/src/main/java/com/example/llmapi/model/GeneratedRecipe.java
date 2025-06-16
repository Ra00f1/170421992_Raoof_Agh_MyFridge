package com.example.llmapi.model;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.util.List;

public class GeneratedRecipe {

    @JsonProperty("yemek_adi")
    public String yemekAdi;

    @JsonProperty("kisi_sayisi")
    public int kisiSayisi;

    @JsonProperty("hazirlik_suresi_dk")
    public int hazirlikSuresiDk;

    @JsonProperty("pisirme_suresi_dk")
    public int pisirmeSuresiDk;

    public List<String> malzemeler;
    public List<String> yapilis;
    public List<String> ipucular;

    @JsonProperty("eksik_malzemeler_aciklama")
    public String eksikMalzemeAciklama;
}
