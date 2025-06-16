package com.example.mainapi.model;

import lombok.*;
import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

@Data
@NoArgsConstructor
@AllArgsConstructor
public class Recipe 
{
    @JsonProperty("yemek_adi")
    private String yemekAdi;

    @JsonProperty("kisi_sayisi")
    private int kisiSayisi;

    @JsonProperty("hazirlik_suresi_dk")
    private int hazirlikSuresiDk;

    @JsonProperty("pisirme_suresi_dk")
    private int pisirmeSuresiDk;
    
    private List<String> yapilis;
    private List<String> ipucular;
    private List<String> malzemeler;

    @JsonProperty("eksik_malzeme_aciklama")
    private String eksikMalzemeAciklama;
}
