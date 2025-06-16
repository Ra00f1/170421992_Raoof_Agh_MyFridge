package com.example.dbapi.model;

import jakarta.persistence.*;
import lombok.*;

import java.util.List;

import com.fasterxml.jackson.annotation.JsonProperty;

@Entity
@Data
@NoArgsConstructor
@AllArgsConstructor
public class Recipe 
{

    @Id
    @GeneratedValue(strategy = GenerationType.IDENTITY)
    private Long id;

    @JsonProperty("yemek_adi")
    private String yemekAdi;

    @JsonProperty("kisi_sayisi")
    private int kisiSayisi;

    @JsonProperty("hazirlik_suresi_dk")
    private int hazirlikSuresiDk;

    @JsonProperty("pisirme_suresi_dk")
    private int pisirmeSuresiDk;


    @ElementCollection
    private List<String> yapilis;

    @ElementCollection
    private List<String> ipucular;

    @ElementCollection
    private List<String> malzemeler;

    @JsonProperty("eksik_malzeme_aciklama")
    private String eksikMalzemeAciklama;
}
